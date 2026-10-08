"""Label every existing output column without changing any stored cell value."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'outputs'/'extended_202944_final'
UNK=ROOT/'outputs'/'unknown_202944'
OUT=ROOT/'outputs'/'classified_202944'
REG={}

def add(names,tag,label,unit,method):
    for name in names.split():
        assert name not in REG,name
        REG[name]={'tag':tag,'meaning':label,'unit':unit,'basis':method}

add('replay_time_ms','READ','回放包时间','ms','解析器读取的回放时间；不是导弹自身飞行时钟')
add('replay_time_s','CALC','回放包时间换算','s','replay_time_ms / 1000')
add('since_entity_created_s','CALC','相对实体创建事件时间','s','(replay_time_ms - created_at_ms) / 1000')
add('dt_s','CALC','相邻导弹采样时间间隔','s','相邻 replay_time_ms 之差 / 1000；首行为 NA')
add('simulation_time_s_candidate','EST','候选模拟时钟','s','b5 / 48；48 Hz 的时间含义仍属候选解释')
add('flight_time_sf1_s_candidate','CAND_READ','候选飞行时长','s 候选','复制上游解码的 sf1；将其命名为飞行时长尚未独立确认')
add('x_m y_m z_m','READ','导弹世界坐标 XYZ','m','回放位置字段解码；Y 为向上轴')
add('roll_rad_decoded heading_rad_decoded pitch_rad_decoded','READ','导弹滚转、航向、俯仰','rad','回放姿态字段经上游解码为欧拉角；按 Dagor 轴顺序解释')
add('roll_deg heading_deg pitch_deg','CALC','滚转、航向、俯仰角度换算','deg','对应弧度值 × 180 / pi')
add('derived_world_speed_mps derived_world_speed_kmh','EST','相邻点差分速率','m/s 或 km/h','相邻采样位置的直线距离 / 回放时间间隔；km/h 再乘 3.6')
add('sampled_path_length_m','EST','累计采样折线长度','m','累加相邻导弹位置直线距离；不是连续真实航迹的精确长度')
add('velocity_x_mps_candidate velocity_y_mps_candidate velocity_z_mps_candidate','EST','候选导弹世界速度 XYZ','m/s 候选','signed16(b7_raw_u16) × 4000 / 65535；符号与比例解释仍为候选')
add('speed_mps_candidate speed_kmh_candidate','EST','候选导弹速率','m/s 或 km/h','候选速度向量的模；km/h 再乘 3.6；不是 IAS')
add('path_speed_mps_estimate','EST','中心差分速率（回放时间）','m/s','坐标中心差分 / 回放时间差，首尾单边差分')
add('path_speed_simulation_clock_mps_estimate','EST','中心差分速率（候选模拟时钟）','m/s','坐标中心差分 / 候选模拟时间差，首尾单边差分')
add('flight_path_angle_deg_estimate','EST','速度方向爬升/下俯角','deg','atan2(v_y, hypot(v_x,v_z))；依赖候选速度')
add('aoa_deg_estimate_zero_wind','EST','有符号攻角 AoA','deg','零风假设下将候选速度转到弹体系，atan2(-v_body_y,v_body_x)')
add('sideslip_deg_estimate_zero_wind','EST','侧滑角','deg','零风假设下 atan2(v_body_z,hypot(v_body_x,v_body_y))')
add('total_incidence_deg_estimate_zero_wind','EST','弹轴与速度的三维总夹角','deg','零风假设下 angle(弹轴,候选速度)；非负，不等于有符号 AoA')
add('aoa_from_position_difference_deg_estimate','EST','位置差分法攻角','deg','用回放时间坐标差分速度替代 b7 候选速度，再按同一弹体系角度公式计算；同样假设零风')
add('target_x_m_interpolated target_y_m_interpolated target_z_m_interpolated','EST','目标在导弹采样时刻的位置 XYZ','m','在目标原始采样点间线性插值；没有外推')
add('target_eid','META','本次对照目标实体号','整数','实体号本身可读；选择 TargetAI_0 作为对照目标是本次分析指定，不是已解码的锁定目标 ID')
add('target_bracketing_gap_ms','CALC','目标插值所用两采样点的间隔','ms','目标两条原始记录时间相减')
add('target_slant_distance_m','EST','三维弹目距离','m','norm(目标插值位置 - 导弹位置)；不是原生距离字段，也不是脱靶量')
add('target_horizontal_distance_m','EST','水平弹目距离','m','hypot(相对 X,相对 Z)；使用目标插值位置')
add('target_height_difference_m','EST','目标相对高度','m','目标插值 Y - 导弹 Y')
add('target_closing_speed_mps_estimate','EST','视线方向接近速度','m/s','-dot(相对位置,目标差分速度 - 导弹候选速度) / 距离；正值为接近')
add('target_off_boresight_deg_estimate','EST','目标离弹轴角','deg','angle(弹轴,目标插值位置 - 导弹位置)；不是 AoA')
add('target_velocity_vector_angle_deg_estimate','EST','导弹速度与目标视线夹角','deg','angle(导弹候选速度,目标插值位置 - 导弹位置)；字段名中的 velocity 不是目标速度')
add('shooter_distance_m','EST','导弹距发射机距离','m','发射机位置线性插值后与导弹位置求距离')
add('kinematic_tangential_accel_mps2_estimate','EST','切向运动学加速度','m/s2','候选速度按候选模拟时钟差分，再投影到速度方向')
add('kinematic_normal_accel_g_estimate','EST','法向运动学加速度','g','速度差分加速度去掉切向分量后取模 / 9.80665；不是气动过载或制导指令')
add('angular_rate_x_candidate angular_rate_y_candidate angular_rate_z_candidate','EST','候选角速度 XYZ','标度30，单位未确认','signed16(b8_raw_u16) × 30 / 65535；不能默认轴系和 rad/s 已证实')
add('raw_b5','READ','b5 原始计数','未知','保留实际解码的计数；与候选模拟时钟分开')
for group,scale in [('b7',4000),('b8',30)]:
    add(' '.join(f'raw_{group}_{a}_raw_u16' for a in 'xyz'),'READ',f'{group} 的三个原始无符号整数','u16','保留原整数；不是已确认的物理速度/角速度')
    add(' '.join(f'raw_{group}_{a}_upstream' for a in 'xyz'),'CALC',f'上游对 {group} 的无符号比例换算','物理单位未确认',f'unsigned16(raw) × {scale} / 65535；尽管原列名以 raw 开头，数值已换算；负分量可能发生回绕')
add('raw_bool1 raw_bool2 raw_position_uncompressed raw_sf_flags raw_vv raw_vv2','READ','未解释的分支标志或数值','未知','从同步字段解码，保留原名；本回放 sf_flags=15，其余这些字段为0')
add('raw_sf1 raw_sf2 raw_sf3 raw_sf4','READ','上游 sf1–sf4 解码结果','上游按秒标度，含义未定','上游已按其格式读取/解码，保留数值；sf2–sf4 在本回放全0')
add('raw_euler_x raw_euler_y raw_euler_z','READ','原名保留的姿态欧拉角','rad','分别与 roll_rad_decoded/heading_rad_decoded/pitch_rad_decoded 重复')
add('raw_skipped_bits','READ','同步未知区长度','bit','从同步包读取的长度字段，本回放为607')
add('raw_start_bit raw_end_bit raw_seeker_start_bit','META','解析器记录的位偏移','bit','解析位置，不是导弹遥测字段')
add('sample_index','META','输出样本序号','整数','本地生成的0起始行号，不是回放载荷')
add('prefix_11_bits_raw','READ','未知区开头11位','bit string','原始位值保留；内部结构和标志含义未确认')
add('track_position_x_m_candidate track_position_y_m_candidate track_position_z_m_candidate','CAND_READ','候选目标跟踪位置 XYZ','m 候选','未知区位11起按连续3个小端float32读取；布局与位置含义为候选')
add('track_velocity_x_mps_candidate track_velocity_y_mps_candidate track_velocity_z_mps_candidate','CAND_READ','候选目标跟踪速度 XYZ','m/s 候选','未知区位107起按连续3个小端float32读取；布局与速度含义为候选')
add('unknown_float_at_bit_203 unknown_float_at_bit_235 unknown_float_at_bit_267','CAND_READ','未知的三个浮点候选','未知','未知区所示位偏移读取float32；没有赋予加速度或其他物理含义')
add('target_velocity_x_mps_from_position target_velocity_y_mps_from_position target_velocity_z_mps_from_position','EST','目标轨迹差分速度 XYZ','m/s','使用包围当前导弹时刻的目标位置采样点作差分')
add('track_position_error_to_target_m_candidate','EST','候选跟踪位置与目标插值位置之差','m','两个位置向量作差取模；只是对照差，不是已确认的传感器误差')
add('track_velocity_error_to_target_mps_candidate','EST','候选跟踪速度与目标差分速度之差','m/s','两个速度向量作差取模；不是已确认的传感器误差')
add('track_range_from_missile_m_candidate','EST','候选跟踪点到导弹距离','m','候选跟踪位置与导弹位置作差取模；不能代替目标轨迹距离')
add('remaining_bits_from_299_count','META','未解释尾部位数','bit','607-299=308，是本地拆分长度')
add('remaining_bits_from_299_hex_right_zero_padded','READ','未知区尾部原始位流','hex','保留第299位起的308位；存储最后字节右补零，补零不属于回放')

def file_hash(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def stats(rows,key):
    values=[r[key] for r in rows if r[key]!='NA']
    result={'first':rows[0][key],'last':rows[-1][key],'missing_count':sum(r[key]=='NA' for r in rows)}
    if 'hex' in key or 'prefix' in key:
        result['distinct_count']=len(set(values))
        return result
    try:
        numbers=[float(v) for v in values]
        result.update({'min':min(numbers),'max':max(numbers)})
    except ValueError:
        result['distinct_count']=len(set(values))
    return result

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw=json.loads((SRC/'raw_extended.json').read_text(encoding='utf-8'))
    assert Path(raw['source']).name=='#2026.09.27 20.29.44.wrpl'
    assert len(raw['rockets'])==1 and raw['rockets'][0]['eid']==32770
    assert raw['rockets'][0]['weapon_name']=='pi28_r771_h_110'
    paths=[SRC/'missile_all_fields.dat',UNK/'seeker_candidates.dat']
    before={str(p):file_hash(p) for p in paths}
    catalog={'scope':'Existing extracted and calculated fields for the specified one-missile replay; not all possible WRPL data', 'source':raw['source'],'source_sha256':raw['source_sha256'],'labels':{'READ':'实际字段经解析器解码或保留原始位值；含义可能未知','CAND_READ':'从位流读取的候选数值，结构/物理含义未完全验证','CALC':'时间、角度等直接换算或精确差值；不是独立测量','EST':'差分、插值、几何推算，或依赖候选标度/时钟的估计','META':'解析器元数据或分析指定信息，不是导弹物理量'},'tables':{}}
    checks=[]
    for path in paths:
        with path.open(encoding='utf-8',newline='') as f:
            reader=csv.DictReader(f,delimiter='\t');cols=reader.fieldnames;rows=list(reader)
        assert all(c in REG for c in cols),set(cols)-REG.keys()
        renamed={c:REG[c]['tag']+'__'+c for c in cols}
        dest=OUT/(path.stem+'_labeled.dat')
        with dest.open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(renamed.values()),delimiter='\t');writer.writeheader()
            writer.writerows({renamed[k]:v for k,v in r.items()} for r in rows)
        with dest.open(encoding='utf-8',newline='') as f:copy=list(csv.DictReader(f,delimiter='\t'))
        assert len(rows)==len(copy) and all(all(r[k]==c[renamed[k]] for k in cols) for r,c in zip(rows,copy))
        catalog['tables'][path.name]={'row_count':len(rows),'column_count':len(cols),'labeled_file':str(dest),'classification_counts':dict(Counter(REG[c]['tag'] for c in cols)),'fields':[{ 'original_name':c,'labeled_name':renamed[c],**REG[c],'observed':stats(rows,c)} for c in cols]}
        checks.append({'file':path.name,'all_columns_classified':True,'all_cell_text_unchanged':True,'row_count':len(rows),'column_count':len(cols)})
    # Preserve all other extracted fields, with physical naming caveats separate.
    catalog['other_extracted']={'rocket':{k:v for k,v in raw['rockets'][0].items() if k!='track'},'units':[{k:v for k,v in u.items() if k!='track'} for u in raw['units']], 'unit_track_fields':{'t_ms':'READ: 回放时间 ms','xyz':'READ: 世界坐标 XYZ m','euler_raw':'READ: 欧拉角 XYZ rad'},'unit_track_samples':{u['player_internal_name']:len(u['track']) for u in raw['units']},'component_snapshots':raw['missile_component_snapshots'],'component_caveat':'解析后的实体组件状态可能含模板默认值；没有逐项证明每个组件值都是该包字面载荷。active=false 不作为飞行或开机状态结论。', 'raw_fragment_container_fields':{'fields':'已有主表字段及 eid/time_ms/x/y/z','byte_slice_hex':'READ: 完整同步片段字节','first_valid_bit_offset_in_slice':'META:首有效位偏移','valid_bit_count':'META:有效位数','chunks.bit_count':'READ:未知区长度607','chunks.hex_right_zero_padded':'READ:完整未知区原始位流，末字节右补零'},'administrative_fields':['source','source_sha256','backend_revision','capture_method','capture_script_sha256','replay_length_ms']}
    assert before=={str(p):file_hash(p) for p in paths}
    (OUT/'field_catalog.json').write_text(json.dumps(catalog,indent=2,ensure_ascii=False),encoding='utf-8')
    (OUT/'classification_qa.json').write_text(json.dumps({'checks':checks,'source_tables_unchanged':True,'source_table_hashes':before},indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# 全部已导出字段分类','',f"来源：`{Path(raw['source']).name}`；导弹 `pi28_r771_h_110 / 0x00008002`。",'', '只给列名加分类前缀，没有改动任何数据单元格。所有 EST 均为估计/推算，CAND_READ 的数值来自位流但物理含义未定。READ 表示经格式解码，不保证是未压缩的原始 float。','']
    for tag,desc in catalog['labels'].items():lines.append(f'- **{tag}**：{desc}')
    for name,t in catalog['tables'].items():
        lines+=['',f'## {name}：{t["row_count"]} 行、{t["column_count"]} 列','', '| 原列名 | 标签 | 含义 | 单位 | 计算/读取依据 | 本次最小值～最大值 |','|---|---|---|---|---|---|']
        for f in t['fields']:
            o=f['observed'];span=f"{o['min']:.9g} ～ {o['max']:.9g}" if 'min' in o and 'hex' not in f['original_name'] and 'prefix' not in f['original_name'] else f"首值 `{o['first']}`"
            lines.append(f"| `{f['original_name']}` | **{f['tag']}** | {f['meaning']} | {f['unit']} | {f['basis']} | {span} |")
    lines+=['','## 创建记录、实体身份与飞机原始轨迹','', '完整原始值与组件清单保存于同目录 `field_catalog.json` 的 `other_extracted`。创建字段均保留上游名称；候选解释不改写数值。','', '| 创建字段 | 实际值 | 解释状态 |','|---|---|---|']
    notes={'starting_pos':'初始世界位置 m','u12_2':'候选初始速度；数值实际解码，含义交叉检查支持','u12_3':'候选初始角速度；轴系/单位未证实','u12_4':'候选继承/发射机速度；尚未确认','u12_5':'候选初始姿态向量；尚未正式确认','u16_1':'与初始姿态匹配的四元数候选','creation_time':'上游命名的创建时钟；与回放包时间不同','weapon_ref':'武器引用，不是目标ID','maybe_sensor_info':'817 bit 未解释位流，部分与首个同步片段相同','some_weap_type_info':'空位流','bomb_info':'空位流'}
    for k,v in raw['rockets'][0]['creation_raw_fields'].items():
        value=json.dumps(v,ensure_ascii=False) if not isinstance(v,dict) else f"{v['bit_count']} bit；原始 hex 见 JSON"
        lines.append(f'| `{k}` | `{value}` | {notes.get(k,"实际字段解码，物理含义未知")} |')
    lines+=['','三个飞机实体各有 51 个原始时间、XYZ 坐标及 XYZ 欧拉角采样。身份、出生矩阵、owner_pid、创建/死亡/销毁事件时间全部保存在 JSON。4294967295 是未记录相应事件的哨兵，不能视作实际死亡或销毁时刻。','',catalog['other_extracted']['component_caveat'],'','`weapon_name`、实体模板名和组件类型来自解析与加载配置的关联；`target_eid` 是本次对照目标选择；它们不能用于声称已读到导引头锁定目标。','', '原始片段容器和创建字段的全部值均保留，统计量只用于展示，不是额外遥测字段。']
    lines+=['','## 其他已提取字段的逐项清单','', '| 对象 | 字段 | 值/内容 |','|---|---|---|']
    for k,v in catalog['other_extracted']['rocket'].items():
        if k!='creation_raw_fields':lines.append(f'| 导弹 | `{k}` | `{v}` |')
    for u in catalog['other_extracted']['units']:
        for k,v in u.items():lines.append(f'| {u["player_internal_name"]} | `{k}` | `{v}` |')
    for snap in raw['missile_component_snapshots']:
        for k in ['time_ms','eid','template']:lines.append(f'| 组件快照 | `{k}` | `{snap[k]}` |')
        for k,v in snap['components'].items():
            content='与上方创建字段重复' if k=='rocket_component' else json.dumps(v,ensure_ascii=False)
            lines.append(f'| 组件快照 | `{k}` | `{content}` |')
    for k,v in catalog['other_extracted']['raw_fragment_container_fields'].items():lines.append(f'| 原始位流容器 | `{k}` | {v} |')
    for k in catalog['other_extracted']['administrative_fields']:lines.append(f'| 文件/解析元数据 | `{k}` | `{raw[k]}` |')
    (OUT/'field_catalog.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({n:{k:t[k] for k in ['row_count','column_count','classification_counts']} for n,t in catalog['tables'].items()},indent=2))

if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source', type=Path, default=SRC, help='Extended export directory for the validated 2026-09-27 single-missile replay')
    cli.add_argument('--candidates', type=Path, default=UNK, help='Output from inspect_unknown_chunks.py')
    cli.add_argument('--output', type=Path, default=OUT)
    args=cli.parse_args()
    SRC=args.source.resolve()
    UNK=args.candidates.resolve()
    OUT=args.output.resolve()
    main()
