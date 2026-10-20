#!/usr/bin/env python3
"""Read-only checks for game-replay PN comparisons; not independent game truth."""
from __future__ import annotations
import argparse,math,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'audits/paper_pn_20261020'))
import compare as A
from aim120_model.effective_surface_core import EffectiveSurfaceGame

def cross(a,b):
    return np.array([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])
def dot(a,b):return math.fsum(float(x)*float(y) for x,y in zip(a,b))
def norm(a):return math.sqrt(dot(a,a))

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--commands',type=Path,required=True);p.add_argument('--flights',type=Path,required=True);p.add_argument('--out',type=Path,required=True);z=p.parse_args();z.out.mkdir(parents=True,exist_ok=False)
    checked=A.verify(z.input);source=A.read(z.input/'fresh.json.gz');original={(x['replay'],x['record']['index']):x['record'] for x in source};rows=A.read(z.commands/'rows.json.gz')
    cfg=A.read(A.ROOT/'config/effective_surface_defaults.json');profile=A.read(A.ROOT/'data/profile_sets/2.59.0.27/missiles/su_r_77_1.json')
    assert cfg['pid_error_frame']=='velocity_normal_projected_body'
    model=SimpleNamespace(config=cfg,autopilot={'reqAccelMax':profile['guidance']['maximum_lateral_acceleration_g']})
    maximum={'paper_lambda_deg_s':0.,'raw_command_mps2':0.,'limited_delta_g':0.};checks=0
    for row in rows:
        r=original[row['replay'],row['index']];assert r['raw_sha']==row['raw_sha'] and r['tick']==row['tick']
        n=np.array(r['n'],float);n/=norm(n);w=np.array(r['w'],float);ndot=cross(n,w)
        # Independent spherical-azimuth derivative, not the vector-pole helper.
        horizontal2=n[0]*n[0]+n[2]*n[2]
        if horizontal2<=1e-8:raise AssertionError('Unexpected chart singularity in retained records')
        azimuth_rate=(n[0]*ndot[2]-n[2]*ndot[0])/horizontal2
        lam=azimuth_rate*n[1];u=w-dot(n,w)*n;paper=u+lam*n
        maximum['paper_lambda_deg_s']=max(maximum['paper_lambda_deg_s'],abs(lam*A.D-row['lambda_paper_deg_s']))
        v=np.array([r['R']*x-r['closing']*y for x,y in zip(ndot,n)])
        full=row['gain']*cross(w,v);other=row['gain']*cross(paper,v)
        maximum['raw_command_mps2']=max(maximum['raw_command_mps2'],norm(full-row['command_full_native']),norm(other-row['command_paper_native']))
        state=(0.,0.,0.,*(A.C@np.array(r['vm'])))
        f=EffectiveSurfaceGame.limit_acceleration_request(model,tuple(A.C@full),state)
        g=EffectiveSurfaceGame.limit_acceleration_request(model,tuple(A.C@other),state)
        maximum['limited_delta_g']=max(maximum['limited_delta_g'],abs(norm(np.array(f)-g)/A.G-row['paper_limited_difference_g']))
        checks+=1
    assert maximum['paper_lambda_deg_s']<1e-9 and maximum['raw_command_mps2']<1e-7 and maximum['limited_delta_g']<1e-8,maximum
    anchors=A.read(z.input/'anchors.json.gz');summary=A.read(z.flights/'summary.json');flight_checks=0;target_difference=0.;score_difference=0.;intermodel={}
    for uid,case in A.CASES.items():
        ref={r['tick']:r for r in anchors[uid]};base=A.read(z.flights/f'{uid}_full_4.json.gz');intermodel[uid]={}
        for item in summary['cases'][uid]['models']:
            kind=item['kind'];tr=A.read(z.flights/f'{uid}_{kind}_4.json.gz');assert len(base)==len(tr)
            for key in ('position','velocity','quaternion_xyzw','omega_body'):assert tr[0][key]==base[0][key]
            distances=[]
            for a,b in zip(base,tr):
                assert a['tick']==b['tick'] and a['age']==b['age']
                for key in ('track_position','track_velocity'):target_difference=max(target_difference,norm(np.array(a[key])-b[key]))
                distances.append(norm(np.array(a['position'])-b['position']))
                if 'score' in b:
                    r=ref[b['tick']];assert r['raw_sha']==b['score']['raw_sha']
                    dp=np.array(b['position'])-A.C@np.array(r['p']);score_difference=max(score_difference,abs(norm(dp)-b['score']['position_error_m']));flight_checks+=1
            expected=summary['cases'][uid]['intermodel'][kind]['max_position_difference_from_full_m'];assert abs(max(distances)-expected)<1e-8
            intermodel[uid][kind]=dict(maximum_difference_from_full_m=max(distances),terminal_difference_from_full_m=distances[-1],terminal_difference_from_full_xyz_m=np.array(tr[-1]['position'])-base[-1]['position'])
    assert target_difference<1e-7 and score_difference<1e-8,(target_difference,score_difference)
    receipt=dict(scope='Independent scalar/angle expressions and unchanged production limiter; same game-state data, not independent investigators, new replays or direct game-command measurements.',input_files_verified=checked,command_source_row_checks=checks,maximum_arithmetic_differences=maximum,flight_reference_score_checks=flight_checks,maximum_reference_score_difference_m=score_difference,maximum_target_input_difference_across_guidance_variants=target_difference,initial_own_state_identical=True,intermodel=intermodel,production_limiter_contract=cfg['pid_error_frame'],scope_limit='Commands are PN-only. Flights retain the existing loft logic, controller, and plant. Controller memory is unknown and initialized identically to zero.')
    A.save(z.out/'verification.json',receipt)
    commands=A.read(z.commands/'summary.json')
    lines=['# 实际执行的论文PN对照','','## 相同输入的指令差异','','单位：角速度deg/s，加速度g。下表是同一已有PN计算使用不同rate的结果，不是游戏直接记录的请求。投影／限幅使用冻结生产配置。','','| 样本 | 记录数 | 轴向差RMS | 原始请求差RMS | 投影限幅后差RMS | 请求夹角中位数 |','|---|---:|---:|---:|---:|---:|']
    for name,s in commands['groups'].items():
        m=s['metrics'];lines.append(f"| {name} | {s['n']} | {m['lambda_difference_deg_s']['rms']:.6f} | {m['paper_difference_g']['rms']:.6f} | {m['paper_limited_difference_g']['rms']:.6f} | {m['paper_angle_deg']['median']:.6f} |")
    lines+=['','## 条件TRK窗口自由积分','','每段只初始化一次位置/速度/姿态；后续不重置。只替换制导端rate，原始完整rate仍用于共同回放适配器。最大误差按共同原始保存节点评分。','','| 案例 | rate选择 | 最大3D偏差m | 末点3D偏差m | 最大横向偏差m | 最大速率差m/s | 末点高度差m |','|---|---|---:|---:|---:|---:|---:|']
    for uid,c in summary['cases'].items():
        for r in c['models']:lines.append(f"| {uid} | {r['kind']} | {r['maximum_position_error_m']:.6f} | {r['terminal']['position_error_m']:.6f} | {r['maximum_cross_track_m']:.6f} | {r['maximum_abs_speed_error_mps']:.6f} | {r['terminal']['position_error_xyz_runtime_m'][1]:.6f} |")
    lines+=['','## 积分敏感性','','| 案例 | rate选择 | 4到8子步的最大轨迹差m |','|---|---|---:|']
    for uid,c in summary['cases'].items():
        for r in c['sensitivity']:lines.append(f"| {uid} | {r['kind']} | {r['max_position_change_from_4substeps_m']:.9f} |")
    lines+=['','## 解释边界','','论文Eq8c/9a的球坐标自旋依赖固定极轴，不是任意非零lambda。主结果取世界竖直；两个固定初始化极轴候选全部保留，不按case选择最优。三种角状态给出的制导不同不代表论文不成立，也不能证明游戏源代码的制导端采用了哪一种。相同PID/气动/回放输入下，完整记录rate更接近游戏轨迹，是条件反演证据。原始输入未独立生成、PI记忆未知、基础模型仍有误差，不能宣称已确定游戏算法或bug。']
    (z.out/'TABLES.md').write_text('\n'.join(lines)+'\n')
    print('INDEPENDENT_REVIEW',json.dumps(A.clean(receipt)),flush=True);print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
