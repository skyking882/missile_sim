#!/usr/bin/env python3
"""ζ 判别脚本：只读 02_model/*.csv，不 import 本仓库任何模块。

回答两个问题：
  A. 力矩恒等式 I·ω̇ = K(δ−α) − C·ω 在 CSV 里逐点闭合吗？
     —— 闭合 ⇒ ζ 就是 plant 的阻尼比，不是别的东西的替身。
  B. 在稳态转弯段（ω̇≈0），实测 (δ−α) 与 2ω/ω_n 的比值是多少？
     —— 这就是 zeta_implied；对 baseline 应≈1（自洽），对游戏回放需要≈0.5。

用法：  python3 check_plant_identity.py ../02_model/a7_baseline_dt0.020.csv
"""
from __future__ import annotations
import csv, math, sys

def load(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return [{k: (v if k == "pcc_capture_mode" else (float(v) if v else float("nan")))
                 for k, v in r.items()} for r in csv.DictReader(fh)]

def steady_window(rows):
    """自动挑稳态转弯段：发射后第一个连续 CAPTURE 区间内，捕获律仍饱和(R>=1)
    且角加速度已经收敛的行。固定时间窗对不同变体是错的 —— ζ 变了释放时刻就变了；
    而只按 mode=="capture" 过滤又会混进中段 recapture。"""
    # 2026-08-26 复审修正：只取【发射后第一个连续 CAPTURE 区间】，在首次
    # CAPTURE->HOMING 处永久截断。旧写法会把中段 recapture 的行也收进来
    # （粗步长 three-step fix 的 4.14 s 第二次捕获就是这样混进"稳态窗"的）。
    first = []
    seen_capture = False
    for r in rows:
        if r["pcc_capture_mode"] == "capture":
            seen_capture = True
            first.append(r)
        elif seen_capture:
            break                      # 首次交接 -> 永久截断
    return [r for r in first
            if r["time_s"] > 0.8
            and r["pcc_capture_ratio"] >= 1.0
            and abs(r["yaw_angular_accel_rad_s2"]) < 0.20]

def main(path):
    rows = load(path)
    dt = rows[1]["time_s"] - rows[0]["time_s"]
    win = steady_window(rows)
    if not win:
        print("找不到稳态捕获段"); return
    t0, t1 = win[0]["time_s"], win[-1]["time_s"]
    print(f"file: {path}\nrows: {len(rows)}   dt = {dt:.4f} s")
    print(f"自动选出的稳态转弯段: t = {t0:.2f} .. {t1:.2f} s  ({len(win)} 行)\n")
    keep = {round(r["time_s"], 6) for r in win}

    # ---- A. 力矩恒等式 -----------------------------------------------------
    print("A. I·ω̇ = K(δ−α) − C·ω   逐点闭合检查（yaw 轴，捕获平台段）")
    print(f"{'t':>6} {'ω_n':>7} {'K(δ−α)':>10} {'−C·ω':>10} {'和':>10} "
          f"{'I·ω̇(CSV)':>10} {'残差':>9}")
    worst = 0.0
    for r in win:
        t = r["time_s"]
        wn = r["yaw_natural_frequency_rad_s"]
        K = wn * wn                                  # ω_n² ，单位 1/s²
        spring = K * (r["actual_yaw_fin_angle_rad"] - r["yaw_alpha_rad"])
        damp = -r["yaw_residual_rate_damping_per_s"] * r["yaw_rate_rad_s"]
        lhs = r["yaw_angular_accel_rad_s2"]
        res = spring + damp - lhs
        worst = max(worst, abs(res))
        if abs(t * 5 - round(t * 5)) < 1e-6 or r is win[-1]:
            print(f"{t:6.2f} {wn:7.3f} {spring:10.4f} {damp:10.4f} "
                  f"{spring+damp:10.4f} {lhs:10.4f} {res:9.2e}")
    print(f"\n  稳态段最大残差 |Σ − I·ω̇| = {worst:.3e} rad/s²")
    print("  （非零只可能来自 unit-disk 耦合：两轴合成超过舵盘时 moment_fraction 被径向裁剪）\n")

    # ---- B. 稳态段的隐含 ζ -------------------------------------------------
    print("B. 稳态转弯段的隐含 ζ = (δ−α) / (2ω/ω_n)")
    print(f"{'t':>6} {'|δ|°':>7} {'α°':>7} {'亏损°':>7} {'ω °/s':>8} {'ω̇ rad/s²':>10} "
          f"{'ζ_implied':>10} {'mode':>8}")
    sel = []
    for r in win:
        t = r["time_s"]
        z = r["zeta_implied"]; sel.append(z)
        if abs(t * 5 - round(t * 5)) > 1e-6 and r is not win[-1]: continue
        print(f"{t:6.2f} {math.degrees(r['delta_total_rad']):7.2f} "
              f"{math.degrees(r['angle_of_attack_rad']):7.2f} "
              f"{math.degrees(r['trim_deficit_rad']):7.2f} "
              f"{math.degrees(r['omega_total_rad_s']):8.2f} "
              f"{r['yaw_angular_accel_rad_s2']:10.4f} {z:10.3f} {r['pcc_capture_mode']:>8}")
    if sel:
        print(f"\n  稳态段 ζ_implied 均值 = {sum(sel)/len(sel):.3f}  "
              f"（范围 {min(sel):.3f} .. {max(sel):.3f}）")

    # ---- C. 被丢弃的真实气动阻尼有多大 ------------------------------------
    print("\nC. 真实气动阻尼 ζ_aero = ω_n·Δ/(2V) ，与代码里那个 ζ 的比值")
    print(f"{'t':>6} {'ω_n':>7} {'V m/s':>8} {'ωΔ/V rad':>10} {'ζ_aero':>10} {'倍数':>9}")
    for r in win:
        t = r["time_s"]
        if abs(t * 5 - round(t * 5)) > 1e-6: continue
        wn = r["yaw_natural_frequency_rad_s"]; V = r["speed_mps"]
        w = r["yaw_rate_rad_s"]
        # yaw_tail_rate_incidence_rad 就是 ω·Δ/V ，直接从 CSV 读，不用自己乘
        rate_inc = r["yaw_tail_rate_incidence_rad"]
        arm = rate_inc * V / w if abs(w) > 1e-9 else float("nan")   # 反解 Δ 自检
        z_aero = wn * arm / (2.0 * V)
        z_code = r["yaw_residual_rate_damping_per_s"] / (2.0 * wn)
        print(f"{t:6.2f} {wn:7.3f} {V:8.1f} {rate_inc:10.6f} {z_aero:10.5f} "
              f"{z_code/z_aero:9.0f}x")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../02_model/a7_baseline_dt0.020.csv")
