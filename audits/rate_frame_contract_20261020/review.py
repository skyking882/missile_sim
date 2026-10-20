#!/usr/bin/env python3
"""Independent world-ODE check of exact moving-frame formulas; not AI delegation."""
from __future__ import annotations
import argparse,math,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import check_frames as C

KINDS=('world','inertial_body','relative_body')
def cross(a,b):return np.array([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])
def rotation(rotvec):
    v=np.asarray(rotvec,float);x=math.sqrt(math.fsum(float(z)*float(z) for z in v))
    if x<1e-15:return np.eye(3)
    a=v/x;k=np.array([[0.,-a[2],a[1]],[a[2],0.,-a[0]],[-a[1],a[0],0.]])
    return np.eye(3)+math.sin(x)*k+(1-math.cos(x))*(k@k)

def rk_step(n,w,q0,omega,dt,kind,substeps):
    h=dt/substeps;y=np.array(n,float)
    def f(t,x):
        if kind=='world':rate=w
        else:
            rate=rotation(omega*t)@q0@w
            if kind=='relative_body':rate=rate-omega
        return cross(x,rate)
    for j in range(substeps):
        t=j*h;k1=f(t,y);k2=f(t+h/2,y+h*k1/2);k3=f(t+h/2,y+h*k2/2);k4=f(t+h,y+h*k3)
        y=y+h*(k1+2*k2+2*k3+k4)/6
    return C.unit(y)

def analytic_tests():
    rng=np.random.default_rng(20261020);checks=0;maxdiff=0.;cov=[]
    for _ in range(80):
        n=C.unit(rng.normal(size=3));w=rng.normal(size=3)*.3;q0=Rotation.random(random_state=rng).as_matrix();omega=rng.normal(size=3);dt=.04;q1=rotation(omega*dt)@q0;s=Rotation.random(random_state=rng).as_matrix()
        np.testing.assert_allclose(rotation(omega*dt),Rotation.from_rotvec(omega*dt).as_matrix(),atol=1e-14,rtol=0)
        for kind in KINDS:
            exact=C.exact_step(n,w,q0,q1,dt,kind);num=rk_step(n,w,q0,omega,dt,kind,16);err=C.angle(exact,num);maxdiff=max(maxdiff,err);assert err<1e-6
            rotated=C.exact_step(s@n,s@w if kind=='world' else w,s@q0,s@q1,dt,kind)
            np.testing.assert_allclose(rotated,s@exact,atol=3e-14,rtol=0)
            checks+=1
        # Coherent coordinate roundtrip vs multiplying a world vector by Q.
        np.testing.assert_allclose(q0@(q0.T@w),w,atol=1e-14,rtol=0)
        cov.append(float(np.linalg.norm((s@q0)@(s@w)-s@(q0@w))))
    n=np.array([1.,0.,0.]);w=np.zeros(3);q0=np.eye(3);dt=.2;omega=np.array([0.,0.,.5]);q1=rotation(omega*dt)
    np.testing.assert_allclose(C.exact_step(n,w,q0,q1,dt,'inertial_body'),n,atol=1e-14,rtol=0)
    np.testing.assert_allclose(C.exact_step(n,w,q0,q1,dt,'relative_body'),q1@n,atol=1e-14,rtol=0)
    np.testing.assert_allclose(C.ray([math.pi/6,0.]),[math.sqrt(3)/2,0.,-.5],atol=1e-14,rtol=0)
    return dict(random_ODE_and_covariance_checks=checks,max_synthetic_RK_error_deg=maxdiff,pure_body_motion_control_passed=True,native_ray_sign_test_passed=True,extra_Q_on_world_seed_non_covariance_norm=C.stats(cov))

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--propagation',type=Path,required=True);p.add_argument('--contract',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    verified=C.verify(a.input);eps=C.read(a.input/'episodes.json.gz');reference=C.read(a.propagation/'rows.json.gz');by={(r['episode_id'],r['tick']):r for r in reference};diff={str(k):{m:[] for m in KINDS} for k in (8,16)};metric_error=0.;rows=[]
    for e in eps:
        if not e['strict']:continue
        seed=e['seed'];w=C.vec(seed['w']);states={k:{m:C.unit(seed['n']) for m in KINDS} for k in (8,16)};exact={m:C.unit(seed['n']) for m in KINDS};prev=seed
        for r in e['held']:
            q0,q1=C.vec(prev['Q']),C.vec(r['Q']);dt=(r['tick']-prev['tick'])/48;omega=Rotation.from_matrix(q1@q0.T).as_rotvec()/dt;ref=by[e['episode_id'],r['tick']];assert ref['raw_sha']==r['raw_sha']
            for m in KINDS:
                exact[m]=C.exact_step(exact[m],w,q0,q1,dt,m)
                metric_error=max(metric_error,abs(C.angle(exact[m],r['n'])-ref['errors_deg'][m]))
                for k in (8,16):
                    states[k][m]=rk_step(states[k][m],w,q0,omega,dt,m,k);value=C.angle(states[k][m],exact[m]);diff[str(k)][m].append(value)
            rows.append(dict(episode_id=e['episode_id'],tick=r['tick'],errors16_deg={m:C.angle(states[16][m],exact[m]) for m in KINDS},raw_sha=r['raw_sha']))
            prev=r
    summary=C.read(a.propagation/'summary.json');contract=C.read(a.contract/'summary.json')
    stats={k:{m:C.stats(x) for m,x in d.items()} for k,d in diff.items()}
    assert metric_error<1e-9,metric_error
    assert max(stats['16'][m]['maximum'] for m in KINDS)<1e-4,stats
    # Independently recompute every raw coordinate comparison using scalar ray.
    contract_rows=C.read(a.contract/'rows.json.gz');caps=C.read(a.input/'capture_events.json.gz');maximum_ray_diff=0.
    assert len(caps)==len(contract_rows)==931
    for e,r in zip(caps,contract_rows):
        assert e['old']['raw_sha']==r['raw_sha0'] and e['new']['raw_sha']==r['raw_sha1']
        yaw,pitch=e['old']['angles'];local=[math.cos(pitch)*math.cos(yaw),math.sin(pitch),-math.cos(pitch)*math.sin(yaw)]
        pred=[math.fsum(qi*x for qi,x in zip(qrow,local)) for qrow in e['old']['Q']]
        maximum_ray_diff=max(maximum_ray_diff,abs(C.angle(pred,e['old']['n'])-r['INS_pointing_errors_deg']['Q_native']))
    assert maximum_ray_diff<1e-9
    out=dict(scope='Same-data independent exact-body vs numerical-world ODE verification. Not independent AI reviewers or new blind flights.',input_files_verified=verified,held_row_SHA_checks=len(rows),capture_endpoint_SHA_checks=2*len(caps),recomputed_error_difference_deg=metric_error,independent_ray_max_difference_deg=maximum_ray_diff,RK_vs_exact=stats,math_tests=analytic_tests(),production='No source/config/control changes')
    C.save(a.out/'verification.json',out);C.save(a.out/'numerical_differences.json.gz',rows)
    lines=['# Rate frame-contract results','','Q maps body coordinates to native-world coordinates. n_dot=n cross w. All models use identical lastfresh n/w; no later n/rate resets. Own Q is recorded and interpolated geodesically. No fit.','','## Held direction errors (degrees)','','| Group | Episodes / records | World RMS | Body-expressed inertial RMS | Body-relative RMS |','|---|---:|---:|---:|---:|']
    for name,g in summary['groups'].items():lines.append(f"| {name} | {g['episodes']} / {g['records']} | {g['errors']['world']['rms']:.9f} | {g['errors']['inertial_body']['rms']:.9f} | {g['errors']['relative_body']['rms']:.9f} |")
    lines+=['','## Anchor endpoints','','| UID / seed tick | Body rotation path deg | World | Body-expressed inertial | Body-relative | Zero axial world |','|---|---:|---:|---:|---:|---:|']
    for e in summary['anchors']:
        er=e['endpoint']['errors_deg'];lines.append(f"| {e['uid']} / {e['seed_tick']} | {e['rotation_path_deg']:.6f} | {er['world']:.9f} | {er['inertial_body']:.9f} | {er['relative_body']:.9f} | {er['zero_axial_world']:.9f} |")
    lines+=['','## INS rate geometry (degrees/second)','','| Group | Raw as world | Raw as body (Qw) | Inverse-Q control |','|---|---:|---:|---:|']
    for name,g in contract['groups'].items():lines.append(f"| {name} | {g['INS_rate']['world']['rms']:.9f} | {g['INS_rate']['body']['rms']:.9f} | {g['INS_rate']['inverse_Q_control']['rms']:.9f} |")
    lines+=['','## Important correction','','The prior actual_gimbal comparison used the wrong sign for native yaw z. It is superseded by contract/summary.json. Prior no-gimbal held and axial capture calculations did not call that ray helper. The new coordinate and propagation experiments use the parser convention throughout.','','## Interpretation limits','','World-frame serialization can coexist with body-frame internal tracking. These tests establish output contracts, not private game implementation or developer intent. Near-capture scalar projection compatibility is not full-vector initialization proof. Common world rotation preserves a correctly transformed dot product. No PN/full-trajectory A/B was performed.']
    (a.out/'TABLES.md').write_text('\n'.join(lines)+'\n')
    print('VERIFICATION',json.dumps(C.clean(out)),flush=True);print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
