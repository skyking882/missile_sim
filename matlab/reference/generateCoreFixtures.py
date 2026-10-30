"""Generate pinned reference fixtures; supply checkout of the rc3 reconstructed package.
Usage: PYTHONPATH=/path/to/reference python generateCoreFixtures.py
"""
import copy,json,math,pathlib
from reconstructed.resources import Parameters
from reconstructed.python_physics118 import PhysicalState,aerodynamic_force
from reconstructed.python_physics148 import Physics148
from reconstructed.python_math118 import euler_q,matrix,curve,atmosphere
from reconstructed.python_guidance118 import Controller

def motor(p,t,h):
 r=p.rocket;previous=r['mass'];finish=0.;thrust=lost=0.
 for index in range(3):
  suffix=str(index) if index else ''
  if 'timeFire'+suffix not in r:break
  duration=r['timeFire'+suffix];start=finish+r.get('fireDelay'+suffix,0);finish=start+duration
  dm=previous-r['massEnd'+suffix];previous=r['massEnd'+suffix];age=t-start;u=age/duration if duration else float(age>0)
  lost+=dm*max(0,min(1,u))
  if 0<age<duration:thrust+=r['force'+suffix]*(1+.08*(1-2*u)+.06*(6*u*(1-u)-1))
 if 'extPressureToThrustMult' in r:
  k=r['extPressureToThrustMult'];k=[k[:2],k[2:]] if len(k)==4 and isinstance(k[0],(int,float)) else k
  rho,sound=atmosphere(h);thrust*=curve(sorted(k),rho*sound*sound/1.4)
 return thrust,lost

rows=[]
for id in ['us_aim_120a','su_r_77_1']:
 p=Parameters(id);q=euler_q(.02,-.07,.04);state=PhysicalState([0.,5000.,0.],[650.,12.,-4.],q,[.01,-.02,.03],0.)
 class Ref(Physics148):
  def motor(self,elapsed):return motor(p,elapsed,self.state.p[1])
 phys=Ref(p.physical(),copy.deepcopy(state),physical_dt=1/48);phys.control=[.13,-.07]
 static=aerodynamic_force(state.v,state.q,state.omega,phys.control,5000.,p.physical(),[0,0,0]);first=phys.step(1/48,[0,0,0]);
 expected=[copy.deepcopy(phys.state.__dict__)];
 for i in range(1,96):phys.step((i+1)/48,[0,0,0]);expected.append(copy.deepcopy(phys.state.__dict__))
 rows.append(dict(id=id,state=state.__dict__,control=phys.control,static=static,first=first,trajectory=expected))
pathlib.Path(__file__).resolve().parents[1].joinpath('data/core_reference.json').write_text(json.dumps(rows,indent=2)+'\n')
from reconstructed.python_math118 import dot,cross,unit,tv,norm,f
from reconstructed.python_guidance118 import PIDAxis,pn_request
cases=[]
for id in ['us_aim_120a','su_r_77_1']:
 p=Parameters(id);a=p.rocket['guidance']['guidanceAutopilot'];axes=[PIDAxis(),PIDAxis()];q=euler_q(0.,0.,.02);state=PhysicalState([0.,5000.,0.],[650.,12.,-4.],q,[0.,0.,0.],3.)
 R=matrix(q);forward=unit(state.v);z=unit(cross(forward,unit([row[1] for row in R])));y=cross(z,forward);basis=[forward,y,z];feedback=[.3,-9.2,1.1]
 samples=[]
 for index,cmd in enumerate(([5.,-3.],[10.,4.],[-4.,-10.],[300.,-400.],[0.,0.])):
  t=3.+index/48;pid=sorted([v for k,v in a.items() if k.startswith('pid') and k[3:].isdigit()],key=lambda v:v['time'])
  if pid:
   for key,col in [('accelControlProp','prop'),('accelControlIntg','intg'),('accelControlDiff','diff'),('accelControlIntgLim','intgLim')]:a[key]=curve([[r['time'],r[col]] for r in pid],t)
  cap=f(a['reqAccelMax']*f(9.8100004196167));scale=min(1,cap/math.hypot(*cmd)) if math.hypot(*cmd) else 1;fb=[dot(b,feedback) for b in basis];errors=[cmd[0]*scale-fb[2],cmd[1]*scale-fb[1]]
  rho,_=atmosphere(5000);base=f(a['baseIndSpeed']*f(.2777777910232544));B=base*base*1.225000023841858**2/(rho*rho*norm(state.v)**2);gains=[f(a['accelControlProp']*math.sqrt(B)),f(a['accelControlIntg']*math.sqrt(B)),f(a['accelControlDiff']*B)]
  values=[axes[k].step(errors[k],*gains,a['accelControlIntgLim'],1/48,k==0)[0] for k in range(2)]
  samples.append(dict(time=t,command=cmd,control=values))
 cases.append(dict(id=id,state=state.__dict__,feedback=feedback,samples=samples))
pathlib.Path(__file__).resolve().parents[1].joinpath('data/core_pid_reference.json').write_text(json.dumps(cases,indent=2)+'\n')
guidance=[]
for id in ['us_aim_120a','su_r_77_1']:
 p=Parameters(id);a=p.rocket['guidance']['guidanceAutopilot'];state=PhysicalState([0.,5000.,0.],[650.,12.,-4.],euler_q(0.,-.03,.02),[0.,0.,0.],3.)
 target={'position':[6000.,5200.,200.],'velocity':[-200.,0.,5.]};r=[b-c for b,c in zip(target['position'],state.p)];vr=[b-c for b,c in zip(target['velocity'],state.v)];R=norm(r);direction=unit(r);omega=[v/(R*R) for v in cross(vr,r)];D=-dot(vr,direction);tgo=R/D
 def loaded(prefix,x):
  pts=[v for k,v in a.items() if k.startswith(prefix) and k[len(prefix):].isdigit()];return curve(sorted(pts),x) if pts else 1.
 gain=loaded('timeToGain',state.time)*loaded('timeToHitToGain',tgo)
 varg=[-v for v in vr] if p.rocket['guidance'].get('useTargetVel',False) else state.v
 request=pn_request(varg,omega,tv(matrix(state.q),direction),gain,p.get('guidance/guidanceAutopilot','omegaToAccelMult'),p.get('guidance/guidanceAutopilot','angleToAccelMult'),p.get('guidance/guidanceAutopilot','elevationAdd'))
 z=unit(cross(unit(state.v),unit([row[1] for row in matrix(state.q)])));y=cross(z,unit(state.v));cmd=[dot(z,request),dot(y,request)]
 guidance.append(dict(id=id,state=state.__dict__,target=target,command=cmd))
pathlib.Path(__file__).resolve().parents[1].joinpath('data/core_guidance_reference.json').write_text(json.dumps(guidance,indent=2)+'\n')
