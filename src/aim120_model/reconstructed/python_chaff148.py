"""Persistent cloud physics and reflection reconstructed from .80 instructions.

14143E250 precision integration and 1414402FC..4454 radius update;
14369A240 kind-1 reflection and 1436981E0 signature loading.
No post-birth replay projectile state is an input.
Common response shapes may replace radius and intrinsic RCS while retaining
each cloud's original inputs, cloud integration and physical clocks.
"""
from bisect import bisect_right
from dataclasses import dataclass
import json
from .python_math118 import *


def envelope(age,profile):
    a,b,c,d=profile
    if age<a or age>d:return 0.
    if age<b:return f((age-a)/(b-a))
    if age<=c:return 1.
    return f((d-age)/(d-c))


def cloud_radius(old,speed,height,dt,*,from_speed,from_viscosity,response,max_radius,radius_mult):
    """1414403AD..4454, including monotone target and density response."""
    rho=f(atmosphere(height)[0])
    speed_factor=f(1-f(math.exp(f(-speed*from_speed))))
    density_factor=f(1-f(math.exp(f(-from_viscosity/rho))))
    target=f(f(f(speed_factor*max_radius)*density_factor)*radius_mult)
    target=max(target,old);tau=f(rho*response)
    if tau>=9.999999717180685e-10:
        return f(old+f(f(target-old)*f(1-f(math.exp(f(-dt/tau))))))
    return target


class ChaffParameters:
    def __init__(self,base,weapon_name='countermeasure_large_split_launcher_jet',*,responses=None):
        self.base=base;self.weapon_name=weapon_name
        self.responses=responses
        root=base
        path=root/f'{weapon_name}.blk'
        source=json.loads(path.read_text())
        choices=[v['bullet']['rocket'] for v in source.values() if isinstance(v,dict) and
            isinstance(v.get('bullet'),dict) and v['bullet'].get('rocket',{}).get('signature',{}).get('isCloud')]
        if source.get('bullet',{}).get('rocket',{}).get('signature',{}).get('isCloud'):
            choices.insert(0,source['bullet']['rocket'])
        if len(choices)!=1:raise ValueError('Ambiguous recorded chaff resource '+weapon_name)
        self.rocket=choices[0]
        self.global_params=json.loads((root/'gameparams.blk').read_text())
        self.signature=self.rocket['signature']
        self.radius_mult=self.signature['cloudRadiusMult'] if 'cloudRadiusMult' in self.signature else 1.
        # The omitted key is supplied by the literal 3F800000 at 143698231.
        self.doppler_width_mult=self.global_params['dopplerSpeedWidthMult'] if 'dopplerSpeedWidthMult' in self.global_params else .10000000149011612
        # 147997E1C is loaded from dopplerSpeedWidthMult, distinct from the
        # similarly named chaffDopplerSpeedWidthMult. This is its image/loader literal.
        assert self.signature['isCloud'] and self.rocket['precision']==30
        assert self.rocket['timeFire']==0 and self.rocket['mass']==self.rocket['massEnd']
        assert self.rocket['CyK']==self.rocket['CxAoA']==0
        self.limit=self.global_params['shellBallisticsParams']['maxAcceleration']

    def radius(self,old,speed,height,dt):
        if self.responses is not None:
            return self.responses.cloud_radius(self,old,speed,height,dt)
        g=self.global_params
        return cloud_radius(old,speed,height,dt,
            from_speed=g['chaffDispersionFromSpeed'],from_viscosity=g['chaffDispersionFromAirViscosity'],
            response=g['chaffDispersionSpeedFromAirViscosity'],max_radius=g['chaffCloudMaxRadius'],
            radius_mult=self.radius_mult)


@dataclass
class CloudFrame:
    time: float
    p: list
    v: list
    radius: float


class ChaffCloud:
    def __init__(self,record,parameters,wind,*,launch_epoch=None):
        self.record=record;self.params=parameters;self.wind=list(wind)
        raw=record['creation_raw_fields'];self.start=raw['creation_time']
        self.launch_epoch=self.start if launch_epoch is None else launch_epoch
        self.physical_dt=f(1/48)
        self.eid=record['eid'];self.owner=record['owner_eid'];self.q=list(raw['u16_1'])
        radius=0.
        if launch_epoch is not None:
            # Creation calls catchup before the ECS current-state writer.
            # That one completed step already runs 1414402FC..4454. Its radius
            # update reads the recorded *end* V and height; do not advance P/V.
            if f(launch_epoch+self.physical_dt)!=self.start:
                raise ValueError('Cloud birth is not the source-proven one-step creation boundary')
            radius=parameters.radius(0.,norm(raw['u12_2']),raw['starting_pos'][1],self.physical_dt)
        self.frames=[CloudFrame(self.start,list(raw['starting_pos']),list(raw['u12_2']),radius)]
        self.times=[self.start]
        # Legacy scalar fixtures start at construction. Explicit launch-event
        # inputs start at the already advanced record boundary.

    def advance(self,end):
        if end<=self.times[-1]:return
        # .86 14143D57A..D5FC: while stored time < requested time,
        # advance to f32(stored time + nominal dt), even past the request.
        # The final step is never clipped to the remaining clock difference.
        while self.times[-1]<end:
            self._step(f(self.times[-1]+self.physical_dt))

    def _step(self,end):
        if end<=self.times[-1]:return
        old=self.frames[-1];p=list(old.p);v=list(old.v);r=self.params.rocket
        total=self.physical_dt;dt=f(total/r['precision'])
        for _ in range(r['precision']):
            rel=sub(v,self.wind);speed=norm(rel);rho,sound=atmosphere(p[1])
            area=f(f(f(r['caliber'])*f(r['caliber']))*f(math.pi/4))
            k=.5*rho*area*r['CxK']*mach_factor(speed/sound)*speed/r['mass']
            acceleration=mul(rel,-k);acceleration[1]-=f(9.8100004196167)
            magnitude=norm(acceleration)
            if magnitude>self.params.limit:acceleration=mul(acceleration,self.params.limit/magnitude)
            p=fv([p[i]+v[i]*dt+.5*acceleration[i]*dt*dt for i in range(3)])
            v=fv([clip(v[i]+acceleration[i]*dt,-2000.,2000.) for i in range(3)])
        radius=self.params.radius(old.radius,norm(v),p[1],total)
        self.frames.append(CloudFrame(end,p,v,radius));self.times.append(end)

    def sample(self,time):
        if time<self.start:return None
        if time>self.times[-1]:self.advance(time)
        index=bisect_right(self.times,time)-1
        if self.times[index]==time:return self.frames[index]
        a,z=self.frames[index:index+2];u=(time-a.time)/(z.time-a.time)
        return CloudFrame(time,fv([x+(y-x)*u for x,y in zip(a.p,z.p)]),
            fv([x+(y-x)*u for x,y in zip(a.v,z.v)]),f(a.radius+(z.radius-a.radius)*u))

    def reflection_frame(self,time):
        # Rocket target adapter 141436530..6638 interpolates its two physical
        # frame copies with an unclamped ratio, including a negative ratio
        # when an already-enumerated new cloud's older ray precedes birth.
        # Existence is tested by the current world query, not at each endpoint.
        if len(self.frames)<2:raise ValueError('Reflection requires the current and previous physical cloud frames')
        a,z=self.frames[-2:];u=f(f(time-a.time)/f(z.time-a.time))
        interpolate=lambda x,y:[f(f(v)+f(f(w-v)*u)) for v,w in zip(x,y)]
        return CloudFrame(time,interpolate(a.p,z.p),interpolate(a.v,z.v),z.radius)

    def alive(self,time):
        return self.start<=time<=self.start+self.params.rocket['timeLife']

    def intrinsic(self,time,radius):
        if self.params.responses is not None:
            return self.params.responses.cloud_intrinsic(self.params,time-self.launch_epoch,radius)
        r=self.params.rocket;g=self.params.global_params;radar=self.params.signature['radar']
        cloud=f(min(radius,g['chaffCloudRcsMaxRadius'])**2)
        # Current sphere-like projectile dimensions use the initialized rocket
        # radar term 147E91CE0/4; cloud dominates after the early zero-envelope arm.
        body=f(.005627799779176712*(r['caliber']+r['length'])**1.899999976158142)
        # .86 141449CB2..9CFC passes query_time - Rocket+384, independently
        # of the advanced physical birth boundary serialized from Rocket+278.
        return f(f(max(body,cloud)*radar['multiplier0'])*envelope(f(time-self.launch_epoch),radar['profile']))


class ChaffWorld:
    def __init__(self,records,parameters,wind,*,launch_epochs=None):
        resources={parameters.weapon_name:parameters}
        for r in records:
            name=r['weapon_name']
            if name not in resources:resources[name]=ChaffParameters(parameters.base,name,responses=parameters.responses)
        self.clouds=[ChaffCloud(r,resources[r['weapon_name']],wind,
            launch_epoch=None if launch_epochs is None else launch_epochs[str(r['eid'])]['time_s']) for r in records]
        self.parameters=parameters

    def advance(self,time):
        for cloud in self.clouds:
            if cloud.alive(time):cloud.advance(time)

    def active(self,time):return [c for c in self.clouds if c.alive(time)]
