"""Python port of 143FF7C90 routes 5/6/7/8 and their persistent observers.

World inputs are an explicitly declared point aircraft and clear propagation.
No saved postlaunch missile states are read by this module.
"""
from dataclasses import dataclass, field
from .python_math118 import *


def airframe_sigma(airframe, body_nx):
    front, side, rear = airframe['aspects']
    end = front if body_nx >= 0 else rear
    return (.014069699682295322*(airframe['length']+airframe['span'])**1.899999976158142
            * (side+(end-side)*abs(body_nx)))


def uniform_channel(center, width, gate_center, gate_width, reference_width):
    lo, hi = center-width/2, center+width/2
    def mass(w): return max(0., min(hi, gate_center+w/2)-max(lo, gate_center-w/2))
    signal, reference = mass(gate_width)/width, mass(reference_width)/width
    return signal, max(0.,reference-signal)*gate_width/(reference_width-gate_width)


def swept_channel(first,last,width,gate_center,channel):
    """FDC9F0: endpoint sweep, finite linewidth, capped overlap, reference shell."""
    period=channel['maxUnambiguousValue']
    first=f(math.fmod(first,period));last=f(math.fmod(last,period))
    lo=f(min(first,last)-f(width*.5));hi=f(max(first,last)+f(width*.5))
    # This run remains inside one unambiguous interval; reject rather than guess
    # at the separate period-crossing route.
    if width>=period:raise ValueError('Signal wider than unambiguous period')
    gate_center=f(gate_center)
    def mass(window):
        a=max(lo,channel['minValue'],f(gate_center-window*.5))
        z=min(hi,channel['maxValue'],f(gate_center+window*.5))
        return min(width,max(0.,f(z-a)))
    signal=f(mass(channel['width'])/width)
    reference=f(mass(channel['refWidth'])/width)
    noise=f(max(0.,f(reference-signal))*f(channel['width']/(channel['refWidth']-channel['width'])))
    return signal,noise,last


def fdot(a,b): return f(f(f(a[0]*b[0])+f(a[1]*b[1]))+f(a[2]*b[2]))
def fcross(a,b): return [f(f(a[j]*b[k])-f(a[k]*b[j])) for j,k in ((1,2),(2,0),(0,1))]
def funit(a):
    length=f(math.sqrt(fdot(a,a)))
    reciprocal=f(1/length) if length>3.99999987306209e-19 else 0.
    return [f(x*reciprocal) for x in a]


def scheduled_radial(value, derivative, measurement, age, dt, alpha, beta,
                     channel, prolongate, elapsed, timeout, limits):
    """143FF7590 candidates; angular acceptance controls their later commit."""
    effective=f(dt+(age if prolongate else 0.))
    adjusted=f(value-f(derivative*age)) if prolongate else value
    steps=min(f(48*effective),10.)
    a=f(1-f(math.pow(f(1-alpha),steps))) if steps>0 else 1.
    b=f(1-f(math.pow(f(1-beta),f(steps*steps)))) if steps>0 else 1.
    predicted=f(adjusted+f(derivative*effective))
    residual=f(clip(measurement,channel['minValue'],channel['maxValue'])-predicted)
    increment=f(f(b*residual)*f(1/effective)) if abs(effective)>3.99999987306209e-19 else 0.
    new_derivative=f(derivative+increment)
    if elapsed>=timeout:
        limit=f(f(limits[0]+f(f(limits[1]-limits[0])*min(age,1.)))*effective)
        if abs(increment)>=limit or abs(new_derivative)>=2147440000.: return None
    half=f(channel['width']*.5)
    new_value=f(clip(f(predicted+f(a*residual)),f(channel['minValue']+half),f(channel['maxValue']-half)))
    return new_value,new_derivative


@dataclass
class AngularState:
    angles: list
    n: list
    w: list
    lock_limit: float
    angle_limit: float
    rate: float
    alpha: float
    beta: float
    gate: float
    rate_limited: bool

    @staticmethod
    def direction_angles(local):
        return [f(math.atan2(-local[2],local[0])),f(math.atan2(local[1],math.hypot(local[0],local[2])))]

    def beam(self,q):
        az,el=self.angles
        local=funit(fv([math.cos(el)*math.cos(az),math.sin(el),-math.cos(el)*math.sin(az)]))
        return fv(mv(matrix(q),local))

    def initialize(self,q,local_n,local_w,designation):
        # C2A0 uses the clamped gimbal direction; C560 uses measured direction.
        wanted=self.direction_angles(local_n)
        self.angles=[f(clip(x,-self.angle_limit,self.angle_limit)) for x in wanted]
        self.n=self.beam(q) if designation else fv(mv(matrix(q),local_n))
        self.w=[f(clip(x,-10.,10.)) for x in mv(matrix(q),local_w)]
        return all(abs(x)<self.angle_limit for x in wanted)

    def slew(self,local_n,dt,force_unlimited=False):
        # C8F0 changes only gimbal angles. The original false switch selects 10.
        cap=f((self.rate if self.rate_limited and not force_unlimited else 10.)*dt)
        limit=self.lock_limit
        self.angles=[f(old+clip(clip(f(new-old),-cap,cap),f(-limit-old),f(limit-old)))
                     for old,new in zip(self.angles,self.direction_angles(local_n))]

    def held(self,q,dt):
        # D020 predicts n once before setting antenna angles in current body axes.
        self.n=funit([f(x+f(dt*y)) for x,y in zip(self.n,fcross(self.n,self.w))])
        self.slew(fv(tv(matrix(q),self.n)),dt)

    def observe(self,q,local_n,dt):
        # CA70 keeps prediction and gimbal writes even when the angle gate fails.
        self.slew(local_n,dt)
        predicted=funit([f(x+f(dt*y)) for x,y in zip(self.n,fcross(self.n,self.w))])
        self.n=predicted
        measured=fv(mv(matrix(q),local_n))
        error=fcross(fv(sub(measured,predicted)),predicted)
        if any(abs(x)>f(self.gate*dt) for x in error): return False
        cap=f((self.rate if self.rate_limited else 10.)*dt)
        error=[clip(x,-cap,cap) for x in error]
        correction=fcross(predicted,error)
        self.n=funit([f(x+f(self.alpha*y)) for x,y in zip(predicted,correction)])
        k=f(self.beta/dt)
        self.w=[f(clip(f(x+f(k*y)),-10.,10.)) for x,y in zip(self.w,error)]
        return True


@dataclass
class SeekerOutput:
    n: list
    w: list
    R: float
    radial: float
    fresh: bool=False
    held: bool=False
    usable: bool=False
    mode: int=0
    age: float=0.
    power: dict=field(default_factory=dict)


class Seeker:
    def __init__(self,params,initial_q,airframe):
        self.params=params
        self.raw=params.rocket['guidance']['radarSeeker']
        self.airframe=airframe
        get=lambda key:params.get('guidance/radarSeeker',key)
        # Initialized image byte 147994BCD is false: native observer uses 10 rad/s.
        self.angular=AngularState(fv(get('arretAngles')),fv(mv(matrix(initial_q),[1.,0.,0.])),[0.,0.,0.],
            f(math.radians(get('lockAngleMax'))),f(math.radians(get('angleMax'))),f(math.radians(get('rateMax'))),
            f(get('filterAlpha')),f(get('filterBetta')),f(math.radians(get('angleGateRate'))),False)
        self.mode,self.mode_time=0,0.
        self.range_valid=self.doppler_valid=False
        # 143FF7DC8 reset writes. These are executable literals, not gap fillers.
        self.R=f(self.raw['distance']['minValue']+.5*self.raw['distance']['width']);self.Rrate=0.
        self.D=f(self.raw['dopplerSpeed']['minValue']+.5*self.raw['dopplerSpeed']['width']);self.Drate=0.
        self.reset=False;self.loss_age=0.
        self.previous_time=self.previous_p=self.previous_v=None
        self.previous_q=self.previous_target=None
        self.track_reset=get('trackResetTimeMax');self.prolongation=get('prolongationTimeMax')
        self.prolongate_before=params.gameplay['seeker']['prolongateBeforeTrack']
        self.channels={name:{k:params.get('guidance/radarSeeker/'+name,k) for k in
            ('presents','minValue','maxValue','maxUnambiguousValue','width','signalWidthMin','refWidth','leakMult')}
            for name in ('distance','dopplerSpeed')}
        if get('scanHalfSector')!=0: raise ValueError('Resource must select zero-sector acquisition')
        if get('useTargetId'): raise ValueError('Target-id association is not selected by this resource')
        for key in ('mainBeamNotchWidth','zeroDopplerNotchWidth'):
            if get(key)>=0: raise ValueError('Notch route requires explicit spectrum intervals')
        if get('absDopplerSpeed') or get('semiActive'): raise ValueError('Resource must select active relative Doppler')
        self.reacquire=params.get('guidance','reacquireTarget')
        self.output=SeekerOutput(list(self.n),list(self.w),self.R,self.D)

    @property
    def n(self): return self.angular.n
    @property
    def w(self): return self.angular.w
    @property
    def beam_angles(self): return self.angular.angles

    def designation(self,q,velocity,local_n,local_w,R,D,dt):
        """143FF4A50, both radial-valid bits, useTargetId=false."""
        admitted=self.angular.initialize(q,local_n,local_w,True)
        rawR=f(R+f(self.params.get('guidance/radarSeeker','dopplerSpeedToDistMult')*D))
        for name,measurement,old in [('distance',rawR,self.R),('dopplerSpeed',f(D),self.D)]:
            c=self.channels[name];period=c['maxUnambiguousValue'];half=f(.5*c['width'])
            # Native fmod preserves the sign; there is no minValue offset.
            wrapped=f(math.fmod(measurement,period))
            value=f(clip(wrapped,f(c['minValue']+half),f(c['maxValue']-half)))
            admitted=admitted and abs(f(wrapped-value))<f(c['signalWidthMin']*.5)
            derivative=f(f(value-old)/dt)
            if name=='distance':
                self.range_valid=True
                self.R,self.Rrate=value,f(derivative+fdot(velocity,fv(mv(matrix(q),local_n))))
            else:self.doppler_valid=True;self.D,self.Drate=value,derivative
        return admitted

    def radial_observe(self,q,local_n,R,D,elapsed,dt):
        results=[]
        for value,derivative,measurement,section,channel,limitkey in [
            (self.R,self.Rrate,R,'distGate','distance','accelLimit'),
            (self.D,self.Drate,D,'dopplerSpeedGate','dopplerSpeed','rateLimit')]:
            get=lambda key:self.params.get('guidance/radarSeeker/'+section,key)
            pair=scheduled_radial(value,derivative,measurement,self.loss_age,dt,f(get('filterAlpha')),f(get('filterBetta')),
                self.channels[channel],not self.reset or self.prolongate_before,elapsed,get('limitTimeOut'),get(limitkey))
            if pair is None:self.reset=True;return False
            results.append(pair)
        if not self.angular.observe(q,local_n,dt):self.reset=True;return False
        (self.R,self.Rrate),(self.D,self.Drate)=results
        self.range_valid=self.doppler_valid=True;self.reset=False
        return True

    def held(self,q,dt):
        """143FF7B80; own translation/acceleration already applied by caller."""
        for name,value,derivative,valid in [('distance',self.R,self.Rrate,self.range_valid),('dopplerSpeed',self.D,self.Drate,self.doppler_valid)]:
            c=self.channels[name];value=f(value+f(derivative*dt))
            if c['presents'] and valid:value=f(clip(value,f(c['minValue']+.5*c['width']),f(c['maxValue']-.5*c['width'])))
            if name=='distance':self.R=value
            else:self.D=value
        self.angular.held(q,dt)

    def antenna_gain(self,beam,direction,full_degrees,sidelobe_db):
        # 143329870 equal-width point route, with off-axis sidelobe decay.
        h=f(math.radians(full_degrees*.5));k=f(math.log(2)/math.sin(h)**2)
        cosine=dot(beam,direction);sine2=max(0.,1-cosine*cosine)
        if cosine<0.:return 0.  # 143329BCB rejects negative antenna-forward projection.
        main=math.exp(-k*sine2);floor=f(10**(sidelobe_db/10))
        knee=math.asin(math.sqrt(min(1.,-.5*math.log(floor)/k)))
        attenuation=math.exp(2*math.log(floor)*max(0.,math.asin(min(1.,math.sqrt(sine2)))-knee)/(math.pi/2-knee))
        return max(main,floor*attenuation)

    def power(self,physical,target,search,previous):
        tp,tv_,tq=target;rel=sub(tp,physical.p);distance=norm(rel);direction=unit(rel)
        closing=dot(sub(physical.v,tv_),direction)
        pq,pp,pv,pt=previous
        old_direction=unit(sub(pt[0],pp))
        old_rx_direction=unit(sub(pt[0],physical.p))
        # FF4E30 supplies previous transmitter and current receiver endpoints
        # for the older half of the integration interval (object context B0/F0).
        old_distance=.5*(norm(sub(pt[0],pp))+norm(sub(pt[0],physical.p))-norm(sub(pp,physical.p)))
        old_closing=.5*(dot(sub(pv,pt[1]),old_direction)+dot(sub(physical.v,pt[1]),old_rx_direction))
        sigma=airframe_sigma(self.airframe,tv(matrix(tq),mul(direction,-1.))[0])
        tx,rx=self.raw['transmitter'],self.raw['receiver'];beam=self.angular.beam(physical.q)
        gt=self.antenna_gain(beam,direction,tx['antenna']['angleHalfSens'],tx['antenna']['sideLobesSensitivity'])
        gr=self.antenna_gain(beam,direction,rx['antenna']['angleHalfSens'],rx['antenna']['sideLobesSensitivity'])
        candidate=True
        if search:
            # 143FF9E88..A072 candidate cells and explicit designation search spans.
            if self.range_valid:candidate &= abs(distance-self.R)<=self.params.get('guidance/radarSeeker/distGate','distGateSearchRange')
            if self.doppler_valid:candidate &= abs(closing-self.D)<=self.params.get('guidance/radarSeeker/dopplerSpeedGate','dopplerSpeedGateSearchRange')
            Rcenter,Dcenter=distance,closing
        else:Rcenter,Dcenter=self.R,self.D
        rc,dc=self.channels['distance'],self.channels['dopplerSpeed'];period=dc['maxUnambiguousValue']
        wrapped_D=Dcenter+(closing-Dcenter+period/2)%period-period/2
        rg,rn,Rmeasured=swept_channel(old_distance,distance,rc['signalWidthMin'],Rcenter,rc)
        dg,dn,Dmeasured=swept_channel(old_closing,wrapped_D,dc['signalWidthMin'],Dcenter,dc)
        if not self.params.gameplay['radarCommon']['useRefWidthForTargets']:rn=dn=0.
        base=sigma/rx['rcs']*(rx['range']/distance)**4*gt*gr/.5625
        multiplier=self.params.gameplay['radarCommon']['singleTargetDetectionSensitivityMult']
        noise_gain=rn*dn if min(rn,dn)>3.99999987306209e-19 else max(rn,dn)
        signal=multiplier*base*rg*dg;noise=base*noise_gain;score=signal/(1+noise)
        return dict(sigma_m2=sigma,range_m=Rmeasured,closing_mps=Dmeasured,antenna_power=[gt,gr],
            channel_gain=[rg,dg],channel_reference_noise=[rn,dn],raw_over_floor=base,score=score,
            candidate=bool(candidate),accepted=bool(candidate and distance<=rx['rangeMax'] and score>1.),
            n=fv(direction),local_n=fv(tv(matrix(physical.q),direction)),target_radial=dot(tv_,direction),gate_centers=[Rcenter,Dcenter])

    def operation(self,op,physical,target,designation,elapsed,dt,previous_p,previous_v,target_present,previous_q=None,previous_target=None):
        """C90 5/6/7/8: echo acceptance, observer acceptance, hold validity differ."""
        if op in (5,6):
            if designation['valid']:
                self.designation(physical.q,physical.v,designation['local_n'],designation['local_w'],designation['R'],designation['D'],dt)
            else:
                self.angular.angles=fv(self.params.get('guidance/radarSeeker','arretAngles'))
                self.angular.n=self.angular.beam(physical.q);self.angular.w=[0.,0.,0.]
                self.range_valid=self.doppler_valid=False
                self.R=f(self.raw['distance']['minValue']+.5*self.raw['distance']['width']);self.Rrate=0.
                self.D=f(self.raw['dopplerSpeed']['minValue']+.5*self.raw['dopplerSpeed']['width']);self.Drate=0.
            self.loss_age=f(self.loss_age+dt)
            return False,False,{}
        if op not in (7,8):raise ValueError('Unrecovered operation '+str(op))
        if previous_q is None or previous_target is None:raise ValueError('Active echo needs both explicit endpoint frames')
        previous=(previous_q,previous_p,previous_v,previous_target)
        search=op==7
        self.angular.slew(fv(tv(matrix(physical.q),self.n)),dt,force_unlimited=True)
        # Search enumerates before own-motion shift; tracking shifts first.
        measurement=self.power(physical,target,True,previous) if search and target_present else {}
        if search and measurement.get('accepted',False):
            self.R=measurement['range_m'];self.D=measurement['closing_mps']
            self.range_valid=self.doppler_valid=True
        old_n=list(self.n)
        delta_p=fdot(fv(sub(physical.p,previous_p)),old_n);delta_v=fdot(fv(sub(physical.v,previous_v)),old_n)
        self.R=f(self.R-delta_p);self.D=f(self.D+delta_v)
        if not search and target_present:measurement=self.power(physical,target,False,previous)
        accepted=measurement.get('accepted',False)
        if accepted:
            if search:
                # FF74B0 passes S+14 itself as C560's third argument. Preserve
                # that exact rotation of the existing w, including first capture.
                self.angular.initialize(physical.q,measurement['local_n'],list(self.w),False)
            elif self.loss_age>self.track_reset:self.angular.initialize(physical.q,measurement['local_n'],[0.,0.,0.],False)
            accepted=self.radial_observe(physical.q,measurement['local_n'],measurement['range_m'],measurement['closing_mps'],elapsed,dt)
        if accepted:
            # AD72/B2C1 skip conversion when the Doppler channel presents bit
            # is set. +34 remains measured closing speed in this configuration.
            self.loss_age=0.
            return True,False,measurement
        self.held(physical.q,dt);self.loss_age=f(self.loss_age+dt)
        return False,(not search and self.loss_age<self.prolongation),measurement

    def step(self,physical,target,auxiliary,elapsed,dt,target_present=True):
        p=self.params.rocket['guidance']
        previous_time=self.previous_time if self.previous_time is not None else f(physical.time-dt)
        previous_p=self.previous_p if self.previous_p is not None else list(physical.p)
        previous_v=self.previous_v if self.previous_v is not None else list(physical.v)
        if self.mode<2 and elapsed>p['lockTimeOut']:
            forward=dot(auxiliary['n'],self.n)
            if not auxiliary['valid'] or (forward*abs(forward)>math.cos(math.radians(self.raw['lockAngleMax']))**2 and auxiliary['R']<p['lockDistance']):
                self.mode,self.mode_time=2,physical.time
        elif self.mode==4 and physical.time%1<previous_time%1:self.mode,self.mode_time=2,physical.time
        op={0:5,1:6,2:7,3:8,4:6}[self.mode]
        designation=dict(auxiliary,local_n=fv(tv(matrix(physical.q),auxiliary['n'])),local_w=fv(tv(matrix(physical.q),auxiliary['w'])))
        fresh,held,power=self.operation(op,physical,target,designation,elapsed,dt,previous_p,previous_v,target_present,self.previous_q,self.previous_target)
        usable=fresh or held;before=self.mode
        if usable:self.mode=3
        elif self.reacquire:
            if self.mode==3:self.mode,self.mode_time=2,physical.time
            elif self.mode==2:self.mode=4  # zero scan sector <= both antenna widths
        power=dict(power,operation=op,wrapper_mode_before=before,angles=list(self.beam_angles),
                   range_valid=self.range_valid,doppler_valid=self.doppler_valid,Rrate=self.Rrate,Drate=self.Drate)
        # Wrapper 1452BCD45 passes S+2C, not the Doppler field, to INS_update.
        self.output=SeekerOutput(list(self.n),list(self.w),self.R,self.Rrate,fresh,held,usable,self.mode,self.loss_age,power)
        self.previous_time=physical.time;self.previous_p=list(physical.p);self.previous_v=list(physical.v)
        self.previous_q=list(physical.q);self.previous_target=target
        return self.output
