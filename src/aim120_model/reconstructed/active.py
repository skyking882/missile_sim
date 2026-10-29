"""Forward-scene adapter around the persistent #148 active-radar state chain.

Capture/search/hold/loss/reacquisition and radial/angular observers remain the
reconstructed caller ordering. Scene truth produces echoes, never INS recovery.
"""
from .python_echo148 import MultiObjectSeeker,FiniteAircraftSeeker148
from .python_math118 import *
from .python_geometry148 import Bounds
from .types import SensorMeasurement
from .responses import SharedResponses


class PointEchoSeeker(MultiObjectSeeker):
    def __init__(self,*args,aircraft_name=None,**kwargs):
        super().__init__(*args,**kwargs);self.aircraft_name=aircraft_name

    def aircraft_echo(self,physical,target,previous):
        # The point-source path is explicit. It shares #148 channel aggregation
        # and observer state with finite aircraft and cloud returns.
        pq,pp,pv,pt=previous;beam=self.angular.beam(physical.q);oldbeam=self.angular.beam(pq)
        r=sub(target[0],physical.p);R=norm(r);n=unit(r);D=dot(sub(physical.v,target[1]),n)
        ot=sub(pt[0],physical.p);orr=sub(pt[0],pp);rt,rr=norm(ot),norm(orr)
        tx,rx=self.raw['transmitter'],self.raw['receiver']
        def gain(leg,b,d,c):
            if self.responses:return self.responses.point_gain(leg,b,d)
            return self.antenna_gain(b,d,c['antenna']['angleHalfSens'],c['antenna']['sideLobesSensitivity'])
        gt=gain('transmitter',beam,n,tx);gr=gain('receiver',beam,n,rx)
        ogt=gain('transmitter',beam,unit(ot),tx);ogr=gain('receiver',oldbeam,unit(orr),rx)
        if self.airframe.get('explicit_rcs') is not None:sigma=self.airframe['explicit_rcs']
        elif self.responses:sigma=self.responses.aircraft_sigma(self.airframe,tv(matrix(target[2]),mul(n,-1.))[0])
        else:raise ValueError('Point target needs an explicit RCS; no aircraft signature is substituted')
        if self.responses:raw=self.responses.receiver_power(sigma,tx=(gt,R,ogt,rt),rx=(gr,R,ogr,rr))
        else:raw=sigma/rx['rcs']*rx['range']**4/.5625*max(gt/R**2,ogt/rt**2)*max(gr/R**2,ogr/rr**2)
        oldR=.5*(rt+rr-norm(sub(pp,physical.p)))
        oldD=.5*(dot(sub(physical.v,pt[1]),unit(ot))+dot(sub(pv,pt[1]),unit(orr)))
        rc,dc=self.channels['distance'],self.channels['dopplerSpeed'];period=dc['maxUnambiguousValue']
        wrapped=self.D+(D-self.D+period/2)%period-period/2
        rg,rn,Rm=self.channel_response(oldR,R,rc['signalWidthMin'],self.R,rc)
        dg,dn,Dm=self.channel_response(oldD,wrapped,dc['signalWidthMin'],self.D,dc)
        return dict(n=n,range_m=Rm,closing_mps=Dm,sigma_m2=sigma,raw_over_floor=raw if R<=self.range_max else 0.,
            channel_gain=[rg,dg],channel_reference_noise=[rn,dn],antenna_power=[gt,gr],
            angular_width=[0.,0.],range_width=rc['signalWidthMin'],doppler_width=dc['signalWidthMin'])


class ActiveSeeker:
    def __init__(self,params,initial_q,world,*,responses=None,aircraft_name=None,rcs_m2=None,bounds=None):
        if responses is None:responses=SharedResponses(params.rocket['guidance']['radarSeeker'])
        kwargs=dict(responses=responses)
        airframe={'explicit_rcs':rcs_m2}
        if bounds is not None:
            box=Bounds(tuple(bounds[0]),tuple(bounds[1]),'explicit scene bounds')
            self.core=FiniteAircraftSeeker148(params,initial_q,airframe,world,1,center_bounds=box,dimension_bounds=box,
                aircraft_name=aircraft_name,**kwargs)
        else:self.core=PointEchoSeeker(params,initial_q,airframe,world,1,aircraft_name=aircraft_name,**kwargs)

    def update(self,time_s,dt_s,physical,target,auxiliary,*,present=True):
        result=self.core.step(physical,target,auxiliary,time_s,dt_s,present)
        mode='track' if result.fresh else 'held' if result.held else 'search' if result.mode in (0,1,2,4) else 'lost'
        return SensorMeasurement(tuple(result.n),tuple(result.w),result.usable,mode,'active_radar',
            result.R,self.core.D,self.core.range_valid and result.usable,self.core.doppler_valid and result.usable,
            result.fresh,result.held,dict(wrapper_mode=result.mode,loss_age_s=result.age,
                radial_mps=result.radial,reason='accepted' if result.fresh else 'held' if result.held else 'no_accepted_echo',
                signal=self.core.last_measurement or {}))
