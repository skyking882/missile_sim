"""Aircraft/cloud echo gating and persistent seeker, without native execution.

143FDDA90 accumulates classified signal moments and reference noise;
143FE53C0 computes centroids, mean absolute spread and dominant contribution.
The original aircraft-only route remains available to #118. The finite #148
route accepts common resource-scaled responses without changing gate/state logic.
"""
from .python_seeker118 import Seeker,scheduled_radial,airframe_sigma,fdot,fcross,funit,fv,f
from .python_gates148 import swept_channel148
from .python_math118 import *
import json
from pathlib import Path
from .python_geometry148 import point_geometry,endpoint_geometry,reflection_widths,finite_pattern_gain


def scheduled_doppler148(value,derivative,measurement,age,dt,alpha,beta,
                         channel,prolongate,elapsed,timeout,limits):
    """.80 FF77A9..7B14: Doppler limits differ from the distance limits.

    FF7ABB bounds the increment by 2147440000*effective_dt; FF7AFC
    bounds the resulting derivative by the age-dependent configured rate.
    """
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
        limit=f(limits[0]+f(f(limits[1]-limits[0])*min(age,1.)))
        if abs(increment)>=f(2147440000.*effective) or abs(new_derivative)>=limit:return None
    half=f(channel['width']*.5)
    new_value=f(clip(f(predicted+f(a*residual)),f(channel['minValue']+half),f(channel['maxValue']-half)))
    return new_value,new_derivative


def search_cells(objects,R,D,range_valid,doppler_valid,range_channel,doppler_channel,range_span,doppler_span):
    """FF9EB0..FFA22D: guards, two 1024-bit tables, rank by squared residual.

    A cell step is signalWidthMin/2, not the receiver window width. The two
    occupancy tables are independent: retain a candidate if either bin is new.
    Geometry is supplied from the current frame object query, as at FF9D52.
    """
    occupied=[set(),set()];out=[]
    for obj in objects:
        centers=[f(obj['range_m']),f(obj['closing_mps'])];rank=0.;admitted=True
        for value,center,valid,span in zip(centers,[R,D],[range_valid,doppler_valid],[range_span,doppler_span]):
            if not valid:continue
            residual=abs(f(value-center))
            if residual>span:admitted=False;break
            rank=f(rank+f(f(residual/span)**2))
        if not admitted:continue
        seen=[];bins=[]
        for i,(value,channel) in enumerate(zip(centers,[range_channel,doppler_channel])):
            if not channel['presents']:seen.append(True);bins.append(None);continue
            width=channel['signalWidthMin'];inverse=f(2/width) if abs(f(width*.5))>3.99999987306209e-19 else 0.
            index=math.floor(f(f(f(value-channel['minValue'])*inverse)+.5))
            bins.append(index)
            if not 0<=index<=1023:seen.append(False);continue
            seen.append(index in occupied[i]);occupied[i].add(index)
        if all(seen):continue
        out.append(dict(obj,range_m=centers[0],closing_mps=centers[1],rank=rank,bins=bins))
    # The selected small candidate array uses the native strict-less insertion
    # tail. Equal ranks retain their incoming order in that tail.
    out.sort(key=lambda row:row['rank'])
    return out


def candidate_geometry(position,velocity,current,beam,cone_cosine,range_max):
    """FF9D52 queries frame.time (+28). Active current Tx and Rx coincide.

    This is the current frame argument of C90, not FDD's older sweep endpoint.
    """
    r=fv(sub(position,current.p));r2=f(f(f(r[2]*r[2])+f(r[1]*r[1]))+f(r[0]*r[0]))
    projection=fdot(r,beam)
    if f(projection*abs(projection))<f(f(cone_cosine*cone_cosine)*r2):return None
    R=f(math.sqrt(r2))
    if R>range_max:return None
    direction=[f(v*f(1/R)) for v in r] if R>1e-9 else [0.,0.,0.]
    return R,fdot(fv(sub(current.v,velocity)),direction)


def antenna_shape(antenna):
    """Loaded h/k/side/knee from the required scalar antenna resource branch."""
    h=max(f(f(antenna['angleHalfSens'])*f(.008726646192371845)),f(.0001))
    sine=f(math.sin(h));k=f(f(math.log(2))/f(sine*sine))
    side=f(2**f(f(antenna['sideLobesSensitivity'])*f(.3321928083896637)))
    knee=f(math.asin(math.sqrt(min(1.,f(f(-.5*f(math.log(max(side,f(side*side)))))/k))))) if side<=.9998999834060669 else 0.
    return dict(h=h,k=k,side=side,knee=knee)


def fold_main_lobe(direction,basis,signal,zones):
    """FDF1DB..4B8: wrap weighted transverse components with CRT fmodf.

    Rebuild the forward component at the original signal magnitude. This is
    neither an angle clip nor an observer acceptance test.
    """
    if signal<=0:return list(direction),False
    moment=[f(v*signal) for v in direction];local=inverse_project(basis,moment)
    limits=[f(z*signal) for z in zones]
    if abs(local[2])<=limits[0] and abs(local[1])<=limits[1]:return list(direction),False
    z=f(math.fmod(local[2],limits[0]));y=f(math.fmod(local[1],limits[1]))
    x=f(math.sqrt(max(0.,f(f(signal*signal)-f(f(z*z)+f(y*y))))))
    world=[fdot(row,[x,y,z]) for row in basis];reciprocal=f(1/signal)
    return [f(v*reciprocal) for v in world],True


def receiver_basis(direction):
    """FF4F92..50xx: horizon-aligned receiver frame, columns forward/up/side."""
    x,y,z=direction;h=f(math.sqrt(f(f(x*x)+f(z*z))))
    side=[0.,0.,1.] if h<=1e-9 else [f(-z/h),0.,f(x/h)]
    up=fcross(side,direction)
    return [[direction[i],up[i],side[i]] for i in range(3)]


def inverse_project(basis,vector):
    # FE5DDF..5F70 uses the actual inverse; do not replace it by a transpose
    # when the three float32 basis vectors are not exactly orthonormal.
    cols=[[basis[j][i] for j in range(3)] for i in range(3)]
    cofactors=[fcross(cols[1],cols[2]),fcross(cols[2],cols[0]),fcross(cols[0],cols[1])]
    determinant=fdot(cols[0],cofactors[0]);inverse=f(1/determinant)
    return [f(fdot(row,vector)*inverse) for row in cofactors]


def aggregate_returns(returns,range_floor,doppler_floor,*,basis):
    """FE57B8..57FD sums; FE5996 normalizes; FE5D30 adds absolute spread."""
    accepted=[r for r in returns if r['signal']>0.]
    def fsum(values):
        total=0.
        for value in values:total=f(total+value)
        return total
    signal=fsum(r['signal'] for r in accepted);noise=fsum(r['noise'] for r in returns)
    if signal<=0:return dict(signal=0.,noise=noise,contributions=[],object_count=0)
    reciprocal=f(1/signal)
    moment=lambda r,key:f(r['signal']*r[key])
    weighted=lambda key:f(fsum(moment(r,key) for r in accepted)*reciprocal)
    n=[f(fsum(f(r['signal']*r['n'][i]) for r in accepted)*reciprocal) for i in range(3)]
    R,D=weighted('range_m'),weighted('closing_mps')
    angular=[f(fsum(f(r['signal']*r['angular_width'][i]) for r in accepted)*reciprocal) for i in range(2)]
    Rwidth=max(range_floor,weighted('range_width'))
    Dwidth=max(doppler_floor,weighted('doppler_width'))
    if len(accepted)>=2:
        Rwidth=f(Rwidth+f(fsum(abs(f(moment(r,'range_m')-f(r['signal']*R))) for r in accepted)*reciprocal))
        Dwidth=f(Dwidth+f(fsum(abs(f(moment(r,'closing_mps')-f(r['signal']*D))) for r in accepted)*reciprocal))
        spread=[f(fsum(abs(f(f(r['signal']*r['n'][i])-f(r['signal']*n[i]))) for r in accepted)*reciprocal) for i in range(3)]
        local=inverse_project(basis,spread)
        # FE5F7F..5FA6 adds twice the projected mean absolute directional spread.
        angular=[f(angular[0]+f(2*abs(local[2]))),f(angular[1]+f(2*abs(local[1])))]
    contributions=sorted([dict(eid=r['eid'],kind=r['kind'],fraction=r['signal']/signal,
        signal=r['signal'],noise=r['noise'],range_m=r['range_m'],closing_mps=r['closing_mps'],
        channel_gain=r['channel_gain'],sigma_m2=r['sigma_m2']) for r in accepted],key=lambda r:-r['fraction'])
    return dict(signal=signal,noise=noise,n=fv(n),range_m=f(R),closing_mps=f(D),
        range_width=f(Rwidth),doppler_width=f(Dwidth),angular_width=angular,object_count=len(accepted),
        contributions=contributions,dominant_object=contributions[0]['eid'])


class MultiObjectSeeker(Seeker):
    def channel_response(self,first,last,width,center,channel):
        # FF5DE8..5E17 writes [R,0,0] and [D,0,0]. The inherited
        # constructor checks the resource's disabled main/zero notch arms.
        return swept_channel148(first,last,width,center,channel,
            use_reference_width=self.params.gameplay['radarCommon']['useRefWidthForTargets'],
            dynamic_gate_width=0.,minimum_gate_width=0.,excluded_intervals=[])

    def __init__(self,params,initial_q,airframe,world,target_eid,*,responses=None):
        super().__init__(params,initial_q,airframe)
        self.responses=responses
        # 143FF7E54..E64 initializes the invalid Doppler center using the
        # signal width (descriptor+19C), rather than receiver window width.
        self.D=f(self.channels['dopplerSpeed']['minValue']+.5*self.channels['dopplerSpeed']['signalWidthMin'])
        self.world=world;self.target_eid=target_eid;self.last_objects=[]
        self.last_measurement=None
        # 143FF4222..422F and 4430..443D multiply the resource's full
        # search spans by .5 before storing descriptor+39C/+3A0.
        self.search_spans=[f(f(params.get('guidance/radarSeeker/'+section,key))*f(.5))
            for section,key in [('distGate','distGateSearchRange'),
                                ('dopplerSpeedGate','dopplerSpeedGateSearchRange')]]
        # Query/folding/extent guards retain each original resource's values;
        # response-function replacement does not replace these boundaries.
        shapes=[antenna_shape(self.raw[k]['antenna']) for k in ['transmitter','receiver']]
        self.rx_shape=shapes[1]
        search_cos=f(math.cos(f(self.params.get('guidance/radarSeeker','searchFov')*f(.008726646192371845))))
        self.query_cosine=max(f(math.cos(self.rx_shape['knee'])),f(search_cos*abs(search_cos)))
        self.range_max=self.raw['receiver']['rangeMax']
        reads=json.loads((Path(__file__).resolve().parent/'data/sensor_initializer.json').read_text())
        floors=[v['value'] for v in reads if v['path']=='config/gameplay.blk/radarCommon' and v['key']=='directionMainBeamWidthMin']
        if not floors or any(v!=floors[0] for v in floors):raise ValueError('Missing verified directionMainBeamWidthMin initializer')
        self.direction_floor=floors[0]

    def radial_observe(self,q,local_n,R,D,elapsed,dt):
        guards=[]
        results=[]
        for value,derivative,measurement,section,channel,limitkey in [
            (self.R,self.Rrate,R,'distGate','distance','accelLimit'),
            (self.D,self.Drate,D,'dopplerSpeedGate','dopplerSpeed','rateLimit')]:
            get=lambda key:self.params.get('guidance/radarSeeker/'+section,key)
            candidate=scheduled_radial if channel=='distance' else scheduled_doppler148
            pair=candidate(value,derivative,measurement,self.loss_age,dt,f(get('filterAlpha')),f(get('filterBetta')),
                self.channels[channel],not self.reset or self.prolongate_before,elapsed,get('limitTimeOut'),get(limitkey))
            guards.append(dict(channel=channel,candidate_accepted=pair is not None,old_value=value,
                derivative=derivative,measurement=measurement,candidate=pair))
            if pair is None:break
            results.append(pair)
        accepted=len(results)==2
        angular_guard=None
        if accepted:
            predicted=funit([f(x+f(dt*y)) for x,y in zip(self.n,fcross(self.n,self.w))])
            measured=fv(mv(matrix(q),local_n))
            error=fcross(fv(sub(measured,predicted)),predicted)
            angular_guard=dict(predicted_n=predicted,measured_n=measured,error=error,
                component_limit=f(self.angular.gate*dt))
            accepted=self.angular.observe(q,local_n,dt)
            angular_guard['accepted']=accepted
        if accepted:
            (self.R,self.Rrate),(self.D,self.Drate)=results
            self.range_valid=self.doppler_valid=True;self.reset=False
        else:self.reset=True
        if self.last_measurement is not None:
            failure=next((g['channel'] for g in guards if not g['candidate_accepted']),None)
            self.last_measurement['observer']=dict(accepted=accepted,radial_candidates=guards,angular_guard=angular_guard,
                rejection_stage=failure or ('angular' if not accepted else None))
        return accepted

    def aircraft_echo(self,physical,target,previous):
        echo=super().power(physical,target,False,previous)
        pq,pp,pv,pt=previous
        old_tx=sub(pt[0],physical.p);old_rx=sub(pt[0],pp)
        old_beam=self.angular.beam(pq)
        tx,rx=self.raw['transmitter'],self.raw['receiver']
        old_gt=self.antenna_gain(self.angular.beam(physical.q),unit(old_tx),tx['antenna']['angleHalfSens'],tx['antenna']['sideLobesSensitivity'])
        old_gr=self.antenna_gain(old_beam,unit(old_rx),rx['antenna']['angleHalfSens'],rx['antenna']['sideLobesSensitivity'])
        distance=norm(sub(target[0],physical.p));gt,gr=echo['antenna_power']
        # FDD's two FDB490 contexts copy both Tx and Rx antenna frames:
        # current +30/+F0 versus older +70/+B0. The active caller keeps
        # Tx at the current frame for both target endpoints. FDE924 and
        # FDFA30 take each leg's endpoint maximum before multiplying legs.
        raw=echo['sigma_m2']/rx['rcs']*rx['range']**4/.5625
        raw*=max(gt/distance**2,old_gt/norm(old_tx)**2)*max(gr/distance**2,old_gr/norm(old_rx)**2)
        echo['raw_over_floor']=raw
        echo['antenna_endpoints']=[[gt,gr],[old_gt,old_gr]]
        echo.update(angular_width=[0.,0.],range_width=self.channels['distance']['signalWidthMin'],
                    doppler_width=self.channels['dopplerSpeed']['signalWidthMin'])
        return echo

    def cloud_echo(self,physical,cloud,previous,Rcenter,Dcenter):
        if len(cloud.frames)<2:return None
        now=cloud.reflection_frame(physical.time);old=cloud.reflection_frame(previous[3])
        rel=sub(now.p,physical.p);R=norm(rel)
        if R<=10.:return None
        n=unit(rel);D=dot(sub(physical.v,now.v),n)
        pq,pp,pv,oldtime=previous
        old_n=unit(sub(old.p,physical.p));old_rx_n=unit(sub(old.p,pp))
        old_R=.5*(norm(sub(old.p,pp))+norm(sub(old.p,physical.p))-norm(sub(pp,physical.p)))
        old_D=.5*(dot(sub(physical.v,old.v),old_n)+dot(sub(pv,old.v),old_rx_n))
        rc,dc=self.channels['distance'],self.channels['dopplerSpeed'];period=dc['maxUnambiguousValue']
        wrapped=Dcenter+(D-Dcenter+period/2)%period-period/2
        sigma=cloud.intrinsic(physical.time,now.radius)
        tx,rx=self.raw['transmitter'],self.raw['receiver'];beam=self.angular.beam(physical.q)
        old_beam=self.angular.beam(pq)
        if self.responses is not None:
            gt=self.responses.point_gain('transmitter',beam,n)
            gr=self.responses.point_gain('receiver',beam,n)
            old_gt=self.responses.point_gain('transmitter',beam,old_n)
            old_gr=self.responses.point_gain('receiver',old_beam,old_rx_n)
            raw=self.responses.receiver_power(sigma,tx=(gt,R,old_gt,norm(sub(old.p,physical.p))),
                rx=(gr,R,old_gr,norm(sub(old.p,pp))))
        else:
            gt=self.antenna_gain(beam,n,tx['antenna']['angleHalfSens'],tx['antenna']['sideLobesSensitivity'])
            gr=self.antenna_gain(beam,n,rx['antenna']['angleHalfSens'],rx['antenna']['sideLobesSensitivity'])
            old_gt=self.antenna_gain(beam,old_n,tx['antenna']['angleHalfSens'],tx['antenna']['sideLobesSensitivity'])
            old_gr=self.antenna_gain(old_beam,old_rx_n,rx['antenna']['angleHalfSens'],rx['antenna']['sideLobesSensitivity'])
            # FDD takes transmitter and receiver maxima independently.
            raw=sigma/rx['rcs']*rx['range']**4/.5625
            raw*=max(gt/R**2,old_gt/norm(sub(old.p,physical.p))**2)*max(gr/R**2,old_gr/norm(sub(old.p,pp))**2)
        range_width=f(rc['signalWidthMin']+now.radius)
        # 141449C68..9CB3 passes Rocket+244 directly to 14369A240.
        # A859..A888 reads its stored +1C V, not the target adapter's
        # interpolated query V. The radius is likewise the stored current one.
        current=cloud.frames[-1]
        vx,vy,vz=map(f,current.v)
        speed=f(math.sqrt(f(f(vz*vz)+f(f(vx*vx)+f(vy*vy)))))
        intrinsic_width=f(speed*f(cloud.params.doppler_width_mult))
        doppler_width=f(intrinsic_width+f(dc['signalWidthMin']))
        rg,rn,Rm=self.channel_response(old_R,R,range_width,Rcenter,rc)
        dg,dn,Dm=self.channel_response(old_D,wrapped,doppler_width,Dcenter,dc)
        noise_gain=rn*dn if min(rn,dn)>3.99999987306209e-19 else max(rn,dn)
        # FDEFF0..FDF00C uses the older receiver endpoint inverse range
        # with the reflection's current transverse sizes, also for clouds.
        old_inverse=f(1/norm(sub(old.p,pp)))
        return dict(eid=cloud.eid,kind='chaff',n=fv(n),range_m=Rm,closing_mps=Dm,
            angular_width=[f(current.radius*old_inverse)]*2,
            range_width=range_width,doppler_width=doppler_width,signal=raw*rg*dg,noise=raw*noise_gain,
            raw_over_floor=raw,channel_gain=[rg,dg],sigma_m2=sigma,age_s=physical.time-cloud.launch_epoch,
            radius_m=now.radius,position=now.p,velocity=now.v,
            channel_inputs=dict(range_first=old_R,range_last=R,range_width=range_width,range_center=Rcenter,
                doppler_first=old_D,doppler_last=wrapped,doppler_width=doppler_width,doppler_center=Dcenter,
                reflection_physical_time=current.time,reflection_velocity=current.v,
                reflection_intrinsic_doppler_width=intrinsic_width))

    def power(self,physical,target,search,previous):
        self.world.advance(physical.time)
        clouds=self.world.active(physical.time)
        # C90's frame+28 is current physical time. FDD separately integrates
        # both endpoints after candidate selection. The same candidate, moment,
        # direction-fold and acceptance route applies before any cloud exists.
        candidates=[]
        if search:
            dr,dd=self.search_spans
            beam=self.angular.beam(physical.q);objects=[]
            center=candidate_geometry(target[0],target[1],physical,beam,self.query_cosine,self.range_max)
            if center is not None:objects.append(dict(eid=self.target_eid,range_m=center[0],closing_mps=center[1]))
            for cloud in clouds:
                now=cloud.sample(physical.time)
                center=candidate_geometry(now.p,now.v,physical,beam,self.query_cosine,self.range_max)
                if center is not None:objects.append(dict(eid=cloud.eid,range_m=center[0],closing_mps=center[1]))
            candidates=search_cells(objects,self.R,self.D,self.range_valid,self.doppler_valid,
                self.channels['distance'],self.channels['dopplerSpeed'],dr,dd)
        else:candidates=[dict(range_m=self.R,closing_mps=self.D)]
        attempts=[];best=None
        for candidate in candidates:
            Rc,Dc=candidate['range_m'],candidate['closing_mps']
            returns=[]
            # Use the unchanged #118 aircraft computation at this candidate's
            # explicit gate center, then restore persistent observer values.
            saved=self.R,self.D
            self.R,self.D=Rc,Dc
            air=self.aircraft_echo(physical,target,previous)
            self.R,self.D=saved
            rc,dc=self.channels['distance'],self.channels['dopplerSpeed']
            returns.append(dict(eid=self.target_eid,kind='aircraft',n=air['n'],range_m=air['range_m'],closing_mps=air['closing_mps'],
                angular_width=air['angular_width'],range_width=air['range_width'],doppler_width=air['doppler_width'],
                signal=air['raw_over_floor']*air['channel_gain'][0]*air['channel_gain'][1],
                noise=air['raw_over_floor']*(math.prod(air['channel_reference_noise'])
                    if min(air['channel_reference_noise'])>3.99999987306209e-19 else max(air['channel_reference_noise'])),
                channel_gain=air['channel_gain'],sigma_m2=air['sigma_m2']))
            for cloud in clouds:
                echo=self.cloud_echo(physical,cloud,(previous[0],previous[1],previous[2],self.previous_time),Rc,Dc)
                if echo is not None:returns.append(echo)
            basis=receiver_basis(self.angular.beam(physical.q))
            common=self.params.gameplay['radarCommon']
            if common['directionWithinMainLobe']:
                zones=[max(self.direction_floor,f(self.rx_shape['h']*common['angleFinderZoneToMainBeamWidth']))]*2
                for echo in returns:
                    raw_n=list(echo['n'])
                    echo['n'],folded=fold_main_lobe(raw_n,basis,f(echo['signal']),zones)
                    echo['direction_folded']=folded
                    if folded:echo['unfolded_n']=raw_n
            aggregate=aggregate_returns(returns,rc['width'],dc['width'],basis=basis)
            multiplier=self.params.gameplay['radarCommon']['singleTargetDetectionSensitivityMult']
            score=multiplier*aggregate['signal']/(1+aggregate['noise'])
            size_limit=self.params.get('guidance/radarSeeker','targetSizeToBeamWidthMax')
            width_limit=f(f(size_limit+size_limit)*self.rx_shape['knee'])
            angle_accepted=aggregate['signal']>0 and all(v<width_limit for v in aggregate['angular_width'])
            result=dict(aggregate,score=score,candidate=True,accepted=bool(angle_accepted and score>1.),
                angular_extent_accepted=angle_accepted,angular_width_limit=width_limit,
                gate_centers=[Rc,Dc],source_objects=len(returns),object_returns=returns)
            if aggregate['signal']>0:
                result.update(local_n=fv(tv(matrix(physical.q),aggregate['n'])))
            attempts.append(dict(center=[Rc,Dc],score=score,accepted=result['accepted'],objects=aggregate['object_count'],
                candidate_eid=candidate.get('eid'),bins=candidate.get('bins'),rank=candidate.get('rank')))
            best=result
            if result['accepted']:break
        if best is None:best=dict(candidate=False,accepted=False,score=0.,object_count=0,contributions=[])
        best['search_attempts']=attempts;best['candidate_count']=len(candidates)
        self.last_objects=best.get('object_returns',[]);self.last_measurement=best
        return best


class FiniteAircraftSeeker148(MultiObjectSeeker):
    """The full-Unit aircraft route, with explicit resource-backed bounds.

    The external virtual scene selects the whole aircraft. A selected collision
    part would require its own posed center bounds; no part is manufactured here.
    """
    def __init__(self,*args,center_bounds,dimension_bounds,responses=None,aircraft_name=None,**kwargs):
        super().__init__(*args,responses=responses,**kwargs)
        self.aircraft_name=aircraft_name
        self.center_bounds=center_bounds;self.dimension_bounds=dimension_bounds
        self.patterns=[]
        if responses is None:
            attenuation=f(self.params.get('guidance/radarSeeker','sideLobesAttenuation'))
            # .80 143FDB12C..15B writes this coefficient into both pattern axes.
            decay=f(attenuation*f(-.230258509516716)) if attenuation<f(-3.99999987306209e-19) else 0.
            for name in ('transmitter','receiver'):
                shape=antenna_shape(self.raw[name]['antenna'])
                # Scalar loader supplies equal axes and ratio=1; 147994BD1=1.
                self.patterns.append({0x30:1.,0x34:shape['k'],0x38:shape['side'],
                    0x40:shape['knee'],0x44:decay,0x48:1.,0x4c:shape['k'],
                    0x50:shape['side'],0x58:shape['knee'],0x5c:decay,0x60:1.,0x64:True})
        widths=[v['value'] for v in json.loads((Path(__file__).parent/'data/sensor_initializer.json').read_text())
            if v['path']=='config/gameplay.blk/radarCommon' and v['key']=='sizeDependentDopplerSpeedWidth']
        if not widths or any(v!=0. for v in widths):
            raise ValueError('Finite Doppler geometry requires the verified zero size-dependent width route')

    def aircraft_echo(self,physical,target,previous):
        pq,pp,pv,pt=previous
        beam=self.angular.beam(physical.q);old_beam=self.angular.beam(pq)
        frame=receiver_basis(beam);old_frame=receiver_basis(old_beam)
        def geometry(position,pose,origin,antenna):
            return point_geometry(antenna_frame=antenna,antenna_position=position,
                target_origin=origin,target_q=pose,center_bounds=self.center_bounds,dimension_bounds=self.dimension_bounds)
        current=geometry(physical.p,target[2],target[0],frame)
        # FF5C74..5D1B copies the current transmitter into both endpoints;
        # FF5D26..5DB6 supplies previous/current receiver frames separately.
        # FE55A2 copies those four contexts to FDDA90+30/+70/+B0/+F0.
        old_tx=geometry(physical.p,pt[2],pt[0],frame)
        old_rx=geometry(pp,pt[2],pt[0],old_frame)
        if self.responses is not None:
            gains=[self.responses.finite_gain(leg,current) for leg in ('transmitter','receiver')]
            old_gains=[self.responses.finite_gain(leg,g) for leg,g in zip(('transmitter','receiver'),(old_tx,old_rx))]
        else:
            gains=[finite_pattern_gain(current,native_fields=p) for p in self.patterns]
            old_gains=[finite_pattern_gain(g,native_fields=p) for g,p in zip((old_tx,old_rx),self.patterns)]
        distance=current.range_m;direction=list(current.direction_world)
        closing=fdot(fv(sub(physical.v,target[1])),direction)
        old_distance=f(.5*f(f(old_tx.range_m+old_rx.range_m)-f(norm(sub(pp,physical.p)))))
        old_closing=f(.5*f(fdot(fv(sub(physical.v,pt[1])),old_tx.direction_world)+fdot(fv(sub(pv,pt[1])),old_rx.direction_world)))
        older=endpoint_geometry(tx=old_tx,rx=old_rx,bistatic=f(norm(sub(pp,physical.p)))>1.)
        rc,dc=self.channels['distance'],self.channels['dopplerSpeed']
        # FDC982 explicitly writes zero in the disabled size-Doppler route;
        # FBB9C5 zeros normal-body reflection+30, which the aircraft branch keeps.
        widths=reflection_widths(older_endpoint=older,range_signal_min=rc['signalWidthMin'],
            doppler_signal_min=dc['signalWidthMin'],older_doppler_geometry_width=0.,
            current_doppler_geometry_width=0.,reflection_doppler_width=0.)
        period=dc['maxUnambiguousValue'];wrapped=self.D+(closing-self.D+period/2)%period-period/2
        rg,rn,Rm=self.channel_response(old_distance,distance,widths['range_width'],self.R,rc)
        dg,dn,Dm=self.channel_response(old_closing,wrapped,widths['doppler_width'],self.D,dc)
        mu=tv(matrix(target[2]),mul(direction,-1.))[0]
        sigma=(self.airframe['explicit_rcs'] if self.airframe.get('explicit_rcs') is not None else
            self.responses.aircraft_sigma(self.airframe,mu) if self.responses is not None else airframe_sigma(self.airframe,mu))
        if self.responses is not None:
            raw=self.responses.receiver_power(sigma,tx=(gains[0],distance,old_gains[0],old_tx.range_m),
                rx=(gains[1],distance,old_gains[1],old_rx.range_m))
        else:
            rx=self.raw['receiver'];raw=sigma/rx['rcs']*rx['range']**4/.5625
            raw*=max(gains[0]/distance**2,old_gains[0]/old_tx.range_m**2)*max(gains[1]/distance**2,old_gains[1]/old_rx.range_m**2)
        if distance>self.range_max:raw=0.
        return dict(widths,sigma_m2=sigma,range_m=Rm,closing_mps=Dm,n=direction,
            antenna_power=gains,antenna_endpoints=[gains,old_gains],
            channel_gain=[rg,dg],channel_reference_noise=[rn,dn],raw_over_floor=raw,
            center_world=list(current.center_world),projected_dimensions_m=list(current.widths_forward_up_side_m))
