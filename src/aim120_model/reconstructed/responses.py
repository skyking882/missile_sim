"""Common independent response functions, parameterized by unchanged resources.

Only dimensionless internal shapes are shared. This module has no missile or
aircraft IDs, fitted forces, durations, RCS values, beam widths or receiver K.
"""
import copy
import math
from .resources import read
from .python_math118 import atmosphere,curve,dot,f
from .python_chaff148 import envelope


def aircraft_inputs(name):
    """Original dimensions/aspect coefficients; absent scales are not guessed."""
    try:raw=read('aircraft_inputs.json')['aircraft'][name]
    except KeyError:raise ValueError('No original aircraft dimensions/aspects for '+str(name)+'; supply target_airframe or target_rcs_m2') from None
    g=read('raw/gameplay.blk')['sensorsConstants']
    return dict(length=raw['Length'],span=raw['Span'],aspects=[raw.get('radarCrossSectionAspectMult'+side,
        g['aircraftRadarCrossSectionAspectMult'+side]) for side in ('Front','Side','Rear')])


class SharedResponses:
    def __init__(self,radar=None,*,shape_overrides=None):
        config=read('response_shapes.json')
        self.shapes=copy.deepcopy(config['shapes']);self.radar=radar
        for group,values in (shape_overrides or {}).items():
            if group not in self.shapes:raise ValueError('Unknown response-shape group: '+group)
            for name,value in values.items():
                if name not in self.shapes[group] or not isinstance(value,(int,float)) or not math.isfinite(value):
                    raise ValueError('Invalid shared response shape: '+group+'/'+name)
                self.shapes[group][name]=float(value)
        for group in ('antenna','receiver','aircraft','chaff'):
            if any(value<=0 for value in self.shapes[group].values()):raise ValueError('Shared response shapes must be positive: '+group)
        m=self.shapes['motor']
        if abs(m['tilt'])+abs(m['bulge'])>=1:raise ValueError('Motor shape must remain positive over a stage')

    def point_gain(self,leg,beam,direction):
        cosine=dot(beam,direction)
        if cosine<0:return 0.
        theta=math.acos(max(-1.,min(1.,cosine)))
        return self.angular_gain(theta,self.radar[leg]['antenna'])

    def angular_gain(self,theta,antenna):
        p=self.shapes['antenna'];half=math.radians(antenna['angleHalfSens']*.5)
        if half<=0:raise NotImplementedError('A positive original antenna angleHalfSens is required')
        side=10**(antenna['sideLobesSensitivity']/10);weight=min(1.,side**p['sidelobe_weight_power']);x=abs(theta)/half
        return ((1-weight)*math.exp(-(x/p['main_width'])**p['main_power'])+
                weight*math.exp(-(x/p['side_width'])**p['side_power']))

    def finite_gain(self,leg,geometry):
        if geometry.direction_antenna[0]<0:return 0.
        up,side=geometry.reduced_up_side
        theta=math.asin(min(1.,math.hypot(up,side)))
        return self.angular_gain(theta,self.radar[leg]['antenna'])

    def receiver_power(self,sigma,*,tx,rx):
        p=self.shapes['receiver'];transmitter=self.radar['transmitter'];receiver=self.radar['receiver']
        if receiver['rcs']<=0 or receiver['range']<=0:raise NotImplementedError('Original positive receiver rcs/range are required')
        power=transmitter['power']
        if power<=0:return 0.
        def endpoint_mean(leg):
            a,b=leg[0]/leg[1]**2,leg[2]/leg[3]**2
            scale=max(a,b)
            if scale==0:return 0.
            return scale*(.5*((a/scale)**p['endpoint_power']+(b/scale)**p['endpoint_power']))**(1/p['endpoint_power'])
        normalization=p['sensitivity']*receiver['range']**4/receiver['rcs']*power/(power+p['power_unit'])
        return normalization*sigma*endpoint_mean(tx)*endpoint_mean(rx)

    def motor(self,stages,elapsed,*,pressure_knots=None,height=0.):
        p=self.shapes['motor'];thrust=lost=0.
        for start,duration,force,mass_lost in stages:
            age=f(f(elapsed)-f(start));u=f(age/duration) if duration else float(age>0)
            lost=f(lost+f(mass_lost*f(max(0.,min(1.,u)))))
            if 0.<age<duration:
                shape=1+p['tilt']*(1-2*u)+p['bulge']*(6*u*(1-u)-1)
                thrust=f(thrust+force*shape)
        if pressure_knots is not None:
            knots=[pressure_knots[:2],pressure_knots[2:]] if len(pressure_knots)==4 and isinstance(pressure_knots[0],(int,float)) else pressure_knots
            rho,sound=atmosphere(height)
            # Sorting a copy preserves descending original resource knots.
            thrust=f(thrust*curve(sorted(knots),rho*sound*sound/1.4))
        return thrust,lost

    def aircraft_sigma(self,airframe,mu):
        p=self.shapes['aircraft'];front,side,rear=airframe['aspects']
        end=front if mu>=0 else rear
        if min(front,side,rear,airframe['length'],airframe['span'])<=0:raise NotImplementedError('Positive original aircraft dimensions/RCS aspect factors are required')
        aspect=side*math.exp(math.log(end/side)*min(1.,abs(mu))**p['aspect_power'])
        return p['area_fraction']*airframe['length']*airframe['span']*aspect

    def cloud_radius(self,parameters,old,speed,height,dt):
        p=self.shapes['chaff'];g=parameters.global_params;rho=atmosphere(height)[0]
        v0=p['speed_scale']/g['chaffDispersionFromSpeed']
        density=p['density_scale']*g['chaffDispersionFromAirViscosity']
        target=p['radius_scale']*g['chaffCloudMaxRadius']*parameters.radius_mult*(speed/(speed+v0))/(1+(rho/density)**p['density_power'])
        tau=p['response_scale']*rho*g['chaffDispersionSpeedFromAirViscosity']
        return f(old+max(0.,target-old)*(-math.expm1(-dt/tau))) if tau>0 else f(max(old,target))

    def cloud_intrinsic(self,parameters,age,radius):
        p=self.shapes['chaff'];raw=parameters.rocket;radar=parameters.signature['radar'];cap=parameters.global_params['chaffCloudRcsMaxRadius']
        area=radius*radius/(1+(radius/cap)**p['reflection_power'])**(2/p['reflection_power'])
        body=p['body_area_fraction']*(raw['caliber']+raw['length'])**2
        return f(radar['multiplier0']*(body+area)*envelope(f(age),radar['profile']))
