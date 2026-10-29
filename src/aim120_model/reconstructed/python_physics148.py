"""#148 physical caller corrections and explicit reference/fitted motor choice.

14143E2CB/1432920EF pass the caller's physical dt, independently of wrapper
clock subtraction. 143293F74..F8E and 143294378..43BC use nominal inertia.
The frozen #118 implementation remains available for the saved comparison.
An injected response object replaces only thrust and its independent mass-loss
schedule; caller dt, integration and control ordering remain the same.
"""
from .python_physics118 import Physics,aerodynamic_force
from .python_math118 import *


def thrust_force148(q, thrust):
    """14142D321..449: float32 forward column and force preparation."""
    x,y,z,w=q
    forward=[f(f(f(f(x*x)+f(w*w))*2.)-1.),
             f(f(f(x*y)+f(w*z))*2.),
             f(f(f(x*z)-f(w*y))*2.)]
    return [f(v*thrust) for v in forward]


def integrate_translation148(position,velocity,acceleration,dt):
    """1432944EE..464E: double a; float32 increments, sums and stores."""
    halfdt2=f(f(dt*dt)*.5)
    displacement=[f(f(v*dt)+f(a*halfdt2)) for v,a in zip(velocity,acceleration)]
    new_p=[f(p+d) for p,d in zip(position,displacement)]
    new_v=[f(clip(f(v+f(a*dt)),-2000.,2000.)) for v,a in zip(velocity,acceleration)]
    return new_p,new_v


class Physics148(Physics):
    def __init__(self,parameters,state,*,physical_dt,responses=None):
        super().__init__(parameters,state)
        self.physical_dt=f(physical_dt)
        self.responses=responses

    def motor(self,elapsed):
        if self.responses is not None:return self.responses.motor(elapsed)
        # Single constant-thrust segment, 143AFD900. Stage evaluation and
        # cumulative mass loss are floats, including the division first.
        p=self.p;tau=f(f(elapsed)-p['fireDelay'])
        ratio=f(clip(f(tau/p['timeFire']),0.,1.))
        return (p['force'] if 0.<tau<p['timeFire'] else 0.,
                f(f(p['mass']-p['massEnd'])*ratio))

    def step(self,end_time,wind):
        p,s=self.p,self.state
        dt=self.physical_dt
        if dt<=0 or end_time<=s.time:raise ValueError('Nonpositive physical step')
        thrust,lost=self.motor(end_time-self.launch)
        mass=f(p['mass']-lost)
        inertia=p['inertia0']
        external=thrust_force148(s.q,thrust)
        aero=aerodynamic_force(s.v,s.q,s.omega,self.control,s.p[1],p,wind)
        # 143294096..4102 converts the separately stored float forces to
        # double before adding. This does not claim bit-exact upstream aero.
        force=add(fv(aero['force']),external);moment=fv(aero['moment'])
        scale=f(f(f(aero['pressure']*p['S'])*p['length'])*p['length'])
        # Diagonal operator: Q32(diag(WdK) Q32(scale Q32(diag(-k) w))).
        # Preserve each original float32 boundary; do not pre-multiply the
        # diagonal coefficients or reassociate the three scaling stages.
        spin=[f(w*k) for w,k in zip(s.omega,(f(-.01),f(-.05),f(-.05)))]
        raw_damping=[f(f(w*scale)*wd) for w,wd in zip(spin,p['WdK'])]
        braking=[-f(f(f(inertia[i]*s.omega[i])*f(1./dt))+moment[i]) for i in range(3)]
        damping=[b if abs(raw)>abs(b) else raw for raw,b in zip(raw_damping,braking)]
        moment=[f(a+b) for a,b in zip(moment,damping)]
        ix,iy,iz=inertia;wx,wy,wz=s.omega
        gyro=[f(f(f(iz-iy)*wy)*wz),f(f(f(ix-iz)*wz)*wx),f(f(f(iy-ix)*wx)*wy)]
        alpha=[f(f(moment[i]+gyro[i])/inertia[i]) for i in range(3)]
        # 143294201..42BE forms float32 weight, then double force sums
        # multiplied by the float32 reciprocal of current mass.
        inverse_mass=f(1./mass)
        accel=[force[i]*inverse_mass for i in range(3)]
        accel[1]=(force[1]-f(mass*p['gravity']))*inverse_mass
        magnitude=norm(accel)
        if magnitude>p['maxAcceleration']:accel=mul(accel,p['maxAcceleration']/magnitude)
        old_v,old_w=list(s.v),list(s.omega)
        s.p,s.v=integrate_translation148(s.p,s.v,accel,dt)
        s.q=attitude_step(s.q,old_w,alpha,dt)
        s.omega=[f(clip(f(old_w[i]+f(alpha[i]*dt)),-15.,15.)) for i in range(3)]
        speed=norm(s.v)
        if speed>p['endSpeed']:s.v=fv(mul(s.v,p['endSpeed']/speed))
        wrapper_dt=f(end_time-s.time);s.time=end_time
        self.last=dict(thrust=thrust,mass_lost=lost,mass=mass,force=force,moment=moment,
            # 1452BC50E..557 forms controller feedback with wrapper clock dt.
            alpha=alpha,acceleration=accel,
            feedback=[f(f(a-b)*f(1./wrapper_dt)) for a,b in zip(s.v,old_v)],
            control_consumed=list(self.control),pressure=aero['pressure'],
            external_force=external,aerodynamic_force=fv(aero['force']),
            aerodynamic_moment=fv(aero['moment']),damping=damping,
            inverse_mass=inverse_mass,
            physical_dt_s=dt,wrapper_clock_dt_s=wrapper_dt,inertia_nominal=list(inertia))
        return self.last
