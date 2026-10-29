"""Dry-air motor/aerodynamics/rigid-body equations reconstructed as Python.

No engine, emulator, game process, trajectory samples or fitted coefficients.
Normal geometry branch documented in REPORT_CONTROL_CHAIN.zh-CN.txt.
"""
from dataclasses import dataclass
from .python_math118 import *


@dataclass
class PhysicalState:
    p: list
    v: list
    q: list
    omega: list
    time: float


def release_velocity(q, velocity, start_speed, transverse_multiplier):
    r = matrix(q)
    local = tv(r, velocity)
    local[0] += start_speed
    local[1] *= transverse_multiplier
    local[2] *= transverse_multiplier
    return fv(mv(r, local))


def normal_plane_projection_numerator(direction, forward):
    """(|d|² I - d dᵀ) forward, evaluated with antisymmetric minors.

    Dividing by |d|² gives the ordinary normal-plane projection, but that
    positive scale cancels when normalizing. Keeping the numerator preserves
    the existing near-zero normalization threshold for non-unit d. Factoring
    the matrix avoids subtracting two almost-equal O(1) vectors near alignment.
    """
    x,y,z=direction;ax,ay,az=forward
    xy=x*ay-y*ax;xz=x*az-z*ax;yz=y*az-z*ay
    return [-y*xy-z*xz,x*xy-z*yz,x*xz+y*yz]


def aerodynamic_force(velocity, q, omega, control, height, p, wind):
    r = matrix(q)
    forward = unit([row[0] for row in r])
    ey, ez = [row[1] for row in r], [row[2] for row in r]
    rel = sub(velocity, wind)
    speed = norm(rel)
    rho, sound = atmosphere(height)
    mach = speed/sound
    pressure = .5*rho*speed*speed
    direction = unit(add(rel, mv(r, cross(omega, [p['lever'], 0., 0.]))))
    cd0 = p['CxK']*mach_factor(mach)
    cxi = 2.4352500438690186*mach_factor(mach)+.25001898407936096 if p['useCxiMach'] else 1.
    cy = p['CyK']*curve(p['CyMachMult'], mach)

    def force(d):
        cosine = dot(d, forward)
        sine2 = max(0., 1-cosine*cosine)
        cd = cd0+p['CxK']*p['CxAoA']*sine2*cxi
        cl = cy*math.sqrt(sine2)
        if cl > p['CyMaxAoA']: cl = max(0., 2*p['CyMaxAoA']-cl)
        if cosine < 0: cl = -cl
        lift = unit(normal_plane_projection_numerator(d,forward))
        return [-pressure*p['S']*cd*d[i]+pressure*p['Sw']*cl*lift[i] for i in range(3)]

    delta = [control[i]*p['finsAoa'][i] for i in range(2)]
    delta_norm = math.hypot(*delta)
    if pressure*delta_norm > p['Kfin']:
        delta = mul(delta, p['Kfin']/(pressure*delta_norm))
    virtual = unit(add(direction, add(mul(ez, delta[0]), mul(ey, delta[1]))))
    main, tail = force(direction), force(virtual)
    tail_body=tv(r,tail);lever=p['lever']
    moment=[0.,-lever*tail_body[2],lever*tail_body[1]]
    return dict(force=main, moment=moment, pressure=pressure, delta=delta, virtual_force=tail)


class Physics:
    def __init__(self, parameters, state):
        self.p, self.state = parameters, state
        self.control = [0., 0.]
        self.launch = state.time
        self.last = None

    def motor(self, elapsed):
        p = self.p
        tau = elapsed-p['fireDelay']
        burned = clip(tau/p['timeFire'], 0., 1.)
        return (p['force'] if 0 < tau < p['timeFire'] else 0.,
                (p['mass']-p['massEnd'])*burned)

    def step(self, end_time, wind):
        p, s = self.p, self.state
        dt = f(end_time-s.time)
        if dt <= 0: raise ValueError('Nonpositive physical step')
        thrust, lost = self.motor(end_time-self.launch)
        mass = p['mass']-lost
        inertia = mul(p['inertia0'], mass/p['mass'])
        external = mul([row[0] for row in matrix(s.q)], thrust)
        aero = aerodynamic_force(s.v, s.q, s.omega, self.control, s.p[1], p, wind)
        force = add(aero['force'], external)
        moment = aero['moment']
        scale = aero['pressure']*p['S']*p['length']**2
        damping = []
        for i, k in enumerate((.01, .05, .05)):
            raw = -scale*k*p['WdK'][i]*s.omega[i]
            braking = -(inertia[i]*s.omega[i]/dt+moment[i])
            damping.append(braking if abs(raw) > abs(braking) else raw)
        moment = add(moment, damping)
        ix, iy, iz = inertia
        wx, wy, wz = s.omega
        gyro = [(iz-iy)*wy*wz, (ix-iz)*wz*wx, (iy-ix)*wx*wy]
        alpha = [(moment[i]+gyro[i])/inertia[i] for i in range(3)]
        accel = mul(force, 1/mass)
        accel[1] -= p['gravity']
        magnitude = norm(accel)
        if magnitude > p['maxAcceleration']: accel = mul(accel, p['maxAcceleration']/magnitude)
        old_v, old_w = list(s.v), list(s.omega)
        s.p = fv([s.p[i]+s.v[i]*dt+.5*accel[i]*dt*dt for i in range(3)])
        s.v = fv([clip(s.v[i]+accel[i]*dt, -2000., 2000.) for i in range(3)])
        s.q = attitude_step(s.q, old_w, alpha, dt)
        s.omega = fv([clip(old_w[i]+alpha[i]*dt, -15., 15.) for i in range(3)])
        speed = norm(s.v)
        if speed > p['endSpeed']: s.v = fv(mul(s.v, p['endSpeed']/speed))
        s.time = end_time
        self.last = dict(thrust=thrust, mass_lost=lost, mass=mass, force=force,
                         moment=moment, alpha=alpha, acceleration=accel,
                         feedback=fv(mul(sub(s.v, old_v), 1/dt)),
                         control_consumed=list(self.control), pressure=aero['pressure'])
        return self.last
