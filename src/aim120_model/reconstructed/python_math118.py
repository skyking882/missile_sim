"""Scalar reconstructions used by the Python missile model. No PE execution.

Sources: implementation_interfaces.py, native_formulas.py and PORTABLE_NOISE75.py
in the adjoining reverse-engineering evidence. Native coordinates are retained.
"""
import math
import struct


def f(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def add(a, b): return [x+y for x, y in zip(a, b)]
def sub(a, b): return [x-y for x, y in zip(a, b)]
def mul(a, k): return [x*k for x in a]
def dot(a, b): return sum(x*y for x, y in zip(a, b))
def norm(a): return math.sqrt(dot(a, a))
def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
def unit(a):
    length = norm(a)
    return mul(a, 1/length) if length > 3.99999987306209e-19 else [0., 0., 0.]
def clip(x, low, high): return max(low, min(high, x))
def fv(a): return list(map(f, a))


def matrix(q):
    x, y, z, w = q
    d = 2*w*w-1
    return [[2*x*x+d, 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 2*y*y+d, 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 2*z*z+d]]


def mv(r, v): return [dot(row, v) for row in r]
def tv(r, v): return [sum(r[j][i]*v[j] for j in range(3)) for i in range(3)]


def hamilton(a, b):
    x, y, z, w = a
    X, Y, Z, W = b
    return [w*X+x*W+y*Z-z*Y, w*Y-x*Z+y*W+z*X,
            w*Z+x*Y-y*X+z*W, w*W-x*X-y*Y-z*Z]


def euler_q(bank, heading, attitude):
    # Same native Euler order as the existing independent aircraft adapter.
    cy, sy = math.cos(heading/2), math.sin(heading/2)
    cp, sp = math.cos(attitude/2), math.sin(attitude/2)
    cr, sr = math.cos(bank/2), math.sin(bank/2)
    return hamilton(hamilton([0., sy, 0., cy], [0., 0., sp, cp]), [sr, 0., 0., cr])


def native_sincos_half(theta):
    theta = f(theta)
    h = f(-theta*.5)
    a = f(math.copysign(.5, h)-f(theta*f(.31830987334251404)))
    k = math.trunc(a)
    t = f(f(k*f(-1.5707963705062866))+h)
    t2 = f(t*t)
    cp = f(f(f(f(f(f(f(-.0013602249091491103)*t2)+f(.04165669530630112))*t2)
                 +f(-.4999990165233612))*t2)+1)
    sp = f(f(f(f(f(f(f(-.0001950727018993348)*t2)+f(.00833207555115223))*t2)
                 +f(-.16666652262210846))*f(t2*t))+t)
    s, c = (cp, sp) if k & 1 else (sp, cp)
    return f((-1 if k & 2 else 1)*s), f((-1 if (k+1) & 2 else 1)*c)


def attitude_step(q, omega, alpha, dt):
    halfdt2 = f(f(dt*dt)*.5)
    theta = [f(f(w*dt)+f(a*halfdt2)) for w, a in zip(omega, alpha)]
    sx, cx = native_sincos_half(theta[0])
    sy, cy = native_sincos_half(theta[1])
    sz, cz = native_sincos_half(theta[2])
    delta = [sx*cy*cz+cx*sy*sz, cx*sy*cz+sx*cy*sz,
             cx*cy*sz-sx*sy*cz, cx*cy*cz-sx*sy*sz]
    return fv(unit(hamilton(q, delta)))


def curve(points, x):
    if not points:
        raise ValueError('An explicit loaded curve is required')
    if x <= points[0][0]: return points[0][1]
    for (a, y), (b, z) in zip(points, points[1:]):
        if b <= a: raise ValueError('Unsupported non-increasing native curve')
        if x < b: return y+(z-y)*(x-a)/(b-a)
    return points[-1][1]


def atmosphere(h):
    k, H = min(h, 18300.), max(h, 18300.)
    rho = 1.225*18300/H*((((2.287190065873038e-19*k-5.83556030646186e-14)*k
                           +3.5311800150594763e-9)*k-9.593870345270261e-5)*k+1)
    temperature = 288.1600036621094*((((3.9730601514748866e-18*k-5.711039686951899e-14)*k
                                      +2.1806899341836328e-10)*k-2.277120074722916e-5)*k+1)
    return rho, 20.100000381469727*math.sqrt(temperature)


def mach_factor(m):
    if m < .61: return .308
    if m < 1: return .308+.505*(m-.61)**2.31
    if m < 1.4: return .551+.4485*(m-1)**.505*math.exp(-5.68*(m-1))
    if m < 4: return m/(.356*m*m+2.237*m-1.4)
    return .302


class Noise:
    """Original fixed startup LCG/permutation, evaluated in Python."""
    def __init__(self, seed):
        p, g = list(range(256)), []
        s = seed
        for _ in range(256):
            a = []
            for _ in range(6):
                s = (1103515245*s+12345) & 0xffffffff
                a.append((((s >> 16) & 511)-256)/256)
            g.append(a[0])
        for i in range(255, 0, -1):
            s = (1103515245*s+12345) & 0xffffffff
            j = (s >> 16) & 255
            p[i], p[j] = p[j], p[i]
        self.p, self.g = p, g

    def at(self, x):
        x = f(x)
        a = math.floor(x)
        t = f(x-a)
        u = f((3-2*t)*f(t*t))
        left = f(self.g[self.p[a & 255]]*t)
        right = f(self.g[self.p[(a+1) & 255]]*f(t-1))
        return f(left+f(u*f(right-left)))
