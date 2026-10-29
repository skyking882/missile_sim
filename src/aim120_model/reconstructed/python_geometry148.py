"""Finite aircraft echo geometry reconstructed from saved .80 instructions.

This module runs Python arithmetic only. It does not execute a PE function.
Installed Su-30MKM bounds have a separately recorded .83 static loader chain;
they are not a historical #148 Unit-memory measurement. All poses, antenna
frames, selected-center bounds, whole-unit bounds and pattern fields are
explicit inputs. In particular there is no point-target or resource fallback.

Sources: python_geometry148_sources.json, capture_fdb490.asm,
seeker_python_sources/0x143329870.asm, v80/fn_0x143fdda90.asm.
"""
from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
from .python_math118 import f


def _sum3(a, b, c):
    return f(f(a + b) + c)


def _dot(a, b):
    return _sum3(f(a[0]*b[0]), f(a[1]*b[1]), f(a[2]*b[2]))


def _cross(a, b):
    return [f(f(a[1]*b[2])-f(a[2]*b[1])),
            f(f(a[2]*b[0])-f(a[0]*b[2])),
            f(f(a[0]*b[1])-f(a[1]*b[0]))]


def rotation(q):
    """.80 3329EA7..9FE3. The all-zero quaternion is a native identity sentinel."""
    x,y,z,w=map(f,q)
    if x==y==z==w==0.:
        return [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
    xx,yy,zz,ww=[f(v*v) for v in (x,y,z,w)]
    def diagonal(v):
        t=f(v+ww)
        return f(f(t+t)-1.)
    def twice(v):
        return f(v+v)
    xy,xz,yz,xw,yw,zw=[f(a*b) for a,b in
                       ((x,y),(x,z),(y,z),(x,w),(y,w),(z,w))]
    return [[diagonal(xx),twice(f(xy-zw)),twice(f(xz+yw))],
            [twice(f(xy+zw)),diagonal(yy),twice(f(yz-xw))],
            [twice(f(xz-yw)),twice(f(yz+xw)),diagonal(zz)]]


def _inverse_rows(frame):
    # 33299E00..9EA0 uses cofactors, not an assumed transpose.
    columns=[[f(frame[j][i]) for j in range(3)] for i in range(3)]
    rows=[_cross(columns[1],columns[2]),_cross(columns[2],columns[0]),
          _cross(columns[0],columns[1])]
    reciprocal=f(1./_dot(columns[0],rows[0]))
    return [[f(v*reciprocal) for v in row] for row in rows]


@dataclass(frozen=True)
class Bounds:
    minimum: tuple
    maximum: tuple
    provenance: str

    @property
    def center(self):
        # FDD 3FDDD8F..DDDDD: add min/max before multiplying by one half.
        return [f(f(a+b)*.5) for a,b in zip(self.minimum,self.maximum)]

    @property
    def dimensions(self):
        # FDD 3FDE258..2A0: FULL dimensions, without a one-half factor.
        return [f(b-a) for a,b in zip(self.minimum,self.maximum)]


def installed_su30mkm_bounds():
    """Read the verified .83 resource; never label it historical runtime data."""
    source=json.loads(Path(__file__).with_name('data')/'python_geometry148_sources.json'.read_text())
    if source['aircraft']!='su_30mkm' or source['installed_version']!='2.59.0.83':
        raise ValueError('Bounds source identity changed')
    return Bounds(tuple(source['bounds_min_m']),tuple(source['bounds_max_m']),
                  '2.59.0.83 static normal-loader su_30mkm_collision; not historical runtime')


@dataclass(frozen=True)
class PointGeometry:
    center_world: tuple
    relative_world: tuple
    direction_world: tuple
    direction_antenna: tuple
    range_m: float
    inverse_range: float
    widths_forward_up_side_m: tuple
    minimum_transverse_over_range: float
    reduced_up_side: tuple


def point_geometry(*,antenna_frame,antenna_position,target_origin,target_q,
                   center_bounds,dimension_bounds):
    """Geometry and finite-size beam correction from .80 143329870.

    antenna_frame is a row-major 3x3 matrix whose COLUMNS are world-space
    forward/up/side. center_bounds is the adapter +50 result (possibly selected
    collision-node bounds). dimension_bounds is the +60 whole-Unit result.

    Projected FULL dimensions are along the antenna axes, including forward;
    they are not LOS-projected dimensions. Gain correction uses the MINIMUM
    transverse dimension / R, with no extra 1/2 or atan conversion.
    """
    q=rotation(target_q);center=center_bounds.center
    world=[f(f(target_origin[i])+_dot(q[i],center)) for i in range(3)]
    relative=[f(world[i]-f(antenna_position[i])) for i in range(3)]
    r2=_sum3(f(relative[2]*relative[2]),f(relative[1]*relative[1]),f(relative[0]*relative[0]))
    distance=f(math.sqrt(r2))
    inverse=f(1./distance) if distance>f(3.99999987306209e-19) else 0.
    direction=tuple(f(v*inverse) for v in relative)
    local=tuple(f(_dot(row,relative)*inverse) for row in _inverse_rows(antenna_frame))
    dims=dimension_bounds.dimensions
    # 3329FF9..A17E then A45E..A4FB: |Q^T * antenna_axis| * dimensions.
    widths=[]
    for axis in range(3):
        antenna_axis=[f(antenna_frame[i][axis]) for i in range(3)]
        body_axis=[_dot([q[i][j] for i in range(3)],antenna_axis) for j in range(3)]
        terms=[abs(f(dims[i]*body_axis[i])) for i in range(3)]
        widths.append(_sum3(*terms))
    shrink=f(min(widths[1],widths[2])*inverse)
    def reduce(v):
        # A1A0..A1F2: signed subtraction toward zero, not a clamp on angle.
        return max(f(v-shrink),0.) if v>0. else min(f(v+shrink),0.)
    return PointGeometry(tuple(world),tuple(relative),direction,local,distance,inverse,
                         tuple(widths),shrink,(reduce(local[1]),reduce(local[2])))


def finite_pattern_gain(geometry,*,native_fields):
    """A205..A418 directional gain, given REQUIRED native pattern fields.

    native_fields must contain integer offsets 0x30,34,38,40,44,48,4C,50,58,
    5C,60,64 relative to the 3329870 context. The two pattern axes and decay
    coefficients must come from the real loader/descriptor; no guessed defaults
    or silently inferred fields are supplied here. This is the gain portion,
    separate from horizon/range/power rejection earlier in the native function.
    """
    p={off:native_fields[off] for off in
       (0x30,0x34,0x38,0x40,0x44,0x48,0x4c,0x50,0x58,0x5c,0x60,0x64)}
    if geometry.direction_antenna[0]<0.:
        return 0.
    up,side=geometry.reduced_up_side
    side2=f(side*side)
    metric=f(f(f(f(p[0x60])*f(p[0x60]))*f(up*up))+side2)
    def axis_gain(value,cosoff,koff,flooroff,kneeoff,decayoff):
        if f(1.-value)>f(abs(f(p[cosoff]))*f(p[cosoff])):
            return 1.
        gain=f(math.exp(f(-value*f(p[koff]))))
        if not p[0x64] or gain>=f(p[flooroff]):
            return gain
        angle=f(math.asin(min(1.,f(math.sqrt(value)))))
        knee=f(p[kneeoff]);denominator=f(f(1.5707963705062866)-knee)
        fraction=f(max(0.,f(angle-knee))/denominator) if abs(denominator)>f(3.99999987306209e-19) else 0.
        tail=f(math.exp(f(f(f(-2.)*f(p[decayoff]))*fraction)))
        return max(gain,f(tail*f(p[flooroff])))
    main=axis_gain(metric,0x30,0x34,0x38,0x40,0x44)
    side_gain=axis_gain(side2,0x48,0x4c,0x50,0x58,0x5c)
    return f(min(main,side_gain))


@dataclass(frozen=True)
class EndpointGeometry:
    widths_forward_up_side_m: tuple
    inverse_rx_range: float


def endpoint_geometry(*,tx,rx,bistatic):
    """FDB490 +40/+44/+48: bistatic average; monostatic receiver projection."""
    widths=tuple(f(f(a+b)*.5) for a,b in zip(tx.widths_forward_up_side_m,rx.widths_forward_up_side_m)) if bistatic else rx.widths_forward_up_side_m
    return EndpointGeometry(widths,rx.inverse_range)


def reflection_widths(*,older_endpoint,range_signal_min,doppler_signal_min,
                      older_doppler_geometry_width,current_doppler_geometry_width,
                      reflection_doppler_width):
    """FDE8F1 supplies OLD geometry widths to the normal aircraft reflection.

    FDEDB9..DDE4 adds radial/Doppler floors; FDEFF0..FDF00C multiplies OLD
    receiver inverse range by signature transverse widths, reordered side/up.
    Doppler quantities remain mandatory even when their verified value is zero.
    """
    forward,up,side=older_endpoint.widths_forward_up_side_m
    return dict(range_width=f(f(range_signal_min)+forward),
                doppler_width=f(f(max(older_doppler_geometry_width,current_doppler_geometry_width)+reflection_doppler_width)+doppler_signal_min),
                angular_width=[f(side*older_endpoint.inverse_rx_range),f(up*older_endpoint.inverse_rx_range)])


def _check():
    identity=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
    bounds=Bounds((-1.,-2.,-3.),(3.,4.,5.),'explicit analytic fixture')
    def geom(frame,q):
        return point_geometry(antenna_frame=frame,antenna_position=[0.,0.,0.],
            target_origin=[100.,0.,0.],target_q=q,center_bounds=bounds,dimension_bounds=bounds)
    g=geom(identity,[0.,0.,0.,1.])
    assert g.center_world==(101.,1.,1.) and g.widths_forward_up_side_m==(4.,6.,8.)
    assert geom(identity,[0.,0.,0.,0.])==g
    # Independent eight-corner extrema in a rotated antenna frame.
    angle=.37;c,s=math.cos(angle),math.sin(angle)
    frame=[[c,0.,s],[0.,1.,0.],[-s,0.,c]]
    q=[0.,math.sin(.23),0.,math.cos(.23)];gg=geom(frame,q);r=rotation(q)
    corners=[[sum(r[i][j]*v[j] for j in range(3)) for i in range(3)]
             for v in itertools.product(*zip(bounds.minimum,bounds.maximum))]
    for axis,width in enumerate(gg.widths_forward_up_side_m):
        samples=[sum(frame[i][axis]*v[i] for i in range(3)) for v in corners]
        assert abs(width-(max(samples)-min(samples)))<2e-6
    assert gg.widths_forward_up_side_m[0]!=g.widths_forward_up_side_m[0]
    skew=geom([[2.,0.,0.],[0.,1.,0.],[0.,0.,1.]],[0.,0.,0.,1.])
    assert skew.direction_antenna[0]==f(f(101.*.5)*g.inverse_range)
    endpoint=endpoint_geometry(tx=g,rx=g,bistatic=False)
    w=reflection_widths(older_endpoint=endpoint,range_signal_min=10.,doppler_signal_min=2.,
        older_doppler_geometry_width=0.,current_doppler_geometry_width=0.,reflection_doppler_width=0.)
    assert w['range_width']==14. and w['doppler_width']==2.
    assert w['angular_width']==[f(8*g.inverse_range),f(6*g.inverse_range)]
    p={0x30:1.,0x34:2.,0x38:.1,0x40:.2,0x44:2.,0x48:1.,
       0x4c:2.,0x50:.1,0x58:.2,0x5c:2.,0x60:1.,0x64:True}
    assert finite_pattern_gain(g,native_fields=p)==1.
    zero=Bounds((0.,0.,0.),(0.,0.,0.),'explicit point-limit fixture')
    offaxis=dict(antenna_frame=identity,antenna_position=[0.,0.,0.],
                 target_origin=[100.,30.,40.],target_q=[0.,0.,0.,1.],center_bounds=zero)
    finite=point_geometry(**offaxis,dimension_bounds=bounds)
    point=point_geometry(**offaxis,dimension_bounds=zero)
    shrink=f(6.*finite.inverse_range)
    assert finite.reduced_up_side==tuple(f(v-shrink) for v in finite.direction_antenna[1:])
    assert finite_pattern_gain(finite,native_fields=p)>finite_pattern_gain(point,native_fields=p)
    try:
        finite_pattern_gain(g,native_fields={})
    except KeyError:
        pass
    else:
        raise AssertionError('Missing native pattern fields must not get defaults')
    installed=installed_su30mkm_bounds()
    src=json.loads(Path(__file__).with_name('data')/'python_geometry148_sources.json'.read_text())
    import struct
    assert tuple(struct.unpack('<6f',bytes.fromhex(src['resource']['bbox_bytes_hex'])))==installed.minimum+installed.maximum
    print(json.dumps(dict(passed=True,checks=['off-origin center','full dimensions',
        'zero quaternion sentinel','rotated eight-corner projection','antenna-forward range width',
        'actual inverse frame','older-endpoint widths and floors','finite gain core',
        'minimum transverse shrink and finite-vs-point gain',
        'no missing-pattern defaults','source byte roundtrip'],installed_center_m=installed.center,
        installed_full_dimensions_m=installed.dimensions),indent=2))


if __name__=='__main__':
    _check()
