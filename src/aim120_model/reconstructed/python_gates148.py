"""143FDC9F0 channel response, with explicit caller context.

The channel layout is independently checked by fixed .86 143FDD5B0:
present/min/period/max/signalWidthMin/width/refWidth/refCoefficient/leakMult.
No game instructions are executed. Every dynamic input is required.
"""
import math
from .python_math118 import f


def swept_channel148(first,last,width,gate_center,channel,*,use_reference_width,
                     dynamic_gate_width,minimum_gate_width,excluded_intervals):
    """CA49..CE69, retaining float32 operation order and branch guards.

    Excluded intervals are the sorted, merged notch intervals built by
    FDD656..704/FDDA0D..A8A. Empty intervals are an explicit caller choice.
    Returns signal gain, reference noise gain, and reported measurement;
    None is the native failure return at FDCEC0.
    """
    first,last,width=map(f,(first,last,width))
    if not channel['presents']:
        return 1.,0.,last
    period=f(channel['maxUnambiguousValue'])
    a=f(math.fmod(first,period));b=f(math.fmod(last,period))
    inverse=f(1./width) if abs(width)>f(3.99999987306209e-19) else 0.
    turns=f(math.ceil(f(width/period)))
    lo,hi=min(a,b),max(a,b)
    window=f(channel['width']);lower=f(channel['minValue']);upper=f(channel['maxValue'])
    half=f(width*.5)
    if width<period:
        lo=f(lo-half);hi=f(hi+half)
        if window<=0.:
            lo=max(lo,lower);hi=min(hi,upper)
    else:
        lo,hi=lower,upper
        if window>0.:
            reported=f(f(math.fmod(f(b-lower),f(upper-lower)))+lower)
    leak=f(channel['leakMult'])
    bounds=[];start=lower
    for left,right in excluded_intervals:
        bounds.append((start,f(left)));start=f(right)
    if excluded_intervals:bounds.append((start,upper))

    def mass(left,right):
        overlap=min(width,max(0.,f(min(hi,right)-max(lo,left))))
        return f(f(f(width-overlap)*leak)+overlap)

    if window<=0.:
        if excluded_intervals:
            total=0.
            for left,right in bounds:
                overlap=min(width,max(0.,f(min(right,hi)-max(left,lo))))
                total=f(total+overlap)
                total=f(total+f(f(width-overlap)*leak))
            signal=f(f(inverse*turns)*total)
        else:
            signal=f(f(inverse*turns)*mass(lo,hi))
        return signal,0.,last

    window=max(window,min(f(minimum_gate_width),width))
    center=f(gate_center);dynamic=f(dynamic_gate_width)
    if dynamic>0.:
        dynamic_half=f(dynamic*.5)
        left=f(center-dynamic_half);right=f(center+dynamic_half)
        if lo<right and hi>left:
            center=min(right,max(left,f(f(a+b)*.5)))
    half=f(window*.5)
    left=min(upper,max(lower,f(center-half)))
    right=min(upper,max(lower,f(center+half)))
    if excluded_intervals:
        total=0.
        for start,end in bounds:
            overlap=min(width,max(0.,f(min(end,right,hi)-max(start,left,lo))))
            total=f(total+overlap)
            total=f(total+f(f(width-overlap)*leak))
    else:
        total=mass(left,right)
    signal=f(total*f(inverse*turns))
    measurement=b if width<period else reported
    delta=f(f(channel['refWidth'])-f(channel['width']))
    coefficient=f(f(channel['width'])/delta) if abs(delta)>f(3.99999987306209e-19) else 0.
    if not (f(.0010000000474974513)<coefficient<f(.9990000128746033) and use_reference_width):
        return signal,0.,measurement
    half=f(f(channel['refWidth'])*.5)
    reference=f(min(hi,upper,max(lower,f(center+half)))-max(lo,lower,min(upper,f(center-half))))
    reference=f(f(f(width-reference)*leak)+reference)
    if reference<0.:
        return None
    noise=f(f(max(0.,f(min(reference,width)-total))*inverse)*coefficient)
    return signal,noise,measurement
