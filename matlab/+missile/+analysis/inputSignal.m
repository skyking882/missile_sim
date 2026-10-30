function u=inputSignal(t,o)
% Start and width are rounded to nearest sample; pulse has at least one sample.
Ts=1/48;k=round(t/Ts);start=round(o.startTime/Ts);width=max(1,round(o.pulseWidth/Ts));
u=zeros(2,1);active=k>=start;
switch lower(o.signal)
 case 'step'
 case {'pulse','impulse'},active=active && k<start+width;
 otherwise,error('missile:signal','Signal must be step, pulse or impulse (finite pulse in nonlinear mode).');
end
u(o.axis)=o.amplitude*active;
end
