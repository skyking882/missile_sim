function [O,M] = observe(O,X,target,Ts,options)
% Filtered geometric direction and LOS rate, with explicit absence/noise.
o=missile.terminal.defaultOptions(options);
r=target.position(:)-X.position(:); v=target.velocity(:)-X.velocity(:);
R=norm(r); n=r/max(R,eps); w=cross(v,r)/max(R^2,eps);
a=[atan2(n(3),n(1));atan2(n(2),hypot(n(1),n(3)))];
valid=R>eps;
for j=1:size(o.absentIntervals,1)
 valid=valid && ~(X.time>=o.absentIntervals(j,1) && X.time<o.absentIntervals(j,2));
end
if isempty(O), O=struct('angles',a,'omega',w); end
if valid
 gain=1; if o.observerTau>0, gain=-expm1(-Ts/o.observerTau); end
 da=a+o.directionNoise(:)-O.angles; da(1)=atan2(sin(da(1)),cos(da(1)));
 O.angles=O.angles+gain*da; O.omega=O.omega+gain*(w+o.rateNoise(:)-O.omega);
end
az=O.angles(1); el=O.angles(2);
M=struct('direction',[cos(el)*cos(az);sin(el);cos(el)*sin(az)],...
 'omega',O.omega,'valid',valid,'range',R,'closing',-dot(v,n),...
 'kind','explicit filtered geometric observer');
end
