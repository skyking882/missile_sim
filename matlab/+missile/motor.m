function [thrust,mass]=motor(P,t,height)
r=P.rocket;previous=r.mass;finish=0;thrust=0;lost=0;
for index=0:2
if index==0,s='';else,s=num2str(index);end
if ~isfield(r,['timeFire' s]),break,end
duration=r.(['timeFire' s]);delay=0;if isfield(r,['fireDelay' s]),delay=r.(['fireDelay' s]);end
start=finish+delay;finish=start+duration;dm=previous-r.(['massEnd' s]);previous=r.(['massEnd' s]);
age=t-start;if duration>0,u=age/duration;else,u=double(age>0);end
lost=lost+dm*max(0,min(1,u));
if age>0 && age<duration,thrust=thrust+r.(['force' s])*(1+P.motorShape.tilt*(1-2*u)+P.motorShape.bulge*(6*u*(1-u)-1));end
end
if isfield(r,'extPressureToThrustMult')
k=r.extPressureToThrustMult;if isvector(k),k=reshape(k,2,[])';end;k=sortrows(k);[rho,sound]=missile.atmosphere(height);thrust=thrust*missile.curve(k,rho*sound^2/1.4);
end
mass=r.mass-lost;
end
