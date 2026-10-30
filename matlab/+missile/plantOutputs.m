function Y=plantOutputs(X,u,P,env)
% Pure evaluation of the source force/moment and signed diagnostic outputs.
if nargin<4,env=struct;end
p=P.physical;h=X.position(2);t=X.time;v=X.velocity(:);q=X.quaternion(:);w=X.omega(:);wind=zeros(3,1);
if isfield(env,'wind'),wind=env.wind(:);end
frozen=struct;if isfield(env,'frozen'),frozen=env.frozen;if isfield(frozen,'height'),h=frozen.height;end;if isfield(frozen,'elapsed'),t=frozen.elapsed;end;end
[thrust,mass]=missile.motor(P,t,h);if isfield(frozen,'mass'),mass=frozen.mass;end;if isfield(frozen,'thrust'),thrust=frozen.thrust;end
R=missile.rotation(q);forward=unit(R(:,1));rel=v-wind;speed=norm(rel);[rho,sound]=missile.atmosphere(h);mach=speed/sound;pressure=.5*rho*speed^2;
direction=unit(rel+R*cross(w,[p.lever;0;0]));m=mach;
if m<.61,mf=.308;elseif m<1,mf=.308+.505*(m-.61)^2.31;elseif m<1.4,mf=.551+.4485*(m-1)^.505*exp(-5.68*(m-1));elseif m<4,mf=m/(.356*m*m+2.237*m-1.4);else,mf=.302;end
cxi=1;if p.useCxiMach,cxi=2.4352500438690186*mf+.25001898407936096;end
cy=p.CyK*missile.curve(p.CyMachMult,mach);delta=u(:).*p.finsAoa;dn=norm(delta);finLimited=pressure*dn>p.Kfin;if finLimited,delta=delta*p.Kfin/(pressure*dn);end
virtual=unit(direction+R(:,3)*delta(1)+R(:,2)*delta(2));main=force(direction);tail=force(virtual);tb=R'*tail;
aeroMoment=[0;-p.lever*tb(3);p.lever*tb(2)];
% Source damping operator and braking limiter; nominal inertia stays fixed.
scale=pressure*p.S*p.length^2;raw=([-0.01;-0.05;-0.05].*w)*scale.*p.WdK;inertia=p.inertia0;
braking=-(inertia.*w/P.Ts+aeroMoment);damping=raw;ii=abs(raw)>abs(braking);damping(ii)=braking(ii);moment=aeroMoment+damping;
gyro=[(inertia(3)-inertia(2))*w(2)*w(3);(inertia(1)-inertia(3))*w(3)*w(1);(inertia(2)-inertia(1))*w(1)*w(2)];alpha=(moment+gyro)./inertia;
external=R(:,1)*thrust;total=main+external;accel=total/mass-[0;p.gravity;0];if norm(accel)>p.maxAcceleration,accel=accel*p.maxAcceleration/norm(accel);end
B=missile.referenceBasis(X);local=R'*unit(rel);aRef=B'*accel;
Y=struct('force',total,'moment',moment,'aerodynamicForce',main,'aerodynamicMoment',aeroMoment,'damping',damping,'externalForce',external,'angularAcceleration',alpha,'accelerationWorld',accel,'feedbackWorld',accel,'lateralAccel',aRef([3 2]),'alpha',atan2(local(2),local(1)),'beta',atan2(local(3),hypot(local(1),local(2))),'angularRate',w,'pressure',pressure,'mach',mach,'mass',mass,'thrust',thrust,'delta',delta,'finLimited',finLimited,'controlConsumed',u(:));
    function F=force(d)
    cosine=dot(d,forward);sine2=max(0,1-cosine^2);cd=p.CxK*mf+p.CxK*p.CxAoA*sine2*cxi;cl=cy*sqrt(sine2);if cl>p.CyMaxAoA,cl=max(0,2*p.CyMaxAoA-cl);end;if cosine<0,cl=-cl;end
    % Stable equivalent projection avoids subtractive cancellation at zero AoA.
    lift=unit(cross(d,cross(forward,d)));F=-pressure*p.S*cd*d+pressure*p.Sw*cl*lift;
    end
end
function y=unit(x),if norm(x)>3.99999987306209e-19,y=x/norm(x);else,y=zeros(size(x));end,end
