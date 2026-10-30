function [X,Y]=plantStep(X,u,P,env)
% One held-control interval, source constant-acceleration integration.
if nargin<4,env=struct;end
analysis=isfield(env,'analysis') && env.analysis;dt=P.Ts;old=X;endX=X;endX.time=X.time+dt;Y=missile.plantOutputs(endX,u,P,env);
a=Y.accelerationWorld;al=Y.angularAcceleration;X.position=cast(old.position+old.velocity*dt+a*(.5*dt^2));X.velocity=cast(max(-2000,min(2000,old.velocity+a*dt)));
theta=old.omega*dt+al*(.5*dt^2);s=sin(-theta/2);c=cos(-theta/2);
d=[s(1)*c(2)*c(3)+c(1)*s(2)*s(3);c(1)*s(2)*c(3)+s(1)*c(2)*s(3);c(1)*c(2)*s(3)-s(1)*s(2)*c(3);prod(c)-prod(s)];
q=old.quaternion;xyz=q(4)*d(1:3)+d(4)*q(1:3)+cross(q(1:3),d(1:3));qn=[xyz;q(4)*d(4)-dot(q(1:3),d(1:3))];X.quaternion=cast(qn/norm(qn));X.omega=cast(max(-15,min(15,old.omega+al*dt)));
if norm(X.velocity)>P.physical.endSpeed,X.velocity=cast(X.velocity*P.physical.endSpeed/norm(X.velocity));end
if isfield(env,'frozen')
f=env.frozen;if isfield(f,'height'),X.position(2)=f.height;end;if isfield(f,'speed'),X.velocity=X.velocity*f.speed/norm(X.velocity);end
end
X.time=old.time+dt;Y.feedbackWorld=cast((X.velocity-old.velocity)/dt);B=missile.referenceBasis(X);fb=B'*Y.feedbackWorld;Y.lateralAccel=fb([3 2]);Y.angularRate=X.omega;
    function a=cast(a),if ~analysis,a=double(single(a));end,end
end
