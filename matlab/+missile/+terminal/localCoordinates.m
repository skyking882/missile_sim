function out = localCoordinates(action,value,reference)
% 26 independent local coordinates: relative r, height, v, rotation, omega,
% PID memories, held control and five observer coordinates. No quaternion pole.
if strcmp(action,'encode')
 S=value; qrel=qmul(qconj(reference.X.quaternion),S.X.quaternion);
 if qrel(4)<0, qrel=-qrel; end
 nv=norm(qrel(1:3)); if nv<1e-12, rot=2*qrel(1:3); else, rot=2*atan2(nv,qrel(4))*qrel(1:3)/nv; end
 out=[S.target.position-S.X.position;S.X.position(2);S.X.velocity;rot;S.X.omega;...
 S.C.integral;S.C.previousError;S.C.derivative;S.u;S.O.angles;S.O.omega];
elseif strcmp(action,'decode')
 z=value; out=reference;
 out.X.position(2)=z(4); out.X.velocity=z(5:7); rot=z(8:10); a=norm(rot);
 if a<1e-12, dq=[rot/2;1]; else, dq=[rot*sin(a/2)/a;cos(a/2)]; end
 out.X.quaternion=qmul(reference.X.quaternion,dq); out.X.quaternion=out.X.quaternion/norm(out.X.quaternion);
 out.X.omega=z(11:13); out.C.integral=z(14:15); out.C.previousError=z(16:17); out.C.derivative=z(18:19);
 out.u=z(20:21); out.O.angles=z(22:23); out.O.omega=z(24:26);
 out.target.position=out.X.position+z(1:3);
else, error('missile:terminal:Coordinates','Unknown coordinate operation'); end
end
function q=qconj(q), q=[-q(1:3);q(4)]; end
function q=qmul(a,b), q=[a(4)*b(1:3)+b(4)*a(1:3)+cross(a(1:3),b(1:3));a(4)*b(4)-dot(a(1:3),b(1:3))]; end
