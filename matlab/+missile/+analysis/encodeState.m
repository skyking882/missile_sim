function z=encodeState(X,X0)
q=X.quaternion(:);p=X0.quaternion(:);qrel=qmul([-p(1:3);p(4)],q);if qrel(4)<0,qrel=-qrel;end
v=qrel(1:3);a=2*atan2(norm(v),qrel(4));if norm(v)>0,v=v*a/norm(v);end
z=[X.velocity(2:3)-X0.velocity(2:3);v(2:3);X.omega(2:3)-X0.omega(2:3)];
end
function q=qmul(a,b)
q=[a(4)*b(1:3)+b(4)*a(1:3)+cross(a(1:3),b(1:3));a(4)*b(4)-dot(a(1:3),b(1:3))];
end
