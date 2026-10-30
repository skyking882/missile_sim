function X=decodeState(z,X0)
X=X0;X.velocity(2:3)=X.velocity(2:3)+z(1:2);v=[0;z(3:4)];a=norm(v);if a==0,q=[0;0;0;1];else,q=[v*sin(a/2)/a;cos(a/2)];end
p=X0.quaternion(:);X.quaternion=[p(4)*q(1:3)+q(4)*p(1:3)+cross(p(1:3),q(1:3));p(4)*q(4)-dot(p(1:3),q(1:3))];
X.omega(2:3)=X.omega(2:3)+z(5:6);
end
