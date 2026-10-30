function B=referenceBasis(X)
% Columns are forward, vertical, horizontal velocity-reference directions.
f=X.velocity(:)/norm(X.velocity);R=missile.rotation(X.quaternion);z=cross(f,R(:,2));
if norm(z)<1e-12,error('missile:DegenerateFrame','Velocity and body up are parallel.');end
z=z/norm(z); B=[f,cross(z,f),z];
end
