function result=core_tests
%CORE_TESTS Cross-language fixtures from pinned Python reference.
root=fileparts(fileparts(mfilename('fullpath')));addpath(root);
r=jsondecode(fileread(fullfile(root,'data','core_reference.json')));result=struct;
for j=1:numel(r)
p=missile.loadParameters(r(j).id);s=r(j).state;X=struct('position',s.p(:),'velocity',s.v(:),'quaternion',s.q(:),'omega',s.omega(:),'time',s.time);u=r(j).control(:);
y=missile.plantOutputs(X,u,p);af=r(j).static.force(:);am=r(j).static.moment(:);
assert(norm(y.aerodynamicForce-af)<1e-5*max(1,norm(af)));assert(norm(y.aerodynamicMoment-am)<1e-5*max(1,norm(am)));
maxP=0;maxV=0;maxQ=0;
for k=1:numel(r(j).trajectory)
[X,y]=missile.plantStep(X,u,p);e=r(j).trajectory(k);maxP=max(maxP,norm(X.position-e.p(:)));maxV=max(maxV,norm(X.velocity-e.v(:)));maxQ=max(maxQ,min(norm(X.quaternion-e.q(:)),norm(X.quaternion+e.q(:))));
end
assert(maxP<.25,'Position parity exceeded 0.25 m');assert(maxV<.25,'Velocity parity exceeded .25 m/s');assert(maxQ<.002,'Attitude parity exceeded .002 quaternion norm');
C=missile.initAutopilot(p);[C,u,d]=missile.autopilotStep(C,[5;-3],y,X,p);assert(all(isfinite(u)));[held,off]=missile.autopilotStep(C,[5;-3],y,X,p,struct('allowed',false));assert(isequal(held,C)&&all(off==0));
result.(r(j).id)=struct('positionError_m',maxP,'velocityError_mps',maxV,'quaternionError',maxQ);disp(result.(r(j).id));
end
end
