function result=core_pid_tests
root=fileparts(fileparts(mfilename('fullpath')));addpath(root);r=jsondecode(fileread(fullfile(root,'data','core_pid_reference.json')));result=struct;
for j=1:numel(r)
p=missile.loadParameters(r(j).id);s=r(j).state;X=struct('position',s.p(:),'velocity',s.v(:),'quaternion',s.q(:),'omega',s.omega(:),'time',s.time);C=missile.initAutopilot(p);Y=struct('feedbackWorld',r(j).feedback(:));err=0;
for k=1:numel(r(j).samples),v=r(j).samples(k);X.time=v.time;[C,u]=missile.autopilotStep(C,v.command(:),Y,X,p);err=max(err,norm(u-v.control(:)));end
assert(err<2e-6,'Direct acceleration PID mismatch');result.(r(j).id)=err;
end
disp(result);
end
