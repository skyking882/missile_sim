function result=core_guidance_tests
root=fileparts(fileparts(mfilename('fullpath')));addpath(root);r=jsondecode(fileread(fullfile(root,'data','core_guidance_reference.json')));result=struct;
for j=1:numel(r)
p=missile.loadParameters(r(j).id);s=r(j).state;X=struct('position',s.p(:),'velocity',s.v(:),'quaternion',s.q(:),'omega',s.omega(:),'time',s.time);
[cmd,g,d]=missile.guidanceStep(X,r(j).target,[],p);err=norm(cmd-r(j).command(:));assert(d.allowed&&err<1e-5);[cmd,~,d]=missile.guidanceStep(X,r(j).target,g,p,struct('valid',false));assert(all(cmd==0)&&~d.allowed);result.(r(j).id)=err;
end
disp(result)
end
