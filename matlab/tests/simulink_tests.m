function results=simulink_tests
root=fileparts(fileparts(mfilename('fullpath')));paths={fullfile(root,'models','missile_plant_test.slx');fullfile(root,'models','missile_autopilot_test.slx')};
assert(all(cellfun(@isfile,paths)),'Build both models with build_models first.');
cleanup=onCleanup(@()closeModels(paths));errors=zeros(2,4);commandErrors=errors;resetErrors=zeros(2,1);
for k=1:2
 load_system(paths{k});[~,name]=fileparts(paths{k});mw=get_param(name,'ModelWorkspace');
 for c=1:4
  o=missile.analysis.defaultOptions(struct('closedLoop',k==2,'duration',1,'axis',1+mod(c,2),'freeFlight',c>2,'startTime',.173,'pulseWidth',.047,'amplitude',.2));
  if mod(c,2),o.signal='step';sw='1';else,o.signal='pulse';sw='0';end
  assignin(mw,'benchOptions',o);set_param(name,'StopTime','1');set_param([name '/Step or Pulse'],'sw',sw);
  set_param(name,'SimulationCommand','update');s=sim(name);r=missile.analysis.runResponse(o);
  assert(size(s.telemetry.Data,1)==size(r.output,1));errors(k,c)=max(abs(s.telemetry.Data-r.output),[],'all');commandErrors(k,c)=max(abs(s.command.Data-r.input),[],'all');
  assert(errors(k,c)<1e-10,'Simulink/script telemetry mismatch: %g',errors(k,c));assert(commandErrors(k,c)<1e-12,'Signal mismatch: %g',commandErrors(k,c));
  if c==1,s2=sim(name);resetErrors(k)=max(abs(s2.telemetry.Data-s.telemetry.Data),[],'all');assert(resetErrors(k)==0,'State reset mismatch');end
 end
 close_system(name,0);
end
results=struct('cases',8,'repeatedRuns',2,'maximumTelemetryError',max(errors,[],'all'),'maximumCommandError',max(commandErrors,[],'all'),'maximumResetError',max(resetErrors),'models',{paths});disp(results)
end
function closeModels(paths)
for k=1:numel(paths),[~,name]=fileparts(paths{k});if bdIsLoaded(name),close_system(name,0);end,end
end
