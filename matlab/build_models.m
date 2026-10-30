function paths=build_models(selected)
if nargin==0,selected=1:2;end
root=fileparts(mfilename('fullpath'));addpath(root,fullfile(root,'models'));
paths=cell(2,1);names={'missile_plant_test','missile_autopilot_test'};
for k=selected
 fprintf('Building model %d/2\n',k);name=names{k};if bdIsLoaded(name),close_system(name,0);end
 new_system(name);o=missile.analysis.defaultOptions(struct('closedLoop',k==2));
 set_param(name,'SolverType','Fixed-step','Solver','FixedStepDiscrete','FixedStep','1/48','StopTime','4','ReturnWorkspaceOutputs','on');
 mw=get_param(name,'ModelWorkspace');assignin(mw,'benchOptions',o);
 init=sprintf('addpath(fileparts(fileparts(get_param(bdroot,''FileName''))));addpath(fileparts(get_param(bdroot,''FileName'')));');set_param(name,'InitFcn',init);
 add_block('simulink/Sources/Step',[name '/Step'],'Time','round(benchOptions.startTime*48)/48','After','benchOptions.amplitude','SampleTime','1/48','Position',[40 40 90 70]);
 add_block('simulink/Sources/Pulse Generator',[name '/Pulse'],'PulseType','Sample based','Amplitude','benchOptions.amplitude','Period','100000','PulseWidth','max(1,round(benchOptions.pulseWidth*48))','PhaseDelay','round(benchOptions.startTime*48)','SampleTime','1/48','Position',[40 115 90 145]);
 add_block('simulink/Signal Routing/Manual Switch',[name '/Step or Pulse'],'sw','1','Position',[145 50 175 130]);
 add_line(name,'Step/1','Step or Pulse/1');add_line(name,'Pulse/1','Step or Pulse/2');
 add_block('simulink/Math Operations/Gain',[name '/Select axis'],'Gain','[benchOptions.axis==1;benchOptions.axis==2]','Multiplication','Matrix(K*u)','Position',[225 65 305 105]);add_line(name,'Step or Pulse/1','Select axis/1');
 add_block('simulink/User-Defined Functions/Level-2 MATLAB S-Function',[name '/Missile kernel'],'FunctionName','missile_bench_sfun','Parameters','benchOptions','Position',[370 55 510 115]);add_line(name,'Select axis/1','Missile kernel/1');
 add_block('simulink/Sinks/Scope',[name '/Responses'],'Position',[590 35 630 75]);add_line(name,'Missile kernel/1','Responses/1');
 add_block('simulink/Sinks/To Workspace',[name '/Telemetry'],'VariableName','telemetry','SaveFormat','Timeseries','Position',[575 110 670 145]);add_line(name,'Missile kernel/1','Telemetry/1');
 add_block('simulink/Sinks/To Workspace',[name '/Command'],'VariableName','command','SaveFormat','Timeseries','Position',[365 175 460 210]);add_line(name,'Select axis/1','Command/1');
 note=Simulink.Annotation(name,'48 Hz | a H/V [m/s^2], alpha/beta [rad], omega Y/Z [rad/s], u H/V [unitless]. Edit benchOptions in Model Explorer; switch selects Step/Pulse. Start and pulse width round to samples (min width 1).');note.Position=[30 260 730 310];
 fprintf('Saving %s\n',name);paths{k}=fullfile(root,'models',[name '.slx']);save_system(name,paths{k});fprintf('Saved %s\n',name);close_system(name,0);if isfile([paths{k} '.bak']),delete([paths{k} '.bak']);end;fprintf('Closed %s\n',name);
end
end
