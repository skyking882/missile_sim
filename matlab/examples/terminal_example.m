function output = terminal_example(makePlots)
% Complete nonlinear PN/PID engagement and terminal local analysis.
% Observer is explicitly geometric with a 40 ms first-order filter.
if nargin<1,makePlots=true;end
root=fileparts(fileparts(mfilename('fullpath')));addpath(root);
P=missile.loadParameters('su_r_77_1');
X=missile.initState(P,struct('height',5000,'speed',700,'time',2));
target=struct('position',[5000;5000;300],'velocity',[-200;0;0]);
options=struct('duration',8,'observerTau',0.04);
r=missile.terminal.run(P,X,target,options);report=missile.terminal.analyze(r);
candidate=options;candidate.pidScale=0.8;
comparison=missile.terminal.compare(P,X,target,options,candidate);
if makePlots
 missile.terminal.plotResults(r,report);
 figure('Name','Terminal controller comparison');
 a=comparison.baseline.rows;b=comparison.candidate.rows;
 tiledlayout(2,1);nexttile;plot([a.time],[a.range],[b.time],[b.range]);ylabel('Range (m)');legend('baseline','PID gains x0.8');grid on
 nexttile;aa=[a.acceleration];bb=[b.acceleration];plot([a.time],aa(1,:),[b.time],bb(1,:));ylabel('Horizontal acceleration (m/s^2)');xlabel('Time (s)');grid on
end
output=struct('engagement',r,'analysis',report,'comparison',comparison);
disp(comparison.metrics);
end
