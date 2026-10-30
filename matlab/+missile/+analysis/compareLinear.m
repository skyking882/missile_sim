function result=compareLinear(o,makePlot)
% Compare matched physical input signals; no unit-area nonlinear overdrive.
if nargin<1,o=struct;end
if nargin<2,makePlot=true;end
o=missile.analysis.defaultOptions(o);m=missile.analysis.linearizeModel(o);nonlinear=missile.analysis.runResponse(o);
if o.closedLoop,sys=m.T;else,sys=m.Gfeedback/tf('z',m.Gfeedback.Ts);end
linear=lsim(sys,nonlinear.input,nonlinear.time);baseline=nonlinear.output(1,1:2);actual=nonlinear.output(:,1:2)-baseline;
result=struct('time',nonlinear.time,'nonlinear',actual,'linear',linear,'relativeError',norm(actual-linear,'fro')/max(1e-9,norm(linear,'fro')),'model',m,'response',nonlinear);
if makePlot,figure('Name','Matched input: nonlinear and local linear response');plot(result.time,actual,result.time,linear,'--');xlabel('Time (s)');ylabel('Acceleration perturbation (m/s²)');legend('horizontal nonlinear','vertical nonlinear','horizontal linear','vertical linear');grid on;end
end
