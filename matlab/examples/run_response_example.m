function r=run_response_example(o)
if nargin==0,o=struct;end
r=missile.analysis.runResponse(o);figure('Name','Nonlinear missile response');tiledlayout(3,1);nexttile;
if r.options.closedLoop
 plot(r.time,r.input,'--',r.time,r.output(:,1:2));legend('horizontal command','vertical command','horizontal actual','vertical actual');
else
 plot(r.time,r.output(:,1:2));legend('horizontal actual','vertical actual');
end
ylabel('Acceleration (m/s²)');grid on
nexttile;plot(r.time,r.output(:,3:6));ylabel('rad, rad/s');grid on
nexttile;
if r.options.closedLoop
 plot(r.time,r.output(:,7:8));legend('horizontal control','vertical control');
else
 plot(r.time,r.input,'--',r.time,r.output(:,7:8));legend('horizontal perturbation','vertical perturbation','horizontal control','vertical control');
end
ylabel('normalized control');xlabel('Time (s)');grid on;disp(r.metrics)
end
