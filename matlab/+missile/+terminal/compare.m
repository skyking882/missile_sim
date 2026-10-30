function comparison = compare(P,X,target,baselineOptions,candidateOptions)
% Nonlinear before/after comparison of explicitly selected loop/observer settings.
a=missile.terminal.run(P,X,target,baselineOptions);b=missile.terminal.run(P,X,target,candidateOptions);
comparison=struct('baseline',a,'candidate',b,'metrics',[metrics(a);metrics(b)]);
 function m=metrics(r)
  rows=r.rows([r.rows.intervalMiss]>r.options.captureRadius);
  if isempty(rows),error('missile:terminal:NoTelemetry','No pre-capture telemetry for comparison');end
  a=[rows.acceleration];c=[rows.command];u=[rows.control];
  take=max(1,size(a,2)-round(2/P.Ts)):size(a,2); error=a(:,take)-c(:,take);
  m=struct('closestApproach',r.closestApproach,'terminalTrackingRms',sqrt(mean(error(:).^2)),...
   'terminalAccelPeakToPeak',max(max(a(:,take),[],2)-min(a(:,take),[],2)),...
   'saturationTime',sum(any(abs(u)>=1-1e-7,1))*P.Ts,'event',r.event);
 end
end
