function result=runResponse(o)
if nargin==0,o=struct;end
b=missile.analysis.initialize(o);o=b.options;t=(0:floor(o.duration/b.P.Ts))'*b.P.Ts;
y=zeros(numel(t),8);u=zeros(numel(t),2);
for k=1:numel(t)
 u(k,:)=missile.analysis.inputSignal(t(k),o)';y(k,:)=missile.analysis.outputVector(b)';
 if k<numel(t),b=missile.analysis.advance(b,u(k,:)');end
end
result=struct('time',t,'output',y,'input',u,'options',o,'finalState',b,'inputArea',sum(u(1:end-1,:),1)*b.P.Ts);
if o.closedLoop,result.metrics=missile.analysis.responseMetrics(t,y(:,o.axis),u(:,o.axis),o);else,result.metrics=struct('status','open-loop: tracking metrics not applicable');end
result.metrics.peakAcceleration=max(abs(y(:,1:2)),[],1);result.metrics.controlSaturationFraction=mean(abs(y(:,7:8))>=1,1);
end
