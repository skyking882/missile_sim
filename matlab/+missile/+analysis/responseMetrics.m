function m=responseMetrics(t,y,r,o)
m=struct('riseTime',NaN,'settlingTime',NaN,'overshootPercent',NaN,'steadyStateError',NaN,'settled',false,'status','not applicable to pulse');
if ~strcmpi(o.signal,'step'),return;end
effectiveStart=round(o.startTime/(t(2)-t(1)))*(t(2)-t(1));j=find(t>=effectiveStart,1);if isempty(j)||o.amplitude==0,m.status='no nonzero step';return;end
baseline=y(max(1,j-1));target=baseline+o.amplitude;e=(y(j:end)-baseline)/o.amplitude;tt=t(j:end)-effectiveStart;
a=find(e>=.1,1);b=find(e>=.9,1);if ~isempty(a)&&~isempty(b),m.riseTime=tt(b)-tt(a);end
m.overshootPercent=max(0,(max(e)-1)*100);m.steadyStateError=target-y(end);
last=find(abs(y(j:end)-target)>.02*abs(o.amplitude),1,'last');
if isempty(last),m.settlingTime=0;m.settled=true;elseif last<numel(tt) && tt(end)-tt(last+1)>=max(.25,5*(t(2)-t(1))),m.settlingTime=tt(last+1);m.settled=true;end
if m.settled,m.status='settled within 2 percent';else,m.status='not settled within simulated interval';end
end
