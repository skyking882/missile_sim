function o = defaultOptions(o)
% Explicit engineering observation bench; no IR/ARH/SARH sensor equivalence.
if nargin==0, o=struct; end
v=struct('duration',12,'captureRadius',12,'observerTau',0.04,...
 'directionNoise',[0;0],'rateNoise',[0;0;0],'absentIntervals',zeros(0,2),...
 'env',struct,'pidScale',1,'analysis',false);
n=fieldnames(v); for k=1:numel(n), if ~isfield(o,n{k}), o.(n{k})=v.(n{k}); end, end
if o.duration<=0 || o.captureRadius<0 || o.observerTau<0, error('missile:terminal:Options','Invalid duration, radius or observer time constant'); end
end
