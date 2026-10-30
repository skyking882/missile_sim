function [C,u,D]=autopilotStep(C,aCmd,Y,X,P,options)
% Source parallel PID with velocity-frame acceleration feedback.
if nargin<6,options=struct;end
if isfield(options,'allowed') && ~options.allowed,u=zeros(2,1);D=struct('called',false);return,end
p=P.autopilot;t=X.time;if isfield(options,'elapsed'),t=options.elapsed;end
names=fieldnames(p);rows=[];for j=1:numel(names),if ~isempty(regexp(names{j},'^pid[0-9]+$','once')),v=p.(names{j});rows(end+1,:)=[v.time,v.prop,v.intg,v.diff,v.intgLim];end,end
if ~isempty(rows),rows=sortrows(rows);p.accelControlProp=missile.curve(rows(:,[1 2]),t);p.accelControlIntg=missile.curve(rows(:,[1 3]),t);p.accelControlDiff=missile.curve(rows(:,[1 4]),t);p.accelControlIntgLim=missile.curve(rows(:,[1 5]),t);end
cap=p.reqAccelMax*9.8100004196167;cmd=aCmd(:);if norm(cmd)>cap,cmd=cmd*cap/norm(cmd);end
B=missile.referenceBasis(X);fb=B'*Y.feedbackWorld;error=cmd-fb([3 2]);[rho,~]=missile.atmosphere(X.position(2));speed=norm(X.velocity);base=p.baseIndSpeed*.2777777910232544;scale=base^2*1.225000023841858^2/(rho^2*speed^2);
gains=[p.accelControlProp*sqrt(scale),p.accelControlIntg*sqrt(scale),p.accelControlDiff*scale];if isfield(options,'gainScale'),gains=gains*options.gainScale;end
C.integral=max(-p.accelControlIntgLim,min(p.accelControlIntgLim,C.integral+error*P.Ts*gains(2)));C.derivative=C.derivative+48*(error-C.previousError-P.Ts*C.derivative);C.previousError=error;
proportional=gains(1)*error;derivative=gains(3)*C.derivative;raw=proportional+C.integral+derivative;u=max(-1,min(1,raw));
D=struct('called',true,'P',proportional,'I',C.integral,'D',derivative,'error',error,'gains',gains,'raw',raw,'saturated',abs(raw)>=1,'command',cmd,'feedback',fb([3 2]));
end
