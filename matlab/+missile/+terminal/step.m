function [S,row] = step(S,P,options)
% Stored control -> physical advance -> observer -> PN -> PID -> next control.
o=missile.terminal.defaultOptions(options); env=o.env;
if o.analysis, env.analysis=true; end
oldX=S.X; consumed=S.u;
[S.X,Y]=missile.plantStep(S.X,consumed,P,env);
a=S.target.acceleration(:); S.target.position=S.target.position+P.Ts*S.target.velocity+0.5*P.Ts^2*a;
S.target.velocity=S.target.velocity+P.Ts*a;
[S.O,M]=missile.terminal.observe(S.O,S.X,S.target,P.Ts,o);
gopts=struct('direction',M.direction,'omega',M.omega,'valid',M.valid,'analysis',o.analysis);
if isfield(P.rocket.guidance,'useTargetVel') && P.rocket.guidance.useTargetVel
 gopts.velocityArgument=cross(M.omega,M.direction*M.range)+M.direction*M.closing;
end
[cmd,S.G,GD]=missile.guidanceStep(S.X,S.target,S.G,P,gopts);
[S.C,S.u,D]=missile.autopilotStep(S.C,cmd,Y,S.X,P,struct('allowed',M.valid && GD.allowed,'analysis',o.analysis,'gainScale',o.pidScale));
r0=S.target.position-P.Ts*S.target.velocity+0.5*P.Ts^2*a-oldX.position;
r1=S.target.position-S.X.position; dr=r1-r0;
fraction=max(0,min(1,-dot(r0,dr)/max(dot(dr,dr),eps)));
row=struct('time',S.X.time,'range',M.range,'closing',M.closing,...
 'command',cmd(:),'acceleration',Y.lateralAccel(:),'control',S.u(:),...
 'controlConsumed',consumed(:),'saturated',any(abs(S.u)>=1-1e-7),...
 'alpha',Y.alpha,'beta',Y.beta,'omega',S.X.omega(:),...
 'position',S.X.position,'velocity',S.X.velocity,'intervalMiss',norm(r0+fraction*dr),...
 'observerValid',M.valid,'observerKind',M.kind,'pid',D,'guidance',GD);
end
