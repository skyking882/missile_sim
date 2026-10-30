function [aCmd,G,D]=guidanceStep(X,target,G,P,options)
% Source PN/angle pursuit + optional loft, ideal or externally observed LOS.
if nargin<5,options=struct;end
p=P.autopilot;if ~isfield(p,'timeOut'),p.timeOut=0;end;if isempty(G),G=struct('loftMode',false);end;if ~isfield(G,'loftMode'),G.loftMode=false;end
r=target.position(:)-X.position;vr=target.velocity(:)-X.velocity;range=norm(r);direction=r/max(range,eps);omega=cross(vr,r)/max(range^2,eps);closing=-dot(vr,direction);
if isfield(options,'direction'),direction=options.direction(:);end;if isfield(options,'omega'),omega=options.omega(:);end
if isfield(options,'range'),range=options.range;end;if isfield(options,'closing'),closing=options.closing;end
tgo=0;if abs(closing)>1e-12,tgo=range/closing;end
gain=curvePrefix(p,'timeToGain',X.time)*curvePrefix(p,'timeToHitToGain',tgo);
valid=true;if isfield(options,'valid'),valid=options.valid;end
allowed=valid && X.time>=p.timeOut && gain>=.001;
if ~allowed,aCmd=zeros(2,1);D=struct('allowed',false,'gain',gain,'range',range,'closing',closing,'tgo',tgo);return,end
R=missile.rotation(X.quaternion);local=R'*direction;pitch=atan2(local(2),hypot(local(1),local(3)));yaw=atan2(-local(3),local(1));A=p.angleToAccelMult;N=p.omegaToAccelMult;if p.purePursuit,A=N;N=0;end
varg=X.velocity;useTarget=isfield(P.rocket.guidance,'useTargetVel') && P.rocket.guidance.useTargetVel;rangeValid=~isfield(options,'rangeValid') || options.rangeValid;dopplerValid=~isfield(options,'dopplerValid') || options.dopplerValid;
if useTarget && rangeValid && dopplerValid,varg=cross(omega,direction*range)+direction*closing;end
if isfield(options,'velocityArgument'),varg=options.velocityArgument(:);end
warg=gain*[N*omega(1);N*omega(2)-A*yaw;N*omega(3)-A*(pitch+p.elevationAdd)];request=cross(varg,warg);
if G.loftMode
 gate=tan(deg2rad(p.loftTargetElevation));gate=gate*abs(gate);
 if direction(2)*abs(direction(2))<(direction(1)^2+direction(3)^2)*gate || norm(omega)>deg2rad(p.loftTargetOmegaMax),G.loftMode=false;
 else,bodyPitch=atan2(R(2,1),hypot(R(1,1),R(3,1)));request(2)=max(request(2),p.loftAngleToAccelMult*(deg2rad(p.loftElevation)-bodyPitch));end
end
B=missile.referenceBasis(X);ref=B'*request;aCmd=ref([3 2]);D=struct('allowed',true,'gain',gain,'range',range,'closing',closing,'tgo',tgo,'requestWorld',request,'direction',direction,'omega',omega);
end
function v=curvePrefix(s,prefix,x)
names=fieldnames(s);pts=[];for j=1:numel(names),if ~isempty(regexp(names{j},['^' prefix '[0-9]+$'],'once')),pts(end+1,:)=s.(names{j})(:).';end,end
if isempty(pts),v=1;else,v=missile.curve(pts,x);end
end
