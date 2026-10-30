function X=initState(P,c)
% condition: height [m], speed [m/s] or mach, pitch/yaw [rad], time [s].
if nargin<2,c=struct;end
h=get(c,'height',5000);[~,sound]=missile.atmosphere(h);v=get(c,'speed',get(c,'mach',2)*sound);
pitch=get(c,'pitch',0);yaw=get(c,'yaw',0);
% Positive pitch raises nose; positive yaw points toward world +Z.
q=[-sin(yaw/2)*sin(pitch/2);-sin(yaw/2)*cos(pitch/2);cos(yaw/2)*sin(pitch/2);cos(yaw/2)*cos(pitch/2)];
X=struct('position',[0;h;0],'velocity',[v*cos(pitch)*cos(yaw);v*sin(pitch);v*cos(pitch)*sin(yaw)],'quaternion',q,'omega',zeros(3,1),'time',get(c,'time',0));
for k={'position','velocity','quaternion','omega'},if isfield(c,k{1}),X.(k{1})=c.(k{1})(:);end,end
end
function v=get(s,k,d),if isfield(s,k),v=s.(k);else,v=d;end,end
