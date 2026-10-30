function S = initialize(P,X,target,options)
% Snapshot-able full engagement state. Target is an explicit kinematic input.
if nargin<4, options=struct; end
o=missile.terminal.defaultOptions(options);
target.position=target.position(:); target.velocity=target.velocity(:);
if ~isfield(target,'acceleration'), target.acceleration=zeros(3,1); end
S=struct('X',X,'target',target,'C',missile.initAutopilot(P),...
 'u',zeros(2,1),'G',struct('loftMode',false),'O',[]);
[S.O,~]=missile.terminal.observe([],X,target,P.Ts,o);
end
