function result = run(P,X,target,options)
% Nonlinear engagement with full restorable snapshots and interval miss metric.
if nargin<4, options=struct; end
o=missile.terminal.defaultOptions(options); S=missile.terminal.initialize(P,X,target,o);
N=ceil(o.duration/P.Ts); snapshots=cell(N,1); rows=cell(N,1); event='duration'; closest=inf;
for k=1:N
 snapshots{k}=S; [S,rows{k}]=missile.terminal.step(S,P,o);
 closest=min(closest,rows{k}.intervalMiss);
 if closest<=o.captureRadius, event='capture'; break; end
 if S.X.position(2)<=0, event='ground'; break; end
 if rows{k}.closing<0 && k>3 && rows{k}.range>closest+100, event='passed'; break; end
end
rows=vertcat(rows{1:k}); snapshots=snapshots(1:k);
result=struct('rows',rows,'snapshots',{snapshots},'finalState',S,'event',event,...
 'closestApproach',closest,'options',o,'parameters',P,...
 'observationModel','explicit filtered geometric observer; no IR/ARH/SARH equivalence');
end
