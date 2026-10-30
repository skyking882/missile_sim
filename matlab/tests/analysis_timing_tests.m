function result=analysis_timing_tests
% Independent composite state map checks every held control / output boundary.
o=missile.analysis.defaultOptions(struct('amplitude',.2));m=missile.analysis.linearizeModel(o);b=m.base;b.env.analysis=true;n=16;
[A,~]=missile.analysis.jacobian(@step,zeros(n,1));[B,~]=missile.analysis.jacobian(@input,zeros(2,1));C=zeros(2,n);C(:,15:16)=eye(2);composite=ss(A,B,C,zeros(2),b.P.Ts);
t=(0:24)'*b.P.Ts;r=zeros(numel(t),2);r(1,1)=.1;r(6,2)=.1;direct=lsim(composite,r,t);factored=lsim(m.T,r,t);
err=norm(direct-factored,'fro')/norm(direct,'fro');assert(err<.01,'Factored/composite sample timing mismatch: %g',err);
% Error should decrease with input amplitude before the single-rounding floor.
errors=zeros(1,2);amplitudes=[.4 .2];
for j=1:2
 [base,actual]=deal(b);u=zeros(numel(t),2);u(1,2)=amplitudes(j);y=zeros(numel(t),2);
 for k=1:numel(t)
  y(k,:)=(actual.Y.lateralAccel-base.Y.lateralAccel)';
  actual=missile.analysis.advance(actual,u(k,:)');base=missile.analysis.advance(base,zeros(2,1));
 end
 pred=lsim(composite,u,t);errors(j)=norm(y-pred,'fro')/norm(pred,'fro');
end
assert(errors(2)<errors(1)||max(errors)<1e-3,'Small-signal convergence failed');
result=struct('compositeFactoredRelativeError',err,'amplitudes',amplitudes,'nonlinearErrors',errors,'samples',numel(t));disp(result)
 function out=step(v)
  q=b;q.X=missile.analysis.decodeState(v(1:6),b.X);q.C.integral=b.C.integral+v(7:8);q.C.previousError=b.C.previousError+v(9:10);q.C.derivative=b.C.derivative+v(11:12);q.u=b.u+v(13:14);q=missile.analysis.advance(q,zeros(2,1));out=pack(q);
 end
 function out=input(u),q=missile.analysis.advance(b,u);out=pack(q);end
 function v=pack(q),v=[missile.analysis.encodeState(q.X,b.X);q.C.integral-b.C.integral;q.C.previousError-b.C.previousError;q.C.derivative-b.C.derivative;q.u-b.u;q.Y.lateralAccel-b.Y.lateralAccel];end
end
