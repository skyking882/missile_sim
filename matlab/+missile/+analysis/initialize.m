function b=initialize(o)
o=missile.analysis.defaultOptions(o);P=missile.loadParameters(o.missile);X=missile.initState(P,o.condition);
env=struct('wind',o.wind(:));
if ~o.freeFlight
 y=missile.plantOutputs(X,zeros(2,1),P,env);
 env.frozen=struct('height',X.position(2),'speed',norm(X.velocity),'mass',y.mass,'thrust',y.thrust,'elapsed',X.time);
end
b=struct('P',P,'X',X,'C',missile.initAutopilot(P),'u',zeros(2,1),'env',env,'options',o);
if ~o.freeFlight
 ae=env;ae.analysis=true;v=fminsearch(@residual,[0;0],optimset('Display','off','TolX',1e-10,'TolFun',1e-14,'MaxIter',400));
 b.trimResidual=sqrt(residual(v));X=missile.analysis.decodeState([0;0;0;v(1);0;0],X);b.X=X;b.u=[0;v(2)];b.C.integral=b.u;

else,b.trimResidual=NaN;
end
b.trimControl=b.u;b.Y=missile.plantOutputs(b.X,b.u,P,env);
 function cost=residual(v)
  xx=missile.analysis.decodeState([0;0;0;v(1);0;0],X);[xn,yy]=missile.plantStep(xx,[0;v(2)],P,ae);
  cost=(yy.lateralAccel(2)/9.81)^2+(xn.omega(3)/P.Ts)^2+max(0,abs(v(2))-1)^2;
 end
end
