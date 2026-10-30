function m=linearizeModel(o)
% Frozen horizontal/vertical lateral dynamics, sampled at the actual 48 Hz.
if nargin==0,o=struct;end
o=missile.analysis.defaultOptions(o);assert(~o.freeFlight,'Linearization requires a frozen operating condition.');
b=missile.analysis.initialize(o);b.env.analysis=true;X0=b.X;Ts=b.P.Ts;
[A,eA]=missile.analysis.jacobian(@plantMap,zeros(6,1));[B,eB]=missile.analysis.jacobian(@inputMap,zeros(2,1));
[C,eC]=missile.analysis.jacobian(@plantOutput,zeros(6,1));[D,eD]=missile.analysis.jacobian(@directOutput,zeros(2,1));
G=ss(A,B,C,D,Ts);G.InputName={'u_horizontal','u_vertical'};G.OutputName={'a_horizontal','a_vertical'};
% PID map is linearized from the implemented memory update and gain schedule.
[K,eK]=missile.analysis.jacobian(@pidMap,zeros(8,1));
controller=ss(K(1:6,1:6),K(1:6,7:8),K(7:8,1:6),K(7:8,7:8),Ts);
% Controller sees completed-interval velocity-difference feedback. Its plant
% transfer uses interval-average feedback, with the command hold included.
[Af,ea]=missile.analysis.jacobian(@feedbackMap,zeros(8,1));
Gfeedback=ss(Af(1:6,1:6),Af(1:6,7:8),Af(7:8,1:6),Af(7:8,7:8),Ts);
% The completed-step measurement reaches the controller at the same boundary.
% Lift output timing from y[k+1] to y[k], then delay the newly computed input.
z=tf('z',Ts); L=minreal(Gfeedback*controller/z,1e-7); T=minreal(feedback(L,eye(2))/z,1e-7);
scalarLoop=minreal(L(o.axis,o.axis),1e-7);
m=struct('G',G,'Gfeedback',Gfeedback,'controller',controller,'L',L,'T',T,'scalarLoop',scalarLoop,'condition',o.condition,'options',o,'base',b,'finiteDifferenceCheck',max([eA eB eC eD eK ea]),'stateNames',{{'v_y','v_z','rotation_y','rotation_z','omega_y','omega_z'}},'equilibriumResidual',plantMap(zeros(6,1)));
 function xp=plantMap(x)
  X=missile.analysis.decodeState(x,X0);[X,~]=missile.plantStep(X,b.u,b.P,b.env);xp=missile.analysis.encodeState(X,X0);
 end
 function xp=inputMap(u)
  [X,~]=missile.plantStep(X0,u+b.u,b.P,b.env);xp=missile.analysis.encodeState(X,X0);
 end
 function y=plantOutput(x)
  X=missile.analysis.decodeState(x,X0);Y=missile.plantOutputs(X,b.u,b.P,b.env);y=Y.lateralAccel;
 end
 function y=directOutput(u)
  Y=missile.plantOutputs(X0,u+b.u,b.P,b.env);y=Y.lateralAccel;
 end
 function y=feedbackMap(v)
  X=missile.analysis.decodeState(v(1:6),X0);[X,Y]=missile.plantStep(X,v(7:8)+b.u,b.P,b.env);y=[missile.analysis.encodeState(X,X0);Y.lateralAccel];
 end
 function y=pidMap(v)
  c=b.C;c.integral=v(1:2);c.previousError=v(3:4);c.derivative=v(5:6);
  Y=b.Y;Y.lateralAccel=zeros(2,1);Y.feedbackWorld=zeros(3,1);
  [c,u]=missile.autopilotStep(c,v(7:8),Y,X0,b.P,struct);
  y=[c.integral;c.previousError;c.derivative;u];
 end
end
