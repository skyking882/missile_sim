function L = linearize(S,P,options)
% Frozen-time tangent of the complete sampled terminal loop, along a trajectory.
% Target acceleration, clock, observer validity and Loft mode are conditioned.
o=missile.terminal.defaultOptions(options); o.analysis=true;
z=missile.terminal.localCoordinates('encode',S,S);
[baseline,~]=missile.terminal.step(S,P,o);
znext=missile.terminal.localCoordinates('encode',baseline,baseline);
scale=[max(abs(z(1:3)),100);1000;max(abs(z(5:7)),100);ones(3,1);ones(3,1);...
 ones(6,1);ones(2,1);ones(5,1)];
h=2e-5*scale; A=jacobian(h); A2=jacobian(h/2);
poles=eig(A2); continuous=log(complex(poles))/P.Ts;
frequency=abs(imag(continuous))/(2*pi); damping=-real(continuous)./max(abs(continuous),eps);
osc=find(frequency>0.05 & imag(poles)>1e-8 & abs(poles)>1e-5);
[~,order]=sort(abs(poles(osc)),'descend'); osc=osc(order);
L=struct('A',A2,'poles',poles,'frequencyHz',frequency,'dampingRatio',damping,...
 'oscillatoryIndices',osc,'differenceRelative',norm(A2-A,'fro')/max(1,norm(A2,'fro')),...
 'reference',S,'referenceNext',baseline,'stepSizes',h/2,...
 'localStable',all(abs(poles)<1-1e-6),'oscillatoryStable',all(abs(poles(osc))<1),'range',norm(S.target.position-S.X.position),...
 'time',S.X.time,'Ts',P.Ts,'controlSaturated',any(abs(S.u)>=1-1e-7),'label','frozen-time full PN/observer/PID/plant terminal tangent',...
 'conditioning','clock, target acceleration and discrete validity/Loft branches conditioned; neutral kinematic modes retained');
 function J=jacobian(steps)
  J=zeros(numel(z));
  for j=1:numel(z)
   zp=z;zm=z;zp(j)=zp(j)+steps(j);zm(j)=zm(j)-steps(j);
   Sp=missile.terminal.localCoordinates('decode',zp,S);Sm=missile.terminal.localCoordinates('decode',zm,S);
   Sp=missile.terminal.step(Sp,P,o);Sm=missile.terminal.step(Sm,P,o);
   fp=missile.terminal.localCoordinates('encode',Sp,baseline);fm=missile.terminal.localCoordinates('encode',Sm,baseline);
   J(:,j)=(fp-fm)/(2*steps(j));
  end
 end
end
