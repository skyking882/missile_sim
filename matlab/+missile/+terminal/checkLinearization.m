function check = checkLinearization(L,P,options)
% Direct one-step nonlinear perturbation check against the full terminal map.
o=missile.terminal.defaultOptions(options);o.analysis=true;
z=missile.terminal.localCoordinates('encode',L.reference,L.reference);
znext=missile.terminal.localCoordinates('encode',L.referenceNext,L.referenceNext);
direction=sin((1:numel(z))');delta=0.1*L.stepSizes.*direction;
S=missile.terminal.localCoordinates('decode',z+delta,L.reference);
next=missile.terminal.step(S,P,o);actual=missile.terminal.localCoordinates('encode',next,L.referenceNext)-znext;
predicted=L.A*delta;
check=struct('relativeError',norm(actual-predicted)/max(norm(actual),1e-12),...
 'absoluteError',norm(actual-predicted),'perturbation',delta,'actual',actual,'predicted',predicted);
check.passed=check.relativeError<0.1;
end
