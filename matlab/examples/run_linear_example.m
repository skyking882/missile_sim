function m=run_linear_example(o)
if nargin==0,o=struct;end
m=missile.analysis.linearizeModel(o);missile.analysis.plotAnalysis(m);
disp(struct('poles',pole(m.T),'finiteDifferenceCheck',m.finiteDifferenceCheck,'equilibriumResidual',norm(m.equilibriumResidual)));
end
