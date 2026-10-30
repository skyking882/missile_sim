function report = analyze(result,indices)
% Select terminal snapshots, retaining full-loop observer/PN and controller memory.
N=numel(result.snapshots);
eligible=find([result.rows.intervalMiss]>result.options.captureRadius);
if isempty(eligible), error('missile:terminal:NoSnapshots','No pre-capture terminal snapshots available'); end
if nargin<2 || isempty(indices)
 last=eligible(end); first=max(1,last-round(3/result.parameters.Ts));
 indices=unique(round(linspace(first,last,6)));
else
 if any(~ismember(indices,eligible)), error('missile:terminal:CaptureSnapshot','A selected one-step snapshot reaches capture; choose an earlier point'); end
end
checks=cell(numel(indices),1);models=cell(numel(indices),1); radius=zeros(numel(indices),1); frequency=nan(size(radius)); damping=frequency;
for k=1:numel(indices)
 models{k}=missile.terminal.linearize(result.snapshots{indices(k)},result.parameters,result.options);
 L=models{k}; checks{k}=missile.terminal.checkLinearization(L,result.parameters,result.options);radius(k)=max(abs(L.poles));
 if ~isempty(L.oscillatoryIndices), j=L.oscillatoryIndices(1); frequency(k)=L.frequencyHz(j); damping(k)=L.dampingRatio(j); end
end
report=struct('models',{models},'indices',indices,'time',cellfun(@(x)x.time,models),...
 'range',cellfun(@(x)x.range,models),'spectralRadius',radius,'dominantFrequencyHz',frequency,...
 'dominantDamping',damping,'closestApproach',result.closestApproach,'perturbationChecks',{checks},'excludedCaptureSnapshots',N-numel(eligible),...
 'limitation','Local time-varying snapshots; validity/mode switches and saturation require nonlinear trajectory review. Neutral geometry poles are included.');
end
