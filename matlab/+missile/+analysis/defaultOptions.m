function o=defaultOptions(o)
if nargin==0,o=struct;end
d=struct('missile','su_r_77_1','duration',4,'axis',2,'amplitude',1,'startTime',0.5,'pulseWidth',1/48,'signal','step','closedLoop',true,'freeFlight',false,'condition',struct('height',5000,'speed',600,'time',6),'wind',[0;0;0]);
f=fieldnames(d);for i=1:numel(f),if ~isfield(o,f{i}),o.(f{i})=d.(f{i});end,end
assert(ismember(o.axis,[1 2]),'axis must be 1 (horizontal) or 2 (vertical)');
assert(o.duration>0 && o.startTime>=0 && o.pulseWidth>0,'Invalid signal timing');
end
