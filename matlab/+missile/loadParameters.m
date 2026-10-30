function P=loadParameters(id)
%LOADPARAMETERS Load pinned 1.2.0rc3 normalized resources (or exported JSON path).
if nargin==0, id='su_r_77_1'; end
file=char(id);
if ~isfile(file), file=fullfile(fileparts(fileparts(mfilename('fullpath'))),'data',[file '.json']); end
P=jsondecode(fileread(file)); P.Ts=1/48;
P.physical.WdK=P.physical.WdK(:); P.physical.inertia0=P.physical.inertia0(:); P.physical.finsAoa=P.physical.finsAoa(:);
end
