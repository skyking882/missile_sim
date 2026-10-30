function startup
% Add the portable missile analysis package and runnable examples.
root=fileparts(mfilename('fullpath')); addpath(root,fullfile(root,'models'),fullfile(root,'examples'),fullfile(root,'tests'));
end
