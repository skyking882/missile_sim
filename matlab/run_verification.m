function results=run_verification
% Focused end-to-end acceptance checks for this MATLAB/Simulink package.
root=fileparts(mfilename('fullpath'));run(fullfile(root,'startup.m'));
results.core=core_tests;
results.pid=core_pid_tests;
results.guidance=core_guidance_tests;
results.analysis=analysis_tests;
results.timing=analysis_timing_tests;
results.simulink=simulink_tests;
results.terminal=terminal_tests;
disp('MATLAB / Simulink package verification passed.');
end
