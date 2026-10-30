function figures = plotResults(result,report)
% Engineering terminal telemetry and conditioned modal history.
r=result.rows; r=r([r.intervalMiss]>result.options.captureRadius);
if isempty(r),error('missile:terminal:NoTelemetry','No pre-capture telemetry available');end
t=[r.time]; a=[r.acceleration]; c=[r.command];u=[r.control];w=[r.omega];
figures(1)=figure('Name','Terminal nonlinear response');
tiledlayout(3,2);
nexttile;plot(t,[a(1,:);c(1,:)]');ylabel('Horizontal acceleration (m/s^2)');legend('actual','command');grid on
nexttile;plot(t,[a(2,:);c(2,:)]');ylabel('Vertical acceleration (m/s^2)');grid on
nexttile;plot(t,u');ylabel('Normalized control');hold on;plot(t,[r.saturated],'k:');grid on
nexttile;plot(t,w');ylabel('Angular rates (rad/s)');grid on
nexttile;plot(t,rad2deg([[r.alpha];[r.beta]])');ylabel('Alpha / beta (deg)');grid on
nexttile;plot(t,[r.range]);ylabel('Range (m)');xlabel('Time (s)');grid on
figures(2)=figure('Name','Frozen terminal modal analysis');tiledlayout(2,2);
nexttile;hold on;theta=linspace(0,2*pi,200);plot(cos(theta),sin(theta),'k:');
for k=1:numel(report.models),p=report.models{k}.poles;plot(real(p),imag(p),'x');end
axis equal;grid on;xlabel('Real z');ylabel('Imaginary z');
nexttile;plot(report.range,report.dominantDamping,'o-');set(gca,'XDir','reverse');ylabel('Oscillatory damping ratio');grid on
nexttile;plot(report.range,report.dominantFrequencyHz,'o-');set(gca,'XDir','reverse');ylabel('Oscillatory frequency (Hz)');xlabel('Range (m)');grid on
nexttile;plot(report.range,report.spectralRadius,'o-');set(gca,'XDir','reverse');yline(1,'k:');ylabel('Full tangent spectral radius');xlabel('Range (m)');grid on
end
