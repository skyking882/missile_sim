function y=outputVector(b)
% Channels: acceleration H/V [m/s^2], alpha/beta [rad], omega Y/Z [rad/s],
% normalized control H/V [unitless]. Horizontal precedes vertical.
y=[b.Y.lateralAccel(:);b.Y.alpha;b.Y.beta;b.X.omega(2:3);b.u(:)];
end
