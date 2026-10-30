function b=advance(b,r)
if b.options.closedLoop
 [b.X,b.Y]=missile.plantStep(b.X,b.u,b.P,b.env);
 options=struct;if isfield(b.env,'frozen'),options.elapsed=b.env.frozen.elapsed;end
 [b.C,b.u,b.controller]=missile.autopilotStep(b.C,r,b.Y,b.X,b.P,options);
else
 b.u=max(-1,min(1,b.trimControl+r));[b.X,b.Y]=missile.plantStep(b.X,b.u,b.P,b.env);
end
end
