function plotAnalysis(m)
figure('Name','Plant Bode: normalized control to signed acceleration');bode(m.G);grid on
figure('Name','Selected scalar loop and margins');margin(m.scalarLoop);grid on
figure('Name','Root locus: overall selected loop gain K');rlocus(m.scalarLoop);grid on
figure('Name','Closed-loop linear step');step(m.T);grid on
figure('Name','Closed-loop unit-area impulse');impulse(m.T);grid on
end
