function missile_bench_sfun(block)
setup(block);
end
function setup(b)
b.NumDialogPrms=1;b.NumInputPorts=1;b.NumOutputPorts=1;b.SetPreCompInpPortInfoToDynamic;b.SetPreCompOutPortInfoToDynamic;
b.InputPort(1).Dimensions=2;b.InputPort(1).DirectFeedthrough=false;b.OutputPort(1).Dimensions=8;b.SampleTimes=[1/48 0];b.SimStateCompliance='DefaultSimState';
b.RegBlockMethod('Start',@start);b.RegBlockMethod('Outputs',@outputs);b.RegBlockMethod('Update',@update);
end
function start(b)
s=missile.analysis.initialize(b.DialogPrm(1).Data);set_param(b.BlockHandle,'UserData',s);
end
function outputs(b)
s=get_param(b.BlockHandle,'UserData');b.OutputPort(1).Data=missile.analysis.outputVector(s);
end
function update(b)
s=get_param(b.BlockHandle,'UserData');s=missile.analysis.advance(s,b.InputPort(1).Data);set_param(b.BlockHandle,'UserData',s);
end
