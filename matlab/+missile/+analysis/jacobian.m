function [J,relativeDifference]=jacobian(fun,x,scale)
if nargin<3,scale=max(1,abs(x));end
h=1e-5*scale(:);J=diffAt(h);J2=diffAt(h*2);relativeDifference=norm(J-J2,'fro')/max(1,norm(J,'fro'));
 function D=diffAt(delta)
  y=fun(x);D=zeros(numel(y),numel(x));
  for j=1:numel(x),v=zeros(size(x));v(j)=delta(j);D(:,j)=(fun(x+v)-fun(x-v))/(2*delta(j));end
 end
end
