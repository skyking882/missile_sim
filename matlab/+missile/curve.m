function v=curve(points,x)
if isempty(points),error('missile:Curve','Explicit loaded curve required.');end
points=sortrows(points);if size(points,1)==1,v=points(1,2);return,end
v=interp1(points(:,1),points(:,2),max(points(1,1),min(points(end,1),x)),'linear');
end
