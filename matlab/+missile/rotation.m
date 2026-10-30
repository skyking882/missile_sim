function R=rotation(q)
%ROTATION Body-to-world matrix, quaternion [x y z w]. World Y is up.
q=q(:); x=q(1);y=q(2);z=q(3);w=q(4);d=2*w*w-1;
R=[2*x*x+d,2*(x*y-z*w),2*(x*z+y*w);2*(x*y+z*w),2*y*y+d,2*(y*z-x*w);2*(x*z-y*w),2*(y*z+x*w),2*z*z+d];
end
