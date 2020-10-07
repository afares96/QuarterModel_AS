function xdot = eqn(x, A, B,u,d)

% [zr, zrdot] = disturbance_blocks(t);

% p = pinv(B*inv(R)*B')*d;
xdot  = A*x+B*u+d;

end


