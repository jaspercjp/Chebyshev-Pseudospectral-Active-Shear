[vx, vy] = streamFuncToVelocity(sol(5,20));
figure();plot(vx);title("vx");figure();plot(vy);title("vy");

% x and y components of velocity at x=0 and t=0.
function [vx, vy] = streamFuncToVelocity(oss)
    assert(isa(oss, 'OSSolution'), 'Input is not an OSSolution');
    Psi = oss.Psi;
    k = oss.k;
    vx = real(diff(Psi));
    vy = -real(i*k*Psi);
end

% vx and vy should be chebfuns
function ret = reynoldsStress(vx, vy)
    % Compute the Reynolds stress by averaging the product vx*vy over y
    ret = -vx * vy;
end