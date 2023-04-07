function PlotVelocity(oss)
    [vx, vy] = streamFuncToVelocity(oss);
    figure(); subplot(2,1,1); plot(vx);
    title("Velocity Plot", ...
        sprintf("$k=%0.3f, \\bar{a}=%0.3f, \\bar{\\tau}=%0.3f$", oss.k, oss.aBar, oss.tBar),...
        'interpreter', 'latex');
    ylabel("$v_x$", 'interpreter', 'latex', 'fontsize', 17);

    subplot(2,1,2);plot(vy); 
    xlabel('$y$', 'interpreter', 'latex', 'fontsize', 17); 
    ylabel('$v_y$', 'interpreter', 'latex', 'fontsize', 17);
end

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