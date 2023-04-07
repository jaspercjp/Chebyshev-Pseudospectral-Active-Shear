function [ev, ef] = solve1DOS(R)
    N = 60;
    % 2nd- and 4th-order differentiation matrices:
    [D,x] = cheb(N, 1); D2 = D^2; D2 = D2(2:N,2:N);
    S = diag([0; 1 ./(1-x(2:N).^2); 0]);
    D4 = (diag(1-x.^2)*D^4 - 8*diag(x)*D^3 - 12*D^2)*S;
    D4 = D4(2:N,2:N);
    % Orr-Sommerfeld operators A,B and generalized eigenvalues:
    I = eye(N-1);
    A = (D4-2*D2+I)/R - 2i*I - 1i*diag(1-x(2:N).^2)*(D2-I);
    B = D2-I;
    [ev, ef] = eig(A,B)
    %     i = N/20-1; subplot('position',[ax(i) ay(i) .38 .38])
    %     plot(ee,'.','markersize',12)
    %     grid on, axis([-.8 .2 -1 0]), axis square
    %     title(['N = ' int2str(N) ' \lambda_{max} = ' ...
    %     num2str(max(real(ee)),'%16.12f')]), drawnow
end