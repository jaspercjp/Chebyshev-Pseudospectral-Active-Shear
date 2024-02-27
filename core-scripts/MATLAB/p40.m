% p40.m - eigenvalues of Orr-Sommerfeld operator (compare p38.m)
% from Trefethen's book
% modified to allow k to vary

k=1.0; R = 5772; clf, [ay,ax] = meshgrid([.56 .04],[.1 .5]);
for N = 40:20:100
    % 2nd- and 4th-order differentiation matrices:
    [ee,x] = cheb(N); D2 = ee^2; D2 = D2(2:N,2:N);
    S = diag([0; 1 ./(1-x(2:N).^2); 0]);
    D4 = (diag(1-x.^2)*ee^4 - 8*diag(x)*ee^3 - 12*ee^2)*S;
    D4 = D4(2:N,2:N);
    
    % Orr-Sommerfeld operators A,B and generalized eigenvalues:
    I = eye(N-1);
    A = (D4-2*k^2*D2+k^4*I)/R - 2i*k*I - 1i*k*diag(1-x(2:N).^2)*(D2-I);
    B = D2-k^2*I;
    [V,ee] = eig(A,B);
    i = N/20-1; subplot('position',[ax(i) ay(i) .38 .38])
    plot(ee,'.','markersize',12)
    grid on, axis([-.8 .2 -1 0])
    axis square
    title(['N = ' int2str(N) '    \lambda_{max} = ' ...
        num2str(max(real(ee)),'%16.12f')]), drawnow
end