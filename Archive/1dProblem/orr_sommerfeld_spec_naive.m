R = 5772; clf, [ay,ax] = meshgrid([.56 .04],[.1 .5]);
figure(); % The eigenvalue spectrum looks about the same
i=1;
for N = 40:20:100
    %2nd- and 4th-order differentiation matrices:
    [D,x] = cheb(N); D2 = D^2; D2 = D2(2:N,2:N);
    D4 = D2^2;
%     S = diag([0; 1 ./(1-x(2:N).^2); 0]);
%     D4 = (diag(1-x.^2)*D^4 - 8*diag(x)*D^3 - 12*D^2)*S;
%     D4 = D4(2:N,2:N);
    % Orr-Sommerfeld operators A,B and generalized eigenvalues:
    I = eye(N-1);
    A = (D4-2*D2+I)/R - 2i*I - 1i*diag(1-x(2:N).^2)*(D2-I);
    A(1,:) = zeros(1,N-1); A(1,1) = 1.0;
    A(2,:) = D(1,2:N);
    A(N-1,:) = zeros(1,N-1); A(N-1,N-1) = 1.0;
    A(N-2,:) = D(N+1,2:N);
    B = D2-I;
    B(1,:) = zeros(1,N-1); B(2,:) = zeros(1,N-1);
    B(N-1,:) = zeros(1,N-1); B(N-2,:) = zeros(1,N-1);
    [V,ee] = eig(A,B);
    subplot(2,2,i); plot(x(2:N), V(:,2));
    i=i+1;
%     i = N/20-1; subplot('position',[ax(i) ay(i) .38 .38])
%     plot(ee,'.','markersize',12)
%     grid on, axis([-.8 .2 -1 0]), axis square
%     title(['N = ' int2str(N) ' \lambda_{max} = ' ...
%     num2str(max(real(ee)),'%15.11f')]), drawnow
end