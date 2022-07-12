% ee = arrayfun(@(R) find_ee(R), 1:6000, 'uniformoutput',false);
ee_c = find_ee(5722);

function ee = find_ee(R)
    k=1.02; clf, N = 40;
  
    % N is the amount of Chebyshev interpolation points the program uses
    % 2nd- and 4th-order differentiation matrices:
    [D,x] = cheb(N); D2 = D^2; D2 = D2(2:N,2:N);
    S = diag([0; 1 ./(1-x(2:N).^2); 0]);
    D4 = (diag(1-x.^2)*D^4 - 8*diag(x)*D^3 - 12*D^2)*S;
    D4 = D4(2:N,2:N); %Obtaining the Chebyshev differentiation matrices

    % Orr-Sommerfeld operators A,B and generalized eigenvalues:
    I = eye(N-1);  % Writing down the operators involved in the eigenvalue problem
    A = (D4-2*k^2*D2+k^4*I)/R - 2i*k*I - 1i*k*diag(1-x(2:N).^2)*(D2-I);
    B = D2-k^2*I;
    ee = eig(A,B); % Solve the eigenvalue problem
end
