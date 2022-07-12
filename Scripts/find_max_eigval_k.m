% Program to find all the (k, R) pairs when the flow just starts to grow
% unstable (eigenvalue becomes greater than 0)

% Set up eigenvalue problem
global N, N = 40;
% N is the amount of Chebyshev interpolation points the program uses
% 2nd- and 4th-order differentiation matrices:
global x, global D2, global D4, global I
[D,x] = cheb(N); D2 = D^2; D2 = D2(2:N,2:N);
S = diag([0; 1 ./(1-x(2:N).^2); 0]);
D4 = (diag(1-x.^2)*D^4 - 8*diag(x)*D^3 - 12*D^2)*S;
D4 = D4(2:N,2:N); %Obtaining the Chebyshev differentiation matrices

% Orr-Sommerfeld operators A,B and generalized eigenvalues:
I = eye(N-1);  % Writing down the operators involved in the eigenvalue problem

% Set up our problem
Rs = 5700:5800;  % Narrow range of Reynolds numbers around the critical value
ks = 0:0.01:1.60;  % Sweep the range of wavevectors
i = 1;
for R = Rs
    for k = ks
        if find_ee(k, R) > 0
            kc(i) = k;
            Rc(i) = R;
            i = i + 1;
        end
    end
end

function max_ee = find_ee(k, R)
    global x, global N, global D2, global D4, global I
    A = (D4-2*k^2*D2+k^4*I)/R - 2i*k*I - 1i*k*diag(1-x(2:N).^2)*(D2-I);
    B = D2-k^2*I;
    ee = eig(A,B); % Solve the eigenvalue problem
    max_ee = max(real(ee));
end
