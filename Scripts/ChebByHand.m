N = 19;

[D1, y] = cheb(N); D2=D1^2; D3=D1^3; D4=D1^4; 

%% CONSTANTS
l = 0.1; % Correlation length
lambda = 1; % particle shape parameter
W = 1; % Channel width
   

%% PARAMETERS
k=3; aBar=1.05; gammaDot=1; tau = 10; tBar = gammaDot * tau;
%% DEFINED CONSTANTS
c1 = lambda * tBar^2 / (1 + tBar^2);
c2 = c1 * tBar;
d1 = 2i*k*lambda*tBar;
d2 = 1i * k * tBar;

%% OPERATORS
A1 = D4 + 2*k^2*D2 + k^4*eye(N+1); 
B1 = - 2i*k*aBar*D1;
C1 = -aBar * (k^2*eye(N+1) + D2);

A2 = -2i*k*lambda*D1 + (lambda * tBar^2)/(1 + tBar^2)*(k^2*eye(N+1) - D2);
B2 = eye(N+1) + (l/W)^2*(k^2*eye(N+1) - D2) + 1i*k*tBar*y;
C2 = tBar * eye(N+1);

A3 = -lambda*tBar^3 / (1 + tBar^2) * (k^2*eye(N+1) - D2) ...
    - lambda*tBar*(k^2 + D2);
B3 = tBar * eye(N+1);
C3 = eye(N+1) + (l/W)^2*(k^2*eye(N+1) - D2) + 1i*k*tBar*y;

LHS = [A1 B1 C1; A2 B2 C2; A3 B3 C3;];
RHS = zeros(3*(N+1), 3*(N+1));
RHS(N+2:end, N+2:end) = eye(2*(N+1));

%% BOUNDARY CONDITIONS
...
    
%% SOLVE
[ef, ev] = eigs(LHS, RHS, 6, 'SM');
Psi = chebfun(ef(1:N+1, :));
Qxx = chebfun(ef(N+2:2*N+2, :));
Qxy = chebfun(ef(2*N+3:3*N+3, :));