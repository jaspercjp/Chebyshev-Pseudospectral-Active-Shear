%% Constants 
l = 0.1; % Correlation length
lambda = 1; % does this have a name?
W = 3; % Channel width

% How to deal with sigma???
sigma = -0.5;

%% PARAMETERS 
k = 1;
tau = 4;
tauA = 2;
gammaDot = 0.3;

%% SETTING UP CHEBYSHEV DIFFERENTIATION
N = 20; [D, y] = cheb(N);
D2 = D^2; D4 = D^4; I = eye(N+1);

% Discretizing the system of diff eqs
A1 = D4 - 2*k^2*D2 + k^4*I;
B1 = -(2*i*k /(gammaDot*tauA)) * D;
C1 = -(1 / (gammaDot * tauA)) * (k^2*I+D2);
A2 = -2*i*k*lambda*D ...
    + (lambda*(gammaDot*tau)^2 / (1+(gammaDot*tau)^2))*(k^2*I-D2);
B2 = I + (l/W)^2*(k^2*I - D2) + i*k*y*gammaDot*tau + sigma*tau*I;
C2 = gammaDot * tau;
A3 = -(lambda * (gammaDot*tau)^3) / (1+(gammaDot*tau)^2) * (k^2*I-D2)...
    - lambda*gammaDot*tau*(k^2*I + D2);
B3 = gammaDot * tau;
C3 = I + (l/W)^2*(I - D) + i*k*y*gammaDot*tau + sigma*tau*I;



%% SETUP EIGENVALUE PROBLEM
QxxOp = -B1*((I-(B2\C2)*(C3\B3))\B2)\(A2-C2*(C3\A3));
QxyOp = -C1*(I-C3\B3*B2\C2)*C3\(A3 - B3*B2\A2);
