 % CONSTANTS
tauCount = 8; aBarCount = 20;
k=1; gammaDot=1; nees=6;

l = 0.005; % Correlation length
lambda = 1; % particle shape parameter
W = 1; % Channel width

% DEFINED CONSTANTS
c1 = lambda * tBar^2 / (1 + tBar^2);
c2 = c1 * tBar;
d1 = 2i*k*lambda*tBar;
d2 = 1i * k * tBar;
r = pi;

% DEFINE OPERATORS
B = chebop(-1/2, 1/2);
B.op = @(y,Psi, Qxx, Qxy)[0 * Psi; Qxx; Qxy];

A = chebop(-1/2, 1/2);
A.op = @(y,Psi, Qxx, Qxy) [...
    (-r^3/2*cos(r*y) + diff(Psi,4)...
    - 2*k^2*(r/2*cos(r*y) + diff(Psi, 2)) ...
    + k^4 * (-1/(2*r)*cos(r*y) + Psi)... 
    - (aBar*(k^2*Qxy + diff(Qxy,2) + 2i*k*diff(Qxx))))/-tau;

    (Qxx - d1*(1/2*sin(r*y) + diff(Psi)) ...
    + (l/W)^2*(k^2.*Qxx - diff(Qxx,2))...
    + d2*(y.*Qxx)...
    - tBar*Qxy...
    + c1*(k^2* (-1/(2*r)*cos(r*y) + Psi) - (r/2*cos(r*y) + diff(Psi,2))))/-tau; 

    (tBar*Qxx + Qxy + d2*(y.*Qxy)...
    + (l/W)^2*(k^2*Qxy - diff(Qxy,2))...
    - c2*(k^2*(-(2*r)^-1*cos(r*y) + Psi) - (r/2*cos(r*y) + diff(Psi,2)))...
    - lambda*tBar*(k^2*(-(2*r)^-1*cos(r*y) + Psi) + (r/2*cos(r*y) + diff(Psi,2))))/-tau]; 

A.lbc = @(Psi, Qxx, Qxy) [diff(Qxx); diff(Psi); Psi; diff(Qxy)];
A.rbc = @(Psi, Qxx, Qxy)[diff(Qxx); diff(Psi); Psi; diff(Qxy)];