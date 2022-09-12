% assume Psi = -1/(2*pi) * cos(pi * y)
nees = 6;
tau = 1;
gammaDot = 350;
tBar = gammaDot * tau;
l = 0.1;
W = 1; 
lambda = 1;

B = chebop(-1/2, 1/2);
B.op = @(y, Qxx, Qxy)[Qxx; Qxy];

A = chebop(-1/2, 1/2);
A.op = @(y, Qxx, Qxy) [
    Qxx/tau - (l/W)^2*diff(Qxx,2)/tau - (tBar/tau)*Qxy...
    - (lambda*tBar^2)/(tau*(1+tBar^2)) * (pi/(2*tau))*cos(pi*y);

    Qxy/tau  - (l/W)^2*diff(Qxy,2)/tau - (tBar/tau)*Qxx...
    - (lambda*tBar^3)/(tau*(1+tBar^2))*(pi/(2*tau))*cos(pi*y)];
   
A.lbc = @(Qxx, Qxy) [diff(Qxx); diff(Qxy)];
A.rbc = @(Qxx, Qxy) [diff(Qxx); diff(Qxy)];

tic; [eigfcns, D] = eigs(A,B, nees, 'SM'); toc;
ee = diag(D)