%% ITERATE 1
clear; clc;
tauCount = 8; aBarCount = 20;
k=1; gammaDot=1; nees=6;
i = 1;
sol(tauCount, aBarCount) = OSSolution;
for tau=linspace(0.8, 1.6, tauCount)
    tBar = gammaDot * tau;
    j = 1;
    for aBar=linspace(0, 8, aBarCount)
        % Solve eigenproblem and create OSSolution object.
        sol(i,j) = generateOSSol(k, aBar, tau, gammaDot, nees);
        j=j+1;
    end
    i=i+1;
end
disp("Finished!")

%% ITERATE 2
clear; clc;
tauCount = 8; aBarCount = 20;
k=realmin; gammaDot=1; nees=6;
i = 1;
sol(tauCount, aBarCount) = OSSolution;
for tau=linspace(0,0.8, tauCount)
    tBar = gammaDot * tau;
    j = 1;
    for aBar=linspace(0, 8, aBarCount)
        % Solve eigenproblem and create OSSolution object.
        sol(i,j) = generateOSSol(k, aBar, tau, gammaDot, nees);
        j=j+1;
    end
    i=i+1;
end
disp("Finished!")

%% FUNCTIONS 
function ossol = generateOSSol(k, aBar, tau, gammaDot, nees)
    [ev, ef] = findEigValsSub(k, aBar, tau, gammaDot*tau, nees);
    PsiTilda = chebfun(ef(1,:));
    Psi = unsub(PsiTilda, [-1/2 1/2]);
    Qxx = chebfun(ef(2,:));
    Qxy = chebfun(ef(3,:));
    ossol = OSSolution(Psi, Qxx, Qxy, ev, k, aBar, tau, gammaDot);
end

function Psi = unsub(PsiTilda, interval)
    y = PsiTilda.points;
    addTerm = chebfun(-1/(2*pi) * cos(pi * y), interval);
    Psi = PsiTilda + addTerm;
end

function [sigma, eigfcns] = findEigValsSub(k, aBar, tau, tBar, nees)
    % CONSTANTS
    l = 0.1; % Correlation length
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
        -r^3/2*cos(r*y) + diff(Psi,4)...
        - 2*k^2*(r/2*cos(r*y) + diff(Psi, 2)) ...
        + k^4 * (-1/(2*r)*cos(r*y) + Psi)... 
        - (aBar*(k^2*Qxy + diff(Qxy,2) + 2i*k*diff(Qxx)));

        Qxx - d1*(1/2*sin(r*y) + diff(Psi)) ...
        + (l/W)^2*(k^2.*Qxx - diff(Qxx,2))...
        + d2*(y.*Qxx)...
        - tBar*Qxy...
        + c1*(k^2* (-1/(2*r)*cos(r*y) + Psi) - (r/2*cos(r*y) + diff(Psi,2))); 

        tBar*Qxx + Qxy + d2*(y.*Qxy)...
        + (l/W)^2*(k^2*Qxy - diff(Qxy,2))...
        - c2*(k^2*(-(2*r)^-1*cos(r*y) + Psi) - (r/2*cos(r*y) + diff(Psi,2)))...
        - lambda*tBar*(k^2*(-(2*r)^-1*cos(r*y) + Psi) + (r/2*cos(r*y) + diff(Psi,2)))]; 

    A.lbc = @(Psi, Qxx, Qxy) [diff(Qxx); diff(Psi); Psi; diff(Qxy)];
    A.rbc = @(Psi, Qxx, Qxy)[diff(Qxx); diff(Psi); Psi; diff(Qxy)];

    % SOLVE FOR THE EIGENVALUES
    sprintf("Solving eigenvalue problem for k=%0.3f, aBar=%0.3f, tBar=%0.3f", k, aBar, tBar)
    tic; [eigfcns, D] = eigs(A,B, nees, 'SM'); toc
    ee = diag(D);
    sigma = ee / (-tau);
end