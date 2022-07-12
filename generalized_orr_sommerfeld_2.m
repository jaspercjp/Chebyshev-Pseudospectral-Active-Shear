close all; clear; clc;

%% CONSTANTS
l = 0.1; % Correlation length
lambda = 1; % does this have a name?
W = 1; % Channel width

%% PARAMETERS 
k = 3;
tau = 1;
% tauA = 1; % replaced by aBar
gammaDot = 10;   % imposed shear rate

aBar = 0; % 1 / gammaDot*tauA
tBar = tau * gammaDot;

%% DEFINED CONSTANTS
c1 = lambda * tBar^2 / (1 + tBar^2);
c2 = c1 * tBar;
d1 = 2i*k*lambda*tBar;
d2 = 1i * k * tBar;
nees = 4;

DELTA = 0.5;

%% SETUP EIGENVALUE PROBLEM 

B = chebop(-1/2, 1/2);
B.op = @(y,Psi, Qxx, Qxy)[0 * Psi; Qxx; Qxy];

A = chebop(-1/2, 1/2);
A.op = @(y,Psi, Qxx, Qxy) [...
    (diff(Psi,4) - 2*k^2*diff(Psi, 2) + k^4 * Psi)... 
        - (aBar*(k^2*Qxy + diff(Qxy,2) + 2i*k*diff(Qxx)));
        
    Qxx - d1*diff(Psi) + (l/W)^2*(k^2.*Qxx - diff(Qxx,2)) + d2*(y.*Qxx)...
        - tBar*Qxy + c1*(k^2.*Psi - diff(Psi,2)); 
        
    tBar*Qxx + Qxy + d2*(y.*Qxy) + (l/W)^2*(k^2*Qxy - diff(Qxy,2))...
        - c2*(k^2*Psi - diff(Psi,2)) - lambda*tBar*(k^2*Psi + diff(Psi,2))]; 
    
A.lbc = @(Psi, Qxx, Qxy) [Psi; diff(Psi); diff(Qxx); diff(Qxy)];
A.rbc = @(Psi, Qxx, Qxy)[Psi; diff(Psi); diff(Qxx); diff(Qxy)];

%% SOLVE EIGENVALUE PROBLEM
disp("Solving eigenvalue problem...");
tic; [eigfcns,D] = eigs(A,B, nees, 'SM'); toc
ee = diag(D);
sigma = ee / (-tBar / gammaDot);

Psi = chebfun(eigfcns(1,:)); 
Qxx = chebfun(eigfcns(2,:));
Qxy = chebfun(eigfcns(3,:));

psiPlot = figure();
subplot(2,1,1);
plot(real(Psi));
subtitleF = sprintf("$\\bar{a}$=%0.3f, $k=$%0.3f, $\\bar{t}=$%0.3f", aBar, k, tBar);
title("$\Psi$ eigenfunctions", subtitleF, "Interpreter", "latex");
ylabel("Re$(\Psi)$", "Interpreter", "latex");
subplot(2,1,2);
plot(imag(Psi));
ylabel("Im$(\Psi)$", "Interpreter", "latex");
xlabel("y", "Interpreter", "latex");


figure();
plot(Qxx);
title("Qxx");
figure
plot(Qxy);
title("Qxy");
