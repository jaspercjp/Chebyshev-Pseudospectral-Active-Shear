close all; clear; clc;

%% CONSTANTS
l = 0.1; % Correlation length
lambda = 1; % does this have a name?
W = 1; % Channel width

%% PARAMETERS 
k = 3;
tau = 1;
gammaDot = 10;   % imposed shear rate

aBar = 1.05; % 1 / gammaDot*tauA
tBar = tau * gammaDot;

%% DEFINED CONSTANTS
c1 = lambda * tBar^2 / (1 + tBar^2);
c2 = c1 * tBar;
d1 = 2i*k*lambda*tBar;
d2 = 1i * k * tBar;
nees = 4;

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

%% PLOT THE EIGENFUNCTIONS 
PLOT = false; % SET TO TRUE WHEN PLOTTING
if (PLOT)
    psiPlot = figure();
    subplot(2,3,1);
    plot(real(Psi));
    subtitleF = sprintf("$\\bar{a}=\\frac{1}{\\dot{\\gamma}\\tau_a}$=%0.3f, $k=$%0.3f, $\\bar{t}=\\dot{\\gamma}\\tau=$%0.3f", aBar, k, tBar);
    title("$\Psi$", "Interpreter", "latex");
    ylabel("Re$(\Psi)$", "Interpreter", "latex");
    subplot(2,3,4);
    plot(imag(Psi));
    ylabel("Im$(\Psi)$", "Interpreter", "latex");
    xlabel("y", "Interpreter", "latex");

    subplot(2,3,2);
    plot(real(Qxx));
    title("$Q_{xx}$", "Interpreter", "latex");
    ylabel("Re($Q_{xx}$)", "Interpreter", "latex");
    subplot(2,3,5);
    plot(imag(Qxx));
    ylabel("Im($Q_{xx}$)", "Interpreter", "latex");
    xlabel("y", "Interpreter", "latex");

    subplot(2,3,3);
    plot(real(Qxy));
    title("$Q_{xy}$", "Interpreter", "latex");
    ylabel("Re($Q_{xy}$)", "Interpreter", "latex");
    subplot(2,3,6);
    plot(imag(Qxx));
    ylabel("Im($Q_{xy}$)", "Interpreter", "latex");
    xlabel("y", "Interpreter", "latex");

    sgtitle(["Eigenfunctions of the Generalized Orr-Sommerfeld Eqns", subtitleF], 'FontSize', 13, 'Interpreter','latex');
    legend(arrayfun(@(z) "\sigma="+complexToString(z), sigma));
end

function complexStr = complexToString(z)
    x = real(z); y = imag(z);
    if (sign(y) == 1)
        complexStr = sprintf("%0.3f + %0.3fi", x, y);
    else
        complexStr = sprintf("%0.3f - %0.3fi", x, abs(y));
    end
end
