clear, clc

%% 1. FIXING k and tBar to investigate critical aBar -- 1
% tBar=10, aBar*tBar ~ 10:0.1:11
k=3;
tau = 1; gammaDot = 10;  
tBar = tau * gammaDot; nees=6; i=1;
for aBar = 1:0.01:1.1
    S1(i,:) = findEigVals(k, aBar, tau, tBar, nees);
    i = i+1;
end

%% 2. FIXING k and aBar to investigate critical gammaDot -- 2
k=3; tau=1; nees=6; i=1;
for gammaDot = 10:10:100
   S2(i,:) = findEigVals(k, aBar, tau, tau*gammaDot, nees);
   i = i+1;
end

%% 3. Testing a newtonian-nonnewtonian transition point from Wan's paper
% tau / tauA = gammaDot * aBar * tau = 0.6
% tBar = gammaDot * tau = 1.5
k = 1; gammaDot = 3; tau = 0.5; aBar = 0.4;
tBar = gammaDot * tau; 
nees = 6;
i = 1;
% Consider preallocating later for optimization
for aBar = 0.1:0.1:1.6
    [S3(i,:), eigfcns] = findEigVals(k, aBar, tau, gammaDot * tau, nees);
    Psi(i) = ChebfunWrapper(chebfun(eigfcns(1,:)));
    Qxx(i) = ChebfunWrapper(chebfun(eigfcns(2,:)));
    Qxy(i) = ChebfunWrapper(chebfun(eigfcns(3,:)));
    i = i+1;
end

%% 4. Testing a newtonian-nonnewtonian transition point from Wan's paper
% tBar = gammaDot * tau = 0.6
k = 1; gammaDot = 1; tau = 0.6; 
tBar = gammaDot * tau; 
nees = 6;
i = 1;
% Consider preallocating later for optimization
for aBar = 1.3:0.15:4.1 % aBar * tBar ranges from 1.95 to 6.15
    [S4(i,:), eigfcns] = findEigVals(k, aBar, tau, gammaDot * tau, nees);
    Psi(i) = ChebfunWrapper(chebfun(eigfcns(1,:)));
    Qxx(i) = ChebfunWrapper(chebfun(eigfcns(2,:)));
    Qxy(i) = ChebfunWrapper(chebfun(eigfcns(3,:)));
    i = i+1;
end

%% 5. Testing a newtonian-nonnewtonian transition point from Wan's paper
% tBar = 0.3
k = 1; gammaDot = 1; tau = 0.3; 
tBar = gammaDot * tau; 
nees = 6;
i = 1;
% Consider preallocating later for optimization
aBars = 3:0.33:6.4;
for aBar = aBars 
    [S5(i,:), eigfcns] = findEigVals(k, aBar, tau, gammaDot * tau, nees);
    Psi(i) = ChebfunWrapper(chebfun(eigfcns(1,:)));
    Qxx(i) = ChebfunWrapper(chebfun(eigfcns(2,:)));
    Qxy(i) = ChebfunWrapper(chebfun(eigfcns(3,:)));
    i = i+1;
end

%% 6. Testing a newtonian-nonnewtonian transition point from Wan's paper
% tBar = 0.3
k = 1; gammaDot = 1; tau = 0.4; 
tBar = gammaDot * tau; 
nees = 6;
i = 1;
% Consider preallocating later for optimization
aBars = 2.5:0.33:5.3;
for aBar = aBars 
    [S6(i,:), eigfcns] = findEigVals(k, aBar, tau, gammaDot * tau, nees);
    Psi(i) = ChebfunWrapper(chebfun(eigfcns(1,:)));
    Qxx(i) = ChebfunWrapper(chebfun(eigfcns(2,:)));
    Qxy(i) = ChebfunWrapper(chebfun(eigfcns(3,:)));
    i = i+1;
end
disp("Finished!")

%% 7 Testing a newtonian-nonnewtonian transition point from Wan's paper
k=1; gammaDot=1; tau=0.5; nees=6;
aBars = 2:0.33:5.3;
[S7, Psi7, Qxx7, Qxy7] = aBarIterate(k, tau, gammaDot, nees, aBars);

%% 8. Ditto
k=1; gammaDot=1; tau=0.6; nees=6;
aBars8=1.67:0.33:4.41;
[S8, Psi8, Qxx8, Qxy8] = aBarIterate(k, tau, gammaDot, nees, aBars8);

%% 9. Ditto
k=1; gammaDot=1; tau=0.7; nees=6;
aBars9 = 1.42:0.33:3.7;
[S9, Psi9, Qxx9, Qxy9] = aBarIterate(k, tau, gammaDot, nees, aBars9);

%% 10. Ditto
k=1; gammaDot=1; tau=0.8; nees=6;
aBars10 = 1.25:0.33:3.4;
[S10, Psi10, Qxx10, Qxy10] = aBarIterate(k, tau, gammaDot, nees, aBars10);


%% TESTING OSSolutions wrapper
clear; clc;
tauCount = 8; aBarCount = 20;
k=0; gammaDot=1; nees=6;
i = 1;
sol(tauCount, aBarCount) = OSSolution;
for tau=linspace(0, 0.7, tauCount)
    tBar = gammaDot * tau;
    j = 1;
    for aBar=linspace(0, 8, aBarCount)
        % Solve eigenproblem and create OSSolution object.
        [ev, ef] = findEigVals(k, aBar, tau, gammaDot*tau, nees);
        Psi = chebfun(ef(1,:));
        Qxx = chebfun(ef(2,:));
        Qxy = chebfun(ef(3,:));
        sol(i,j) = OSSolution(Psi, Qxx, Qxy, ev, k, aBar, tau, gammaDot);
        j = j+1;
    end
    i=i+1;
end
disp("Finished!")

%% TESTING TBAR INCREASE WITH FIXED BOUNDARY CONDITIONS ON DyPsi = pm 1/2
clear; clc;
tauCount = 8;
k=4; aBar=4.6318; gammaDot=1; nees=6;
i = 1;
for tau=linspace(0, 0.7, tauCount)
    [ev, ef] = findEigVals(k, aBar, tau, gammaDot*tau, nees);
    Psi = chebfun(ef(1,:));
    Qxx = chebfun(ef(2,:));
    Qxy = chebfun(ef(3,:));
    sol(i) = OSSolution(Psi, Qxx, Qxy, ev, k, aBar, tau, gammaDot);
    i = i+1;
end
disp("Finished!")

%% Probing around...
k=1; aBar=3; gammaDot=1; tau=0.5; tBar=2; nees=6;
s1 = generateOSSol(k, aBar, tau, gammaDot*tau, nees);

%% FUNCTIONS
function oss = generateOSSol(k, aBar, tau, tBar, nees)
    [ev, ef] = findEigVals(k, aBar, tau, tBar, nees);
    Psi = chebfun(ef(1,:));
    Qxx = chebfun(ef(2,:));
    Qxy = chebfun(ef(3,:));
    oss = OSSolution(Psi, Qxx, Qxy, ev, k, aBar, tau, tBar/tau);
end

function [sigma, eigfcns] = findEigVals(k, aBar, tau, tBar, nees)
    % CONSTANTS
    l = 0.1; % Correlation length
    lambda = 1; % does this have a name?
    W = 40; % Channel width
    
    % DEFINED CONSTANTS
    c1 = lambda * tBar^2 / (1 + tBar^2);
    c2 = c1 * tBar;
    d1 = 2i*k*lambda*tBar;
    d2 = 1i * k * tBar;
    
    % DEFINE OPERATORS
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

    A.lbc = @(Psi, Qxx, Qxy) [diff(Psi) + 0.5; Psi; diff(Qxx); diff(Qxy)];
    A.rbc = @(Psi, Qxx, Qxy)[diff(Psi) - 0.5; Psi; diff(Qxx); diff(Qxy)];

    % SOLVE FOR THE EIGENVALUES
    sprintf("Solving eigenvalue problem for k=%0.3f, aBar=%0.3f, tBar=%0.3f", k, aBar, tBar)
    tic; [eigfcns, D] = eigs(A,B, nees, 'SM'); toc
    ee = diag(D);
    sigma = ee / (-tau);
end