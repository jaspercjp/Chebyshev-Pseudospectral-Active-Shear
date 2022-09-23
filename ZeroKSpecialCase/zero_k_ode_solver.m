syms Psi(y) Qxx(y) Qxy(y) tBar sigma tau lambda

% % CONSTANTS
% gammaDot = 1; tau = 1; tBar = gammaDot * tau;
% l = 0.1; % Correlation length
% lambda = 1; % particle shape parameter
% W = 1; % Channel width
% sigma = 0.5;
% 
% % DEFINED CONSTANTS
% c1 = lambda * tBar^2 / (1 + tBar^2);
% c2 = c1 * tBar;
% d1 = 2i*k*lambda*tBar;
% d2 = 1i * k * tBar;

eqn1 = diff(Psi,4) - tBar*diff(Qxy,2) == 0;
eqn2 = Qxx + sigma*tau*Qxx - tBar*Qxy - c1*diff(Psi,2) == 0;
eqn3 = Qxy + sigma*tau*Qxy + c2*diff(Psi,2) - lambda*tBar*diff(Psi,2) + tBar*Qxx == 0;


system = [eqn1; eqn2; eqn3];
S = dsolve(system);
