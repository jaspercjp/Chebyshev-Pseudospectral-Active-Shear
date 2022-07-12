% Example 38 from Trefethen Ch14

% Solving the equation uxxxx = e^x with boundary conditions
% u(1)=u(-1)=u'(1)=u'(-1)=0

% Preliminary: We fit a polynomial to u(x). In this case we have
% p(x) = (1-x^2)q(x)

% First we have to determine the spectral operator that we would like to 
% use to discretize an approximation to the fourth derivative
% -----------------------------------------------------------------
N=15; [D, x] = cheb(N); % We use N+1=16 sample points. length(x) = N+1;
% Now we define the operator, which we get from taking the fourth
% derivative of p(x)
S = diag([0; 1./(1-x(2:N).^2); 0]);
L = (diag(1-x.^2)*(D^4) - (8 * diag(x) * (D^3)) - (12 * (D^2))) * S;
L = L(2:N, 2:N);

% Then we just have to simple inverse matrix operation in order to solve
% the equation
f = exp(x(2:N));
v = L\f; v = [0; v; 0];
plot(x, v, '.'); grid on; hold on
xs = -1:0.01:1; % cannot directly plot v against the uniform x axis
vv = polyval(polyfit(x, v, N), xs);
plot(xs, vv)


% Then we perform error analysis
% For this we have to find the exact solution to the problem, which
% involves a "Vandermonde matrix" and I have no idea what "A" is...
