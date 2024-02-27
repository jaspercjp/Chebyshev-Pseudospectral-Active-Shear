Ns = 5:2:100; 
v_cond = zeros(length(Ns),1);
h_cond = zeros(length(Ns),1);
for i=1:length(Ns)
    N = Ns(i);
    [D,y] = cheb(N);
    S = diag([0; 1 ./(1-y(2:N).^2); 0]); 
    D2 = D^2; D2=D2(2:N, 2:N);
    D2_H = (diag(1-y.^2)*D^2 - 4*diag(y)*D - 2*eye(N+1)); 
    D2_H=D2_H(2:N,2:N);
    v_cond(i) = cond(D2);
    h_cond(i) = cond(D2_H);
end

figure(); hold on;
plot(Ns, v_cond);
plot(Ns, h_cond);
legend(["Vanilla", "Heinrichs"])
set(gca,"YScale","log");
set(gca,"XScale","log");
plot(Ns, Ns.^2, '--'); plot(Ns, Ns.^4, '.-');