function plotOS(oss)
    plotOrrSommerfeld(oss.Psi, oss.Qxx, oss.Qxy, ...
        oss.eigvals, oss.aBar, oss.k, oss.tBar);
end

function plotOrrSommerfeld(Psi, Qxx, Qxy, sigma, aBar, k, tBar)
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
