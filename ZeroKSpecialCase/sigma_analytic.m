clear; clc;
aBarSteps=3000; tBarSteps=100;
ts = linspace(0,4,tBarSteps);
as = linspace(0,3000,aBarSteps);
[tBar, aBar] = meshgrid(ts, as);

sigma1 = (-2+aBar+(-2+aBar).*tBar.^2 - aBar.*tBar.^3 ...
    + sqrt((aBar.^2.*(1+tBar.^2-tBar.^3).^2) - 4*(tBar+tBar.^3).^2 ...
    - 4*aBar.*(tBar.^3+tBar.^5)));

% calculates the critical activity values for different values of tBar
crit_activity = zeros(1,tBarSteps);
for col=1:tBarSteps
   for row=1:aBarSteps-1
      if sign(real(sigma1(row+1,col))) ~= sign(real(sigma1(row,col)))
         sprintf("Found critical activity (%0.3f->%0.3f) at tBar=%0.3f, aBar=%0.3f",...
             sigma1(row,col), sigma1(row+1,col), tBar(row,col), aBar(row,col))
         crit_activity(col) = aBar(row,col);
      end
   end
end

plot(ts, crit_activity); xlabel("$\bar\tau$",'interpreter', 'latex');
ylabel("Critical Activity");
title("Stablity Plot at $k=0$ and $l=0$",'interpreter','latex');