clear; clc;
aBarSteps=30000; tBarSteps=30000;
ts = linspace(0,4,tBarSteps);
as = linspace(0,100,aBarSteps);
[tBar, aBar] = meshgrid(ts, as);

sigma1 = (-2+aBar+(-2+aBar).*tBar.^2 - aBar.*tBar.^3 ...
    + sqrt((aBar.^2.*(1+tBar.^2-tBar.^3).^2) - 4*(tBar+tBar.^3).^2 ...
    - 4*aBar.*(tBar.^3+tBar.^5)));

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