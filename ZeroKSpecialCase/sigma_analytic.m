close all; clear; clc;
aBarSteps=8000; tBarSteps=800;
ts = linspace(0,1.42,tBarSteps);
tsNeg = linspace(1.01,4.0,tBarSteps);
asNeg = linspace(-700,0,aBarSteps);
asPos = linspace(0,100,aBarSteps);
[tBar, aBarPos] = meshgrid(ts, asPos);
[tBarNeg, aBarNeg] = meshgrid(tsNeg, asNeg);

sigma1_posABar = (-2+aBarPos+(-2+aBarPos).*tBar.^2 - aBarPos.*tBar.^3 ...
    + sqrt((aBarPos.^2.*(1+tBar.^2-tBar.^3).^2) - 4*(tBar+tBar.^3).^2 ...
    - 4*aBarPos.*(tBar.^3+tBar.^5)));


sigma1_negABar = (-2+aBarNeg+(-2+aBarNeg).*tBarNeg.^2 - aBarNeg.*tBarNeg.^3 ...
    + sqrt((aBarNeg.^2.*(1+tBarNeg.^2-tBarNeg.^3).^2) - 4*(tBarNeg+tBarNeg.^3).^2 ...
    - 4*aBarNeg.*(tBarNeg.^3+tBarNeg.^5)));

% sigma2 = -(1./(2*(1 + tBar.^2))).*(2 + 2*tBar.^2 ...
%     + aBar.*(-1+(-1+tBar).*tBar.^2) + sqrt(aBar.^2.*(1 + tBar.^2 - ...
%           tBar.^3).^2 - 4*(tBar + tBar.^3).^2 - 4*aBar.*(tBar.^3 + tBar.^5)));

% calculates the critical activity values for different values of tBar
crit_activity_pos = zeros(1,tBarSteps);
for col=1:tBarSteps
   for row=1:aBarSteps-1
      if sign(real(sigma1_posABar(row+1,col))) ~= sign(real(sigma1_posABar(row,col)))
         sprintf("Found critical activity s1(%0.3f->%0.3f) at tBar=%0.3f, aBar=%0.3f",...
             sigma1_posABar(row,col), sigma1_posABar(row+1,col), tBar(row,col), aBarPos(row,col))
         crit_activity_pos(col) = aBarPos(row,col);
      end
   end
end

crit_activity_neg = zeros(1,tBarSteps);
for col=1:tBarSteps
   for row=1:aBarSteps-1
      if sign(real(sigma1_negABar(row+1,col))) ~= sign(real(sigma1_negABar(row,col)))
         sprintf("Found critical activity s1(%0.3f->%0.3f) at tBar=%0.3f, aBar=%0.3f",...
             sigma1_negABar(row,col), sigma1_negABar(row+1,col), tBarNeg(row,col), aBarNeg(row,col))
         crit_activity_neg(col) = aBarNeg(row,col);
      end
   end
end

% crit_activity2 = zeros(1,tBarSteps);
% for col=1:tBarSteps
%    for row=1:aBarSteps-1
%       if sign(real(sigma2(row+1,col))) ~= sign(real(sigma2(row,col)))
%          sprintf("Found critical activity s2(%0.3f->%0.3f) at tBar=%0.3f, aBar=%0.3f",...
%              sigma2(row,col), sigma2(row+1,col), tBar(row,col), aBar(row,col))
%          crit_activity2(col) = aBar(row,col);
%       end
%    end
% end

figure();hold on;
plot(ts, crit_activity_pos); plot(tsNeg, crit_activity_neg); 
xlabel("$\bar\tau$",'interpreter', 'latex');
ylabel("Critical Activity");
title("Stablity Plot at k=0 and l=0");
t1=text(1,13,"unstable", "FontSize", 15); 
t2=text(1.3,-13,"unstable", "FontSize", 15); 
t3=text(1.15,-1.7,"stable", "FontSize", 15);
% x2 = [tsPos, fliplr(tsPos)];
% inBetween = [crit_activity_pos, fliplr(crit_activity_neg)];
% fill(x2, inBetween, 'b');
% figure();
% plot(ts, crit_activity2);
% title("Stablity Plot at $k=0$ and $l=0$, \sigma_2",'interpreter','latex');
% ylabel("Critical Activity");
