% Each row corresponds to one value of tBar, and various values of aBar
% We just have to test whether that row has an eigenvalue with positive
% real part or not

index = 1;

r5 = eigvalsToStability(S5);
for i = 1:length(aBars)
    if (r5(i)==0)
        tb(index) = tBar;
        ab(index) = aBars(i);
        index = index + 1;
    end
end
plot(tb, tBar*ab, ".", "Markersize", 14)

function v = eigvalsToStability(eigMat)
    l = size(eigMat);
    v = zeros(1, l(1));
    for i = 1:size(eigMat)
        v(i) = isStable(eigMat(i,:));
    end
end

function s = isStable(eigvals)
    for val = eigvals
        if sign(real(val))==1
            s = 0;
            return
        end
    end
    s = 1;
end