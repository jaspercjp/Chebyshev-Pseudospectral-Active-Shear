m = 1; mm = 1;
for i = 1:tauCount
    for j = 1:aBarCount
        s = sol(i,j);
        if (isStable(s.eigvals) == 0)
            unstablePts(m, 1) = s.tBar;
            unstablePts(m, 2) = s.aBar;
            m = m+1;
        else
            stablePts(mm, 1) = s.tBar;
            stablePts(mm, 2) = s.aBar;
            mm = mm+1;
        end
    end
end

plot(unstablePts(:,1), unstablePts(:,2), 'x', 'markersize', 11); hold on;
plot(stablePts(:,1), stablePts(:,2), '.', 'markersize', 14);

%% TESTING
figure();
m=1; mm=1;
[~, l] = size(sol);
for i = 1:l
    s = sol(i)
    if (isStable(s.eigvals) == 0)
        unstablePts(m, 1) = s.tBar;
        unstablePts(m, 2) = s.aBar;
        m = m+1;
    else
        stablePts(mm, 1) = s.tBar;
        stablePts(mm, 2) = s.aBar;
        mm = mm+1;
    end
end
plot(unstablePts(:,1), unstablePts(:,2), 'x', 'markersize', 11); hold on;
plot(stablePts(:,1), stablePts(:,2), '.', 'markersize', 14);

%% FUNCTIONS
function retVal = classify(sol)
        % first look for zero imaginary parts
        [m, n] = size(sol.eigvals);
        eigvals = reshape(sol.eigvals, n, m);
        for val = eigvals
            if (imag(val) == 0)
                if (sign(real(val)) >= 0)
                    retVal = OSSolutionType.Unstable;
                    return;
                else
                    retVal = OSSolutionType.Stable;
                end
            else
                if (sign(real(val)) >= 0)
                    retVal = OSSolutionType.UnstableOscillatory;
                else
                    retVal = OSSolutionType.StableOscillatory;
                end
            end
        end
      end

function v = eigvalsToStability(eigMat)
    l = size(eigMat);
    v = zeros(1, l(1));
    for i = 1:size(eigMat)
        v(i) = isStable(eigMat(i,:));
    end
end

function s = isStable(eigvals)
    [m,n] = size(eigvals);
    eigvals = reshape(eigvals, n, m);
    for val = eigvals
        if (sign(real(val)) == 1)
            s = 0;
            return
        end
    end
    s = 1;
end