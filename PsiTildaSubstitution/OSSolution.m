classdef OSSolution
    properties
       Psi;
       Qxx;
       Qxy;
       eigvals;
       k;
       aBar;
       tau;
       gammaDot;
       tBar;
    end
    methods
        function obj = OSSolution(Psi, Qxx, Qxy, eigvals, k, aBar, tau, gammaDot)
            if (nargin == 8)
               obj.Psi = Psi;
               obj.Qxx = Qxx;
               obj.Qxy = Qxy;
               obj.eigvals = eigvals;
               obj.k = k;
               obj.aBar = aBar;
               obj.tau = tau;
               obj.gammaDot = gammaDot;
               obj.tBar = gammaDot * tau;
            end
        end
    end
end