classdef ChebfunWrapper
    properties
        fun
    end
    methods
        % f should be a chebfun 
        function obj = ChebfunWrapper(f)
            if (nargin == 1) 
                obj.fun = f;
            end
        end
    end
end