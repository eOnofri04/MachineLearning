%linear_cost Compute the mean square error of the hypotesis function
%   Evaluate the mean square error of the hypothesis function `h`:
%   ```math
%       h_\theta(x) =
%       \theta_0 + x_1\theta_1 + \dots + x_n\theta_n =
%       X\theta 
%   ```
%   from the `m` effective value `Y` of the training set.
%
%   @param Xo     : `m` by `n+1` matrix of the Training Set Features;
%                   each row represent a sample with `n` features.
%   @param Y      : `m` column vector of the Training Set Samples.
%   @param theta  : `n+1` column vector of the hypotesis coefficient
%
%   @return J     : Mean Square Error `J` for the choosen `\theta`.
%   @return J_grad: `n+1` column vector of the gradient value of `J`

function [ J, J_grad ] = linear_cost( Xo, Y, theta )
 
    [m, n1] = size(Xo);
    assert(isequal(size(theta), [n1, 1]));
    assert(isequal(size(Y), [m, 1]));
    
    h = Xo * theta;
    
    J = (h - Y)' * (h - Y) / (2 * m);
    J_grad = Xo' * (h - Y) / m;

end

