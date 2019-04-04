%linear_gradient_descent Compute the batch gradient descent
%   Evaluate the gradient descent over the `J` cost function with learning
%   rate `alpha` according to the following simoultaneous updating:
%   ```math
%       \theta_i = \theta_j - \alpha \frac{\partial}{\partial \theta_i}J(\theta)
%   ```
%
%   @param Xo     : `m` by `n+1` matrix of the Training Set Features.
%                   each row represent a sample with `n` features.
%   @param Y      : `m` column vector of the Training Set Samples.
%   @param theta  : `n+1` column vector of the hypotesis coefficient
%   @param alpha  : learning rate of the algorithm
%   @param it_max : iterations of the gradient descent.
%
%   @return theta : optimal `theta` obtained from the gradient descent.
%   @return J_history   : vector of the `J` function values for each step.

function [ theta, J_history ] = linear_gradient_descent( Xo, Y, theta, alpha, it_max )

    [m, n1] = size(Xo);
    assert(isequal(size(theta), [n1, 1]));
    assert(isequal(size(Y), [m, 1]));
    J_history = zeros(it_max, 1);

    for i = 1 : it_max
        [J_history(i), gradient] = linear_cost(Xo, Y, theta);
        theta = theta - alpha * gradient;
    end

end