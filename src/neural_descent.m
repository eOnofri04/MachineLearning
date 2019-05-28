%NEURAL_DESCENT Execute the gradient descent on the given neural network.
%   Evaluate an optimal value for the weigths `Theta` by applying a Batch
%   Gradient Descent with the given parameters by iterating
%   ```math
%       \Theta^{(l)}_{i, j} -= 
%           \eta\frac{\partial J}{\partial \Theta^{(l)}_{i, j}}
%   ```
%   
%   @param Theta  : `L-1` Cell-row-vector of the weight matrices between
%                   each couple of layers. `Theta{l}(i j)` is the weight
%                   from `i`-th unit in the `l` layer to the `j`-th unit
%                   in the`l+1` layer.
%   @param s      : `L` row-vector of the number of units in each layer.
%   @param X      : `m` by `n` matrix of the Training Set Features;
%                   each row represents a sample with `n` features.
%   @param y      : `m` column vector of the Training Set Samples.
%   @param eta    : learning rate of the batch gradient descent.
%   @opt lambda   : regularization parameter (*By Default* = 0).
%   @opt err      : descent sensibility (*By Default* = 10-6).
%   @opt it_max   : max iterations of the descent (*By Default* = 10+3).
%
%   @return Theta : Optimal weigth value.
%   @return Jh    : `i` row-vector containing the historical of the cost
%                   associated to each iteration.
%   @return iter  : number of iteration accomplished (no. updates -1).
%
function [ Theta, Jh, iter ] = neural_descent( Theta, s, X, y, eta, lambda, err, it_max)
%% Optional Parameters Initialization
    assert(nargin >= 5, 'ERROR: too few arguments');
    if nargin < 8
        it_max = 1000;
        if nargin < 7
            err = 0.000001;
            if nargin < 6
                lambda = 0;
            end
        end
    end
 
%% Parameters Initialization && Correctness Check
    assert(it_max > 1, 'ERROR: at least one iteration must be specified');
    
    L = length(s);
    assert(size(s, 1) == 1, 'ERROR: s malformed');
    
    assert(size(X, 1) == length(y), 'ERROR: Size of X and y do not agree');
    assert(size(X, 2) == s(1), ...
        'ERROR: Training sample do not agree with s(1)');
    
    assert(length(Theta) == L-1, 'ERROR: Theta malformed');
    for l = 1 : L-1
        assert(isequal(size(Theta{l}), [s(l)+1, s(l+1)]), ...
            'ERROR: Theta dimensions do not agree with s');
    end
    iter = 2;
    
%% Initial Cost acquisition
    [Jh(2), Jgrad] = neural_cost(Theta, s, X, y, lambda);
    Jh(1) = Jh(2) + 2 * err;

%% Continuous Learning
    while abs(Jh(iter-1) - Jh(iter)) > err && iter <= it_max
        %% Update Weigth
        for l = 1 : L-1
            Theta{l} = Theta{l} - eta * Jgrad{l};
        end
        
        %% New Cost Evaluation
        iter = iter+1;
        [Jh(iter), Jgrad] = neural_cost(Theta, s, X, y, lambda);
    end

%% Output
    iter = iter - 1;
    Jh = Jh(2:end);
end

