%neural_cost Compute neural network cost function and associated gradient.
%   Evaluate the cost function by applying:
%    - forward propagation of the training set for each training sample
%       ```math
%           a^{(l+1)} = g(\Theta^{(l)} * a^{(l)}),
%           \qquad \forall l \in[1,L]
%       ```
%    - error evaluation of the last layer
%    - error backward propagation of the error
%    - evaluation of the gradient
%
%   @param Theta  : `L-1` Cell-row-vector of the weight matrices between
%                   eachcouple of layers. `Theta{l}(i j)` is the weight
%                   from `i`-th unit in the `l` layer to the `j`-th unit
%                   in the`l+1` layer.
%   @param s      : `L` row-vector of the number of units in each layer.
%   @param X      : `m` by `n` matrix of the Training Set Features;
%                   each row represents a sample with `n` features.
%   @param y      : `m` column vector of the Training Set Samples.
%   @param lambda : 
%
%   @return J     : Cost function for the choosen `theta`
%   @return J_grad: `n+1` column vector of the gradient value of `J`

function [ J, J_grad ] = neural_cost( Theta, s, X, y, lambda)
%% Parameters Initialization
    [~, L] = size(s);
    [m, ~] = size(X);
    a = cell(L, 1);
    Theta_biasless = cell(L-1, 1);
    delta = cell(L, 1);
    Y = zeros(m, size(Theta{L-1}, 2));
    Y(sub2ind(size(Y), 1:5000,y')) = 1;
    for l = 1 : L-1
        Theta_biasless{l} = Theta{l}(2:end, :);
    end

%% Input criteria check
    assert(size(s, 1) == 1);
    assert(length(Theta) == L-1);
    for i = 1 : L-1
        assert(isequal(size(Theta{i}), [s(i)+1 s(i+1)]))
    end
    assert(size(X, 2) == s(1));
    assert(isequal(size(y), [m, 1]));

%% Forward Propagation
    a{1} = [ones(m, 1) X];  % First Layer Initialization (+ bias)

    % Hidden Layer Evaluation (+ bias)
    for l = 1 : L-2
        a{l+1} = [ones(m, 1) sigmf(a{l} * Theta{l}, [1 0])];
    end

    % Last Layer Evaluation (no bias)
    a{L} = sigmf(a{L-1} * Theta{L-1}, [1 0]);

%% Error Evaluation on L
    delta{L} = a{L} - Y;
%% Error Backward Propagation
    for l = L-1 : -1 : 1
        delta{l} = a{l}(:, 2:end) .* (1 - a{l}(:, 2:end)) .* (delta{l+1} * Theta_biasless{l}');
    end
%% Gradient Evaluation

%% Output
J = 0;
J_grad = 0;

end