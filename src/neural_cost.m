%NEURAL_COST Compute neural network cost function and associated gradient.
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
%                   each couple of layers. `Theta{l}(i j)` is the weight
%                   from `i`-th unit in the `l` layer to the `j`-th unit
%                   in the`l+1` layer.
%   @param s      : `L` row-vector of the number of units in each layer.
%   @param X      : `m` by `n` matrix of the Training Set Features;
%                   each row represents a sample with `n` features.
%   @param y      : `m` column vector of the Training Set Samples.
%   @param lambda : regularization parameter.
%
%   @return J_grad: `n+1` column vector of the gradient value of `J`

function [ J, grad ] = neural_cost( Theta, s, X, y, lambda )
%% Parameters Initialization
    changing_bias = 0;
    [~, L] = size(s);
    [m, ~] = size(X);
    
    a = cell(L, 1);
    delta = cell(L, 1);
    grad = cell(L, 1);
    
    Y = zeros(m, s(L));
    Y(sub2ind([m s(L)], 1:m,y')) = 1;
    
    Theta_biasless = cell(L-1, 1);
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

%% Cost Evaluation on L
    J = sum(sum(Y .* log(a{L}) + (1-Y) .* log(1-a{L})));
    reg = 0;
    if lambda ~= 0
        for l = 1 : L-1
            reg = reg + sum(sum(Theta{l}.^2));
        end
    end
    J = (lambda/2 * reg - J) / m;
%% Error Evaluation on L
    delta{L} = a{L} - Y;
    
%% Error Backward Propagation
    if changing_bias == 1
        delta{L-1} = a{L-1} .* (1 - a{L-1}) .* (delta{L} * Theta{L-1}');
        for l = L-2 : -1 : 2
            delta{l} = a{l} .* (1 - a{l}) .* (delta{l+1}(:, 2:end) * Theta{l}');
        end
    else
        for l = L-1 : -1 : 2
            delta{l} = a{l}(:, 2:end) .* (1 - a{l}(:, 2:end)) .* (delta{l+1} * Theta_biasless{l}');
        end
    end
    
%% Gradient Evaluation
%     for l = 2 : L
%         %grad{l-1} = a{l-1}' * delta{l};
%         grad{l-1} = a{l-1}(:, 2:end)' * delta{l};
%     end
    Theta1_grad = zeros(size(Theta_biasless{1}));
    Theta2_grad = zeros(size(Theta_biasless{2}));
    for i = 1 : m
        Theta1_grad = Theta1_grad + (delta{2}(i, :)' * a{1}(i, 2:end))';
        Theta2_grad = Theta2_grad + (delta{3}(i, :)' * a{2}(i, 2:end))';
    end
    T1 = Theta1_grad'/m;
    T2 = Theta1_grad'/m;
    grad = [T1(:); T2(:)];
%% Output

end