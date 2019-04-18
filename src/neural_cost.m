function [ J, J_grad ] = neural_cost( Theta, s, X, Y, lambda)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[t1, t2, L] = size(Theta);
[m, n] = size(X);
assert(size(s, 1) == 1);
assert(max(s(1:length(s)-1)) == t1);
assert(max(s(2:length(s))) == t2);
assert(isequal(size(Y), [m, 1]));

for t = 1 : m
    a(:,1) = X(t, :);
    a(:,l+1) = sigmf(Theta(:,:,l) * a(:, l), [1 0]);

J = 0;
J_grad = 0;
end

