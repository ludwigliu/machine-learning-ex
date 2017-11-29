function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% calculate the hypothesis
x0 = ones(size(X(:, 1)));
a = horzcat(x0, X);
z2 = a * Theta1';
a2 = sigmoid(z2);
a0 = ones(size(X(:, 1)));
z3 = horzcat(a0, a2);
h = sigmoid(z3 * Theta2');

indexY = 1:num_labels;
y = y == indexY;

J = sum(sum( (-y).*log(h) - (1 - y).*log(1 - h), 2))/m;


% compute regularized item
% theta inclues 1 items, remove it before calculation
regItem = lambda/(2*m) * (sum(sum(Theta1(:, 2:end).^2, 2, "double")) + sum(sum(Theta2(:, 2:end).^2, 2, "double")));

J = J + regItem;

% backpropagation
Delta2ofi = 0;
Delta1ofi = 0;
for t = 1:m
    xt = X(t, :);
    yt = y(t, :);
    a1 = [1, xt];
    z2 = a1 * Theta1';
    a2 = sigmoid(z2);
    a2 = [1, a2];
    z3 = a2 * Theta2';
    a3 = sigmoid(z3);
    ht = a3;

    delta3 = a3 - yt;
    delta2 = (Theta2' * delta3')(2:end, :) .* sigmoidGradient(z2)';

    Delta2ofi = Delta2ofi + a2' * delta3;
    Delta1ofi = Delta1ofi + delta2 * a1;
end

Delta2ofi = Delta2ofi';

Theta1_grad(1, :) = Delta1ofi(1, :)/m;
Theta1_grad(2, :) = Delta1ofi(2, :)/m + lambda/m * Theta1(2, :);

Theta2_grad(1, :) = Delta2ofi(1, :)/m;
Theta2_grad(2, :) = Delta2ofi(2, :)/m + lambda/m * Theta2(2, :);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
