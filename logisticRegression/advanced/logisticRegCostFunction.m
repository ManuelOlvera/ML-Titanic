function [J, grad] = logisticRegCostFunction(X, y, theta, lambda)
%LOGISTICREGCOSTFUNCTION Compute cost and gradient for regularized logistic 
%regression with multiple variables

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

J = 0;
grad = zeros(size(theta));

g = sigmoid(X * theta);
logleftside = sum(-y' * log(g));
logrightside = sum((ones(size(y)) - y)' * log(ones(size(y)) - g));
regularization = lambda/(2*m) * sum(realpow(theta([2:end]),2));
J = (1/m * (logleftside - logrightside)) + regularization;


error_vector = sigmoid(X * theta) - y;
regularized_theta = (lambda/m) .* theta(2:end);
grad = (1/m * (X' * error_vector)) + vertcat(0, regularized_theta);

% grad = grad(:);

end
