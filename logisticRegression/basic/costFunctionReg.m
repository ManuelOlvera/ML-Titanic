function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
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

end
