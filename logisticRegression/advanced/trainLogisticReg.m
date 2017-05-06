function [theta, J] = trainLogisticReg(X, y, lambda)
%TRAINLOGISTICREG Trains logistic regression given a dataset (X, y) and a
%regularization parameter lambda

% Initialize Theta
initial_theta = zeros(size(X, 2), 1); 

% Create "short hand" for the cost function to be minimized
costFunction = @(t) logisticRegCostFunction(X, y, t, lambda);

% Now, costFunction is a function that takes in only one argument
options = optimset('MaxIter', 100, 'GradObj', 'on');

% Minimize using fmincg
% theta = fmincg(costFunction, initial_theta, options);
[theta, J, exit_flag] = fminunc(@(t)(costFunction(t, X, y, lambda)), initial_theta, options);

end
