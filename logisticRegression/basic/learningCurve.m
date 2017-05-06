function [error_train, error_val] = learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve

m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

for i = 1:m
	theta = trainLinearReg(X(1:i, :), y(1:i), lambda);
	[jTrain, gradTrain] = linearRegCostFunction(X(1:i, :), y(1:i), theta, 0);
	[jVal , gradVal] = linearRegCostFunction(Xval, yval, theta, 0);
	error_train(i) = jTrain;
	error_val(i) = jVal;
endfor

end
