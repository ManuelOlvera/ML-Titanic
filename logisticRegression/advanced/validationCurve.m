function [lambda_vec, error_train, error_val] = validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

for i = 1:length(lambda_vec)
	theta = trainLogisticReg(X, y, lambda_vec(i));
	[jTrain, gradTrain] = logisticRegCostFunction(X, y, theta, 0);
	[jVal , gradVal] = logisticRegCostFunction(Xval, yval, theta, 0);
	error_train(i) = jTrain;
	error_val(i) = jVal;
endfor

end
