function [prediction_train, prediction_test, prediction_val] = logisticRegressionAdvanced(X, y, X_test, y_test)

	% m = number of training examples
	% n = number of features
	[m n] = size(X);
	
	X_split = floor (0.6 * m);
	X = X([1:X_split], :);
	X_val = X([X_split:end], :);
	y = y([1:X_split], :);
	y_val = y([X_split:end], :);

	new_m = size(X, 1);
	p = 1;
	
	% Map X onto Polynomial Features and Normalize
	X_poly = multinom(X, p);
	[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
	X_poly = [ones(new_m, 1), X_poly];                   % Add Ones
	
	% Map X_poly_val and normalize (using mu and sigma)
	X_poly_val = multinom(X_val, p);
	X_poly_val = bsxfun(@minus, X_poly_val, mu);
	X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
	X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones
	
	
	% Map X_poly_test and normalize (using mu and sigma)
	X_poly_test = multinom(X_test, p);
	X_poly_test = bsxfun(@minus, X_poly_test, mu);
	X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
	X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones

	% Learning Curve for Logistic Regression
	lambda = 10;
	[error_train, error_val] = learningCurve(X_poly, y, X_poly_val, y_val, lambda);
	
	plot(1:new_m, error_train, 1:new_m, error_val);
	title('Learning curve for logistic regression');
	legend('Train', 'Cross Validation');
	xlabel('Number of training examples');
	ylabel('Error');
	axis([0 new_m 0 10]);

	%fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
	for i = 1:new_m
		%fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
	end

	fprintf('Program paused. Press enter to continue.\n');
	% pause;
	
	% Validation for Selecting Lambda
	% You will now implement validationCurve to test various values of 
	% lambda on a validation set. You will then use this to select the
	% "best" lambda value

	[lambda_vec, error_train, error_val] = validationCurve(X_poly, y, X_poly_val, y_val);

	%close all;
	%plot(lambda_vec, error_train, lambda_vec, error_val);
	%legend('Train', 'Cross Validation');
	%xlabel('lambda');
	%ylabel('Error');

	%fprintf('lambda\t Train Error\t Validation Error\n');
	%for i = 1:length(lambda_vec)
	%	fprintf(' %f\t %f\t %f\n', lambda_vec(i), error_train(i), error_val(i));
	%end

	% fprintf('Best lambda %f \n', lambda_vec(error_val == min(error_val)));
	
	[theta, J] = trainLogisticReg(X_poly, y, lambda);
	
	% Compute accuracy on our training set
	p_train = predict(theta, X_poly);
	p_test = predict(theta, X_poly_test);
	p_val = predict(theta, X_poly_val);
	
	man1 = multinom([2, 0, 31, 1, 0, 29, 1], p);
	man1 = bsxfun(@minus, man1, mu);
	man1 = bsxfun(@rdivide, man1, sigma);
	man1 = [ones(size(man1, 1), 1), man1];           % Add Ones
	
	woman1 = multinom([2, 1, 37, 1, 0, 29, 1], p);
	woman1 = bsxfun(@minus, woman1, mu);
	woman1 = bsxfun(@rdivide, woman1, sigma);
	woman1 = [ones(size(woman1, 1), 1), woman1];           % Add Ones
	
	% X contains class (1,2,3), sex(1:female, 0:male), age, # of sibilins, # of parents, ticket fare, embarked from (Cherbourg:1, Queenstown:2, Southampton:3)
	fprintf('Man would survive? %f \n', predict(theta, man1));
	fprintf('Woman would survive? %f \n', predict(theta, woman1));

	prediction_train = mean(double(p_train == y)) * 100;
	prediction_test = mean(double(p_test == y_test)) * 100;
	prediction_val = mean(double(p_test == y_val)) * 100;
	
end