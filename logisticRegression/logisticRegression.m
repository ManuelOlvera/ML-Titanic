function prediction = logisticRegression(X, y)

	% Plot training data
	% plotData(X, y, 1, 2, 'Class', 'Sex', 'Survied', 'Died');

	% m = Number of examples
	% n = Number of features
	[m n] = size(X);

	X = [ones(m, 1) X];
	% Initialize fitting parameters
	initial_theta = zeros(size(X, 2), 1);

	% Set regularization parameter lambda to 1 (you should vary this)
	lambda = 1;

	% Set Options
	options = optimset('GradObj', 'on', 'MaxIter', 400);

	% Optimize
	[theta, J, exit_flag] = ...
		fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

	% Compute accuracy on our training set
	p = predict(theta, X);

	prediction = mean(double(p == y)) * 100;

end