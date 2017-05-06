function [prediction_train, prediction_test] = logisticRegressionBasic(X, y, X_test, y_test)
	
	
	% X = multinom(X, 9);
	% X_test = multinom(X_test, 9);
	
	% m = Number of examples
	% n = Number of features
	[m_train n] = size(X);
	[m_test n] = size(X_test);
	
	% first we normalize the X features
	[X_norm, mu, sigma] = featureNormalize(X);
	% first we normalize the X features
	[X_test_norm, mu, sigma] = featureNormalize(X_test);
	
	% Plot training data
	% plotData(X, y, 1, 2, 'Class', 'Sex', 'Survied', 'Died');

	X_norm = [ones(m_train, 1) X_norm];
	X_test_norm = [ones(m_test, 1) X_test_norm];
	% Initialize fitting parameters
	initial_theta = zeros(size(X_norm, 2), 1);

	% Set regularization parameter lambda to 1 (you should vary this)
	lambda = 1;

	% Set Options
	options = optimset('GradObj', 'on', 'MaxIter', 400);

	% Optimize
	[theta, J, exit_flag] = ...
		fminunc(@(t)(costFunctionReg(t, X_norm, y, lambda)), initial_theta, options);

	% Compute accuracy on our training set
	p_train = predict(theta, X_norm);
	p_test = predict(theta, X_test_norm);

	prediction_train = mean(double(p_train == y)) * 100;
	prediction_test = mean(double(p_test == y_test)) * 100;

end