% load Octave package with useful functions
pkg load specfun;
% add path where functions are
addpath('logisticRegression/basic');
addpath('logisticRegression/advanced');
%% Initialization
clear ; close all; clc

% Load Training Data and Test Data
fprintf('Loading and Visualizing Data ...\n');
train_data = csvread('data/train-data-converted.csv');
% removing csv headers
train_data = train_data([2:end], :);
test_data = csvread('data/test-data-converted.csv');
% removing csv headers
test_data = test_data([2:end], :);
y_test = csvread('data/gender_submission.csv');
% removing csv headers
y_test = y_test([2:end], 2);
y = train_data(:, 2);
% X contains class (1,2,3), sex(1:female, 0:male), age, # of sibilins, # of parents, ticket fare, embarked from (Cherbourg:1, Queenstown:2, Southampton:3)
X = train_data(:, [3, 6, 7, 8, 9, 11, 13]);
X_test = test_data(:, [2, 5, 6, 7, 8, 10, 12]);

[logisticPred_train, logisticPred_test] = logisticRegressionBasic(X, y, X_test, y_test);
fprintf('Train Accuracy Train: %f\n', logisticPred_train);
fprintf('Train Accuracy Test: %f\n', logisticPred_test);

[logisticPred_train, logisticPred_test, logisticPred_val] = logisticRegressionAdvanced(X, y, X_test, y_test);
fprintf('Train Accuracy Train: %f\n', logisticPred_train);
fprintf('Train Accuracy Test: %f\n', logisticPred_test);
fprintf('Train Accuracy Val: %f\n', logisticPred_val);
