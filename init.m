addpath('logisticRegression/');
%% Initialization
clear ; close all; clc

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')
train_data = csvread('data/train-data-converted.txt');
y = train_data(:, 2);
% X contains class (1,2,3), sex(1:female, 0:male), age, ticket fare, embarked from (Cherbourg:1, Queenstown:2, Southampton:3)
X = train_data(:, [3, 6, 7, 11, 13]);

logisticPred = logisticRegression(X, y);
fprintf('Train Accuracy: %f\n', logisticPred);