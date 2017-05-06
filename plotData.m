function plotData(X, y, x_axis, y_axis, x_label, y_label, positive_legend, negative_legend)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples.

% Create New Figure
figure; hold on;

% Find Indices of Positive and Negative Examples
pos = find(y == 1); neg = find(y == 0);
% Plot Examples
plot(X(pos, x_axis), X(pos, y_axis), 'k+','LineWidth', 2, 'MarkerSize', 7);
plot(X(neg, x_axis), X(neg, y_axis), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

hold off;

% Put some labels 
hold on;
% Labels and Legend
xlabel(x_label);
ylabel(y_label);

% Specified in plot order
legend(positive_legend, negative_legend);
hold off;

end
