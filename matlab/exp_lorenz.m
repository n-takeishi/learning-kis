% Experiment on chaotic time-series prediction.
% This script shows the results of the experiment.

clear;
addpath('./jsonlab/');

dirname = 'result';
horizon = 10; % adjust with LKIS experiment
show_pred = 10;
range = 1:2500;

%% load results of LKIS-DMD

args = loadjson(sprintf('../exp_lorenz/%s/args.json', dirname));
delay = args.delay;
load(sprintf('../exp_lorenz/%s/output_test_0.mat', dirname), ...
    'prediction_rmses', 'y_preds', 'y1');

%% plot

pl1 = y_preds(show_pred,:);
pl2 = y1(show_pred:end-show_pred);

figure();
hold on;
plot(pl1(range), 'linewidth', 1.5);
plot(pl2(range), 'k--', 'linewidth', 1.5);
hold off;
set(gca, 'xticklabel', [])
grid on;
ylim([-Inf Inf]);
legend({sprintf('%d-step prediction', show_pred), 'truth'}, ...
    'location', 'southeast');