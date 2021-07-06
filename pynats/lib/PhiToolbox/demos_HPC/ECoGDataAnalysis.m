%% Apply Hierarchical partitioning for complex search (HPC) to ECoG data
% We used electrocorticography (ECoG) data recorded from a macaque monkey.
% The dataset is available at an open database, Neurotycho.org
% (http://neurotycho.org/). One hundred twenty-eight electrodes were
% implanted in the left hemisphere. Signals were sampled at a rate
% of 1 kHz. The monkey was awake with the eyes covered by an eye-mask to
% restrain visual responses. To remove line noise and artifacts, we
% performed bipolar re-referencing between adjacent electrode pairs, i.e.
% subtracting the signal of one electrode from that of the other. The
% number of bipolar re-referenced electrodes was 64 in total. Among the 64
% channels, two channels were removed from further analysis because of
% statisticment noise. We extracted 1-minute signals consisting of 1 kHz
% * 60 s = 60,000 samples. We searched for complexes. We approximated the
% probability distribution of the signals with multivariate Gaussian
% distributions. See Section 9.3 in Kitazono et al., 2020 for more details.
% 
% Jun Kitazono, 2020


addpath(genpath('../../PhiToolbox'))

%% Preprocess data
load('misc/Neurotycho/X_window01.mat')
% X: 1-minute signals of 64 channeles, which are bipolar-rereferenced. 64 by 60,000 (1 minutes * 60 sec. * 1000Hz) matrix

% select target channels
target_ch = 1:62;
X = X(target_ch, :);
N = length(target_ch);

params.tau = 1;
options.type_of_dist = 'Gauss'; % type of probability distributions
options.type_of_phi = 'MI1'; % type of phi
options.type_of_MIPsearch = 'Queyranne'; % type of MIP search
options.type_of_complexsearch = 'Recursive'; % type of complex search
probs = data_to_probs(X, params, options);


%% Find complexes and main complexes
tic;
[complexes, phis_complexes, main_complexes, phis_main_complexes, Res] = ...
    Complex_search_probs( probs, options );
t = toc;


%% Plot the results
colormap(KovesiRainbow(256));
cLim = [min(phis_complexes), max(phis_complexes)];

CortexMap = load('ChibiMap_bipolar.mat');
XCoor = CortexMap.X(target_ch);
YCoor = CortexMap.Y(target_ch);

subplot(1,2,1) % Main complexes
image(CortexMap.I*(1/5)+4*256/5); axis equal, hold on
plotComplexes(main_complexes, phis_main_complexes, '2D', 'XData', XCoor, 'YData', YCoor, 'cLim', cLim, 'NodeLabel', [], 'LineWidth', 2)
colorbar
title('Main Complexes')

xlim([0.5 1000.5]);
ylim([0.5, 1250.5]);
set(gca, 'XTickLabel', [], 'YTickLabel', [], 'xtick', [], 'ytick', [])


subplot(1,2,2) % Complexes
image(CortexMap.I*(1/5)+4*256/5); axis equal, hold on
plotComplexes(complexes, phis_complexes, '2D', 'XData', XCoor, 'YData', YCoor, 'cLim', cLim, 'NodeLabel', [], 'LineWidth', 2)
colorbar
title('Complexes')

xlim([0.5 1000.5]);
ylim([0.5, 1250.5]);
set(gca, 'XTickLabel', [], 'YTickLabel', [], 'xtick', [], 'ytick', [])