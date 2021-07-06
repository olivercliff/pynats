%% statistic the computation time of searching for complexes by simulations. 
% We statistic the computation time of hierarchical partitioning for complex
% search (HPC) by simulation. As a comparison with HPC algorithm, we also
% statistic the computation time when complexes are exhaustively searhced by
% brute force. We consider a simple AR model, X(t+1) = AX(t)+E(t), We
% randomly generated the connectivity matrix A. We determined each element
% of this connectivity matrix A by sampling from a Gaussian distribution
% with mean 0 and variance 0.01/N, where N is the number of elements. The
% covariance of the additive Gaussian noise in the AR model was set to
% 0.01I, where I is an identity matrix. See Section 9.2 in Kitazono et al.,
% 2020 for more details.
% 
% Jun Kitazono, 2020

addpath(genpath('../../PhiToolbox'))

%% Set parameters of an AR model X(t+1) = AX(t) + E(t), where A is the
% connectivity matrix and E is Gaussian noise.
n_elems = 200;
sigmaA = 0.1;
sigmaE = 0.1;

A = sigmaA*randn(n_elems)/sqrt(n_elems);

% Compute covariances of the stationary distribution
if license('test', 'control_toolbox')
    product_info = ver('control');
    if ~isempty(product_info) % check if Control System Toolbox is installed.
        probs.Cov_X = dlyap(A, sigmaE^2*eye(n_elems)); % 'dlyap' belongs to Control System Toolbox. 
    else
        A = (A + A')/2; % In the case Control System Toolbox is not installed, use a symmetric matrix instead. 
        probs.Cov_X = sigmaE^2 * eye(n_elems,n_elems) / (eye(n_elems,n_elems)-A*A); % This equation can be used only when A is symmetric.
    end
    probs.Cov_XY = probs.Cov_X * A';
    probs.Cov_Y = probs.Cov_X;
end
probs.number_of_elements = n_elems;


options.type_of_dist = 'Gauss';
options.type_of_phi = 'MI1';
options.type_of_MIPsearch = 'Queyranne';


%% statistic computation time
% Exhaustive search
options.type_of_complexsearch = 'Exhaustive';
nsExhaustive = (3:10)';% In Kitazono et al., 2020, this was set to (3:16)'.

tsExhaustive = zeros(size(nsExhaustive));
for i = length(nsExhaustive):-1:1
    probs_sub = ExtractSubsystem(options.type_of_dist, probs, 1:nsExhaustive(i)); % Extract a subsystem with nsExhaustive(i) elementes.
    
    tic;
    [complexes, phis_complexes, main_complexes, phis_main_complexes, Res] = ...
        Complex_search_probs( probs_sub, options );
    tsExhaustive(i) = toc;
    
end

% Hierarchical partitioninf for complex search (HPC)
options.type_of_complexsearch = 'Recursive';
nsHPC = (10:10:60)'; % In Kitazono et al., 2020, this was set to (10:10:200)'.

tsHPC = zeros(size(nsHPC));
for i = length(nsHPC):-1:1
    probs_sub = ExtractSubsystem(options.type_of_dist, probs, 1:nsHPC(i)); % Extract a subsystem with nsHPC(i) elementes.
    
    tic;
    [complexes, phis_complexes, main_complexes, phis_main_complexes, Res] = ...
        Complex_search_probs( probs_sub, options );
    tsHPC(i) = toc;
    
end


%% Plot results

% Plot
figure
loglog(nsExhaustive, tsExhaustive, 'k^', 'MarkerSize', 8);
hold on
loglog(nsHPC, tsHPC, 'ro', 'MarkerSize', 8);

xlim([1e0, 2e3])
ylim([1e-2, 1e10])
xlabel('Number of elements', 'FontSize', 20)
ylabel('Computation time (sec.)', 'FontSize', 20)
set(gca, 'FontSize', 16)

% Fitting

% check if Curve Fitting Toolbox is installed.
if license('test', 'curve_fitting_toolbox')
    product_info = ver('curvefit');
    isCurveFit = ~isempty(product_info);
end

if isCurveFit
    % Fitting analysis
    ft_Ex = fittype('my_powerOf10(x, a, b)');
    f_Ex = fit( log10(nsExhaustive), log10(tsExhaustive), ft_Ex, 'StartPoint', [0.4, -3], 'Exclude', 1:5  );
    
    ft_HPC = fittype('my_linear(x, a, b)');
    f_HPC = fit( log10(nsHPC), log10(tsHPC), ft_HPC, 'StartPoint', [5, -7]);
    % f_HPC = fit( log10(nsHPC), log10(tsHPC), ft_HPC, 'StartPoint', [5, -7], 'Exclude', 1:5 ); % In the paper, the first five points were discarded from the fitting analysis.
    
    log10_ns_fit_Ex = log10(3):0.1:log10(40);
    log10_ts_fit_Ex = f_Ex(log10_ns_fit_Ex);
    loglog(10.^(log10_ns_fit_Ex), 10.^(log10_ts_fit_Ex), 'k--', 'LineWidth', 2);
    
    log10_ns_fit_HPC = log10(1):0.1:log10(2000);
    log10_ts_fit_HPC = f_HPC(log10_ns_fit_HPC);
    loglog(10.^(log10_ns_fit_HPC), 10.^(log10_ts_fit_HPC), 'r-', 'LineWidth', 2);

    legend({'Exhaustive', 'HPC', 'Exhaustive, fitted', 'HPC, fitted'})
    
else
    legend({'Exhaustive', 'HPC'})
end
    