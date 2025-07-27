%% Signal Generation
% Sparsity - Define the true frequency components
f_active = sort(round((rand(1, K) * (f_max*0.9) + f_max*0.05)*1e-7)*1e7);
amplitudes = logspace(-2,0,K);
amplitudes = amplitudes(randperm(K));
phases = 2 * pi * rand(1, K); %Random phases
x_t = zeros(1, N_nyquist);
for k = 1:K
    x_t = x_t + amplitudes(k) * cos(2 * pi * f_active(k) * t_nyquist + phases(k));
end
% Add a small amount of noise
x_t = awgn(x_t, SNR_input, 'measured');
disp(['Input SNR (dB): ', num2str(SNR_input)]);

% Calculate the true spectrum (for comparison)
X_f = fft(x_t) / N_nyquist;
X_f = fftshift(X_f);
freq_axis_nyquist = linspace(-fs_nyquist/2, fs_nyquist/2-1/T_sim, N_nyquist);

%% MWC Simulation

y_mwc = zeros(M, N_mwc_samples_per_channel); % Matrix to store MWC samples

% Simulate MWC channels
for i = 1:M
    % Mixing
    mixed_signal = x_t .* prn_sequences(i, :);

    % Low-pass filtering
    filtered_signal = filter(b_lpf, a_lpf, mixed_signal);

    % Sampling (Downsampling)
    % Find indices corresponding to MWC sampling times
    mwc_sample_indices = round( (0:N_mwc_samples_per_channel-1) * Ts_mwc / Ts_nyquist ) + 1;
    % Ensure indices are within bounds
    mwc_sample_indices = mwc_sample_indices(mwc_sample_indices <= N_nyquist);
    y_mwc(i, 1:length(mwc_sample_indices)) = filtered_signal(mwc_sample_indices);
end
% Adjust size if needed due to rounding
N_mwc_samples_per_channel = size(y_mwc, 2);
y_vec = [];
for i=1:M
    y_vec = [y_vec;y_mwc(i,:)']; % Vectorize the measurements
end
fprintf('MWC Simulation Complete. Generated %d samples per channel for %d channels.\n', N_mwc_samples_per_channel, M);
fprintf('Total measurements: %d\n', length(y_vec));

y_norm = norm(y_vec);

%% Compressive Sensing Reconstruction (using OMP)
% --- Basic OMP Implementation ---
target_sparsity = 2*K + 2; % Expected sparsity K (can be slightly overestimated)
residual = y_vec; % Initialize residual
support_indices = []; % Initialize support set
%z_hat = zeros(N_fft_recon, 1); % Initialize reconstructed spectrum vector
iteration = 0;
for iter = 1:target_sparsity
    iteration = iteration + 1 ;
    % --- Find atom (column) most correlated with residual ---
    correlations = abs(A_norm' * residual); % Use normalized A for correlation step
    [~, max_idx] = max(correlations);
    
    % --- Update support set ---
    if ~ismember(max_idx, support_indices)
        support_indices = [support_indices, max_idx];
    else
         % If index already selected, maybe break or pick next best
         correlations(max_idx) = -inf; % Avoid re-selecting immediately
         [~, max_idx] = max(correlations);
         if ~ismember(max_idx, support_indices)
             support_indices = [support_indices, max_idx];
         else
             fprintf('Warning: Could not find new index, stopping OMP.\n');
             break; % Stop if we can't find a new index
         end
    end

    % --- Solve least squares problem on the support set ---
    A_support = A(:, support_indices); % Use original A for LS solve
    z_support = A_support \ y_vec; % Least squares solution
    %z_support = pinv(A_support) * y_vec;

    % --- Update residual ---
    residual = y_vec - A_support * z_support;
    % --- Check stopping criterion (e.g., residual norm) ---
    if norm(residual) < 1e-3 * norm(y_vec)
        fprintf('OMP converged early at iteration %d.\n', iter);
        break;
    end
end

% --- Final estimate ---
z_hat = zeros(N_fft_recon, 1);
if ~isempty(support_indices)
    %A_support = A(:, support_indices);
    %z_support = A_support \ y_vec;
    z_hat(support_indices) = z_support; % Place estimated coefficients
end

% De-normalize the estimate if A was normalized for OMP correlation step
% This depends on how OMP uses the matrix. If LS uses original A, no de-norm needed.
% If LS used A_norm, then z_hat(support_indices) = z_hat(support_indices) ./ column_norms(support_indices)';

% Adjust the scale (FFT vs signal amplitude) - This might need calibration
% Often the reconstruction gives relative amplitudes. We scale based on original max.
final_residual_norm = norm(residual);
scale_factor = max(abs(X_f)) / max(abs(z_hat));
z_hat_scaled = z_hat * scale_factor; % Simple scaling for visualization
fprintf('OMP Reconstruction Complete.\n');

%% to ge the exact amplitudes, frequency and phases

positive_indices = freq_bins >= 0;  % logical index for positive frequencies

% Extract positive spectrum parts
X_f_pos = X_f(positive_indices);
z_hat_pos = z_hat(positive_indices);
freq_bins_pos = freq_bins(positive_indices);

% 1. Find peaks in the positive spectrum of original signal
[~, f_index_true_pos] = findpeaks(abs(X_f_pos), 'SortStr', 'descend');

if isempty(f_index_true_pos)
    warning('No peaks found in X_f_pos. Check signal generation or noise level.');
    f_index_true_pos = 1:K;  % fallback to dummy indices
elseif length(f_index_true_pos) < K
    warning('Only %d peaks found in X_f_pos. Padding to K = %d.', length(f_index_true_pos), K);
    f_index_true_pos(end+1:K) = f_index_true_pos(end); % repeat last index
else
    f_index_true_pos = f_index_true_pos(1:K);
end

% 2. Find peaks in the positive spectrum of reconstructed signal
[~, f_index_recon_pos] = findpeaks(abs(z_hat_pos), 'SortStr', 'descend');
f_index_recon_pos = f_index_recon_pos(1:K);  % Take top-K peaks

% 3. Map frequency indices to actual frequency values
f_input = freq_bins_pos(f_index_true_pos);      % Original signal peak frequencies (positive)
f_recon = freq_bins_pos(f_index_recon_pos);     % Reconstructed signal peak frequencies (positive)

% 4. Amplitudes in dB
amp_x_t_db = 20 * log10(abs(X_f_pos(f_index_true_pos)) + 1e-12);      % Original signal amplitudes in dB
amp_z_hat_db = 20 * log10(abs(z_hat_pos(f_index_recon_pos)) + 1e-12); % Reconstructed signal amplitudes in dB

% 5. Phases in degrees
phase_x_t = angle(X_f_pos(f_index_true_pos)) * (180/pi);
phase_z_hat = angle(z_hat_pos(f_index_recon_pos)) * (180/pi);

% --- Display Results ---
fprintf('\nOriginal Frequencies (GHz):\n');
disp(f_input * 1e-9);

fprintf('Reconstructed Frequencies (GHz):\n');
disp(f_recon * 1e-9);

fprintf('Original Amplitudes (dB):\n');
disp(amp_x_t_db.');

fprintf('Reconstructed Amplitudes (dB):\n');
disp(amp_z_hat_db.');

fprintf('Original Phases (degrees):\n');
disp(phase_x_t.');

fprintf('Reconstructed Phases (degrees):\n');
disp(phase_z_hat.');

%% Extract Reconstructed Values at Input Frequencies
[~,f_index] = findpeaks(abs(X_f(1:N_fft_recon/2)),'SortStr','descend');
f_input = freq_bins(f_index(1:K));
[~,f_index1] = findpeaks(abs(z_hat(1:N_fft_recon/2)),'SortStr','descend');
f_recon = freq_bins(f_index1(1:K));

freq_error = rms(f_input-f_recon);
amp_error = rms(abs(X_f(f_index(1:K)))-abs(z_hat(f_index1(1:K)))');
phase_error = rms(angle(X_f(f_index(1:K)))-angle(z_hat(f_index1(1:K)))')*180/pi;