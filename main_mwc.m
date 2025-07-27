% MATLAB Code for MWC Compressive Sensing Simulation and Reconstruction
clear; 
clc;
close all;
%% Parameters Definition
% --- Signal Parameters ---
f_max = 3e9; % Maximum frequency in the original signal (e.g., MHz or GHz)
BW = 2 * f_max; % Total Bandwidth (Nyquist rate = BW)
T_sim = 100e-9; % Simulation duration needs to be long enough
fs_nyquist = 2 * f_max; % Nyquist sampling rate
Ts_nyquist = 1 / fs_nyquist; % Nyquist sampling period
N_nyquist = round(T_sim / Ts_nyquist); % Number of Nyquist samples in simulation duration
t_nyquist = (0:N_nyquist-1) * Ts_nyquist; % Time vector at Nyquist rate
% SNR_input = 40;

K = 2; % Number of active frequency bands (sparsity level)

% --- MWC Parameters ---
M = 4; % Number of MWC channels (typically > K)
fp = f_max / 40; % Frequency of the periodic PRN sequences (defines spectral folding)
Tp = 1 / fp; % Period of the PRN sequences
fs_mwc = 2 * fp; % Sub-Nyquist sampling rate for MWC channels (e.g., >= 2 * baseband BW after mixing)
Ts_mwc = 1 / fs_mwc; % MWC sampling period
N_mwc_samples_per_channel = round(T_sim / Ts_mwc); % Number of samples per channel

% --- LPF Parameters (Chebyshev Type I, 5th order) ---
fp_mhz = 225;
fsb_mhz = 300;
rp = 1;
rs = 20;
fs_nyquist_mhz = fs_nyquist / 1e6;   % Convert Hz to MHz if needed

% Convert to Hz
fp = fp_mhz * 1e6;
fsb = fsb_mhz * 1e6;

Wp = fp / (fs_nyquist / 2);
Ws = fsb / (fs_nyquist / 2);

n = 5;  % Fixed order to match ADS
[b_lpf, a_lpf] = cheby1(n, rp, Wp);  % Design filter
% Generate PRN sequences (e.g., binary +/-1 sequences derived from Gold codes or random)
prn_sequences = zeros(M, N_nyquist);
samples_per_period = round(Tp / Ts_nyquist);
% for i = 1:M
%     % Generate one period of a pseudo-random sequence (+1/-1)
%     prn_period = 2 * randi([0 1], 1, samples_per_period) - 1;
%     % Repeat the period to cover the simulation time
%     prn_sequences(i, :) = repmat(prn_period, 1, ceil(N_nyquist / samples_per_period));
%     prn_sequences(i, :) = prn_sequences(i, 1:N_nyquist); % Trim to exact length
% end

% filename = 'mwc_pnr.xlsx';
% xlswrite(filename, prn_sequences);
 prn_sequences = xlsread('mwc_pnr.xlsx');
%% Sensing Matrix Construction
N_fft_recon = N_nyquist; 
freq_bins = linspace(-fs_nyquist/2, fs_nyquist/2-1/T_sim, N_fft_recon);
prn_coeffs = zeros(M, N_fft_recon);
for i = 1:M
    % Take FFT of one period of the PRN sequence
    prn_period_samples = prn_sequences(i, 1:samples_per_period);
    coeffs_temp = fft(prn_period_samples) / samples_per_period;
    prn_full_fft = fft(prn_sequences(i,:))/N_nyquist;
    prn_coeffs(i,:) = fftshift(prn_full_fft); % Store shifted FFT coefficients
end

% Construct the sensing matrix A
% A relates the unknown spectrum X_f (vectorized) to the measurements y_vec
% Size of A: (M * N_mwc_samples_per_channel) x N_fft_recon

% This construction is complex and based on the MWC aliasing model.
% y_i[n] = sum_{l} c_{i,l} * integral{ X(f) * H(f - l*fp) * exp(j*2*pi*f*n*Ts_mwc) df }
% Discretizing this leads to the matrix A.

% --- Simplified Discrete Construction ---
% Assumes LPF is ideal brickwall at fs_mwc/2 = fp
% Assumes perfect synchronization and discretization matching

A = zeros(M * N_mwc_samples_per_channel, N_fft_recon);
H_fft = fftshift(freqz(b_lpf, a_lpf, N_fft_recon, 'whole', fs_nyquist)); % LPF frequency response

mwc_time_vector = (0:N_mwc_samples_per_channel-1) * Ts_mwc;

fprintf('Constructing Sensing Matrix A ... \n');
tic;

% Loop over each frequency bin (potential location of signal component)
for k_freq = 1:N_fft_recon % Index corresponds to frequency freq_bins(k_freq)
    f_k = freq_bins(k_freq); % Current frequency bin

    % Construct the k_freq-th column of A
    column_k = zeros(M * N_mwc_samples_per_channel, 1);
    
    % Simulate the effect of a signal component *only* at f_k on all measurements
    x_k = exp(1j * 2 * pi * f_k * t_nyquist); % Complex sinusoid at f_k
    row_idx = 1;
    for i = 1:M % For each channel
        mixed_k = x_k .* prn_sequences(i, :);
        filtered_k = filter(b_lpf, a_lpf, mixed_k);
        mwc_sample_indices = round(mwc_time_vector / Ts_nyquist) + 1;
        mwc_sample_indices = mwc_sample_indices(mwc_sample_indices <= N_nyquist);
        sampled_k = filtered_k(mwc_sample_indices);

        column_k(row_idx : row_idx + length(sampled_k) - 1) = sampled_k(:);
        row_idx = row_idx + length(sampled_k);
    end
    A(:, k_freq) = column_k;

    % Progress indicator
    % if mod(k_freq, 100) == 0
    %     fprintf('Constructing column %d of %d\n', k_freq, N_fft_recon);
    % end
end
toc;
fprintf('Sensing Matrix A constructed. Size: %d x %d\n', size(A, 1), size(A, 2));

% Normalize columns of A to have unit norm
column_norms = sqrt(sum(abs(A).^2, 1));
column_norms(column_norms == 0) = 1; % Avoid division by zero
A_norm = A ./ repmat(column_norms, size(A, 1), 1);

%% multiple runs for error estimate
rms_error = [];
for SNR_input=70
    error_mat = zeros(1,3);
    for run_idx=1:1
        mwc
        error_mat = [error_mat;[freq_error phase_error amp_error]];
    end
    rms_error = [rms_error;rms(error_mat)];
end

%% Visualization
%figure;
subplot(2, 1, 1);
plot(freq_axis_nyquist*1e-9, 20*log10(abs(X_f)));
title('Original Signal Spectrum (Magnitude)');
xlabel('Frequency (GHz)');
ylabel('|X(f)| (dB)');
xlim([-f_max f_max]*1e-9);
ylim([-100 0]);
grid on;

subplot(2, 1, 2);
plot(freq_bins*1e-9, 20*log10(abs(z_hat)+1e-6)); % Plot reconstructed spectrum magnitude
% stem(freq_bins, abs(z_hat));
title(sprintf('Reconstructed Spectrum (Magnitude) using MWC + OMP (K=%d, M=%d)', K, M));
xlabel('Frequency (GHz)');
ylabel('|Z_{hat}(f)| (dB)');
xlim([-f_max f_max]*1e-9);
ylim([-100 0]);
grid on;

% Mark original frequencies on reconstruction plot
hold on;
plot(f_active*1e-9, 20*log10(max(abs(z_hat))*1e-3) * ones(1, K), 'rv', 'MarkerFaceColor', 'r');
plot(-f_active*1e-9, 20*log10(max(abs(z_hat))*1e-3) * ones(1, K), 'rv', 'MarkerFaceColor', 'r');
legend('Reconstructed Spectrum', 'Original Frequencies');
hold off;
