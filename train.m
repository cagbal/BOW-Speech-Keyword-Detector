close all
clear all
clc
tic
% Train the speech to text algorithm
%
% Written by Cagatay Odabasi
% E-Mail:    cagatayodabasi91@gmail.com
%            http://www.cagatayodabasi.com
%            Bogazici University
%
% Training data is taken randomly from news websites and they are all
% containing 'banka'. The texts are converted by using google tts api
% (Turkish version).
% Example query:
% http://translate.google.com/translate_tts?tl=tr&q=Banka%20adresleri
%
%
% The script densely divides the data into 25msec sections and extracts the
% features(MFCC). Then,
% cluster them by using K-Means algorithm.

% add path
addpath('./mfcc'); % add mfcc package
addpath('./time_data'); % add data folder

% noise variance
noise_var = 0;

% Parameter of K-Means
% number clusters
k = 150;

% Number of Positive and Negative Training Data for SVM
no_of_negative = 50; %10
no_of_positive = 50; %12

% Name of the negative and positive data, before the number part
%pos_name = 'data_positive_';
%neg_name = 'data_negative_';
%pos_name = 'time';
%neg_name = 'ntime';
pos_name = './time_data/philip/train ';
neg_name = './time_data/philip/ntrain ';

% Word distance threshold
thresh = 15;

% data to construct dictionary
%[y_1, Fs] = audioread('time10.wav');
[y_1, Fs] = audioread('./time_data/philip/train (55).wav');
%[y_2, Fs] = audioread('./time_data/philip/mic_2.wav');
%[y_3, Fs] = audioread('./time_data/philip/train (15).wav');
%[y_4, Fs1] = audioread('./time_data/philip/train (27).wav');
%[y_5, Fs2] = audioread('./time_data/philip/train (37).wav');
%[y_6, Fs3] = audioread('./time_data/philip/train (45).wav');


% decimate the Fs1, Fs2, Fs3
%y_4 = decimate(y_4(:,1),2);
%y_5 = decimate(y_5(:,1),2);
%y_6 = decimate(y_6(:,1),2);


% concatanate all the training data
y = [y_1];

% The number of samples for 25msec
%n = 25e-3*Fs;

%% Extract the MFCC

% Define variables
Tw = 25;                % analysis frame duration (ms)
Ts = 10;                % analysis frame shift (ms)
alpha = 0.97;           % preemphasis coefficient
M = 20;                 % number of filterbank channels
C = 12;                 % number of cepstral coefficients
L = 22;                 % cepstral sine lifter parameter
LF = 100;               % lower frequency limit (Hz)
HF = 3700;              % upper frequency limit (Hz)

% Feature extraction (feature vectors as columns)
[ MFCCs, FBEs, frames ] = ...
    mfcc( y, Fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );



% Generate data needed for plotting
[ Nw, NF ] = size( frames );                % frame length and number of frames
time_frames = [0:NF-1]*Ts*0.001+0.5*Nw/Fs;  % time vector (s) for frames
time = [ 0:length(y)-1 ]/Fs;           % time vector (s) for signal samples
logFBEs = 20*log10( FBEs );                 % compute log FBEs for plotting
logFBEs_floor = max(logFBEs(:))-50;         % get logFBE floor 50 dB below max
logFBEs( logFBEs<logFBEs_floor ) = logFBEs_floor; % limit logFBE dynamic range


% Generate plots
figure('Position', [30 30 800 600], 'PaperPositionMode', 'auto', ...
    'color', 'w', 'PaperOrientation', 'landscape', 'Visible', 'on' );

subplot( 311 );
plot( time, y, 'k' );
xlim( [ min(time_frames) max(time_frames) ] );
xlabel( 'Time (s)' );
ylabel( 'Amplitude' );
title( 'Speech waveform');

subplot( 312 );
imagesc( time_frames, [1:M], logFBEs );
axis( 'xy' );
xlim( [ min(time_frames) max(time_frames) ] );
xlabel( 'Time (s)' );
ylabel( 'Channel index' );
title( 'Log (mel) filterbank energies');

subplot( 313 );
imagesc( time_frames, [1:C], MFCCs(2:end,:) ); % HTK's TARGETKIND: MFCC
%imagesc( time_frames, [1:C+1], MFCCs );       % HTK's TARGETKIND: MFCC_0
axis( 'xy' );
xlim( [ min(time_frames) max(time_frames) ] );
xlabel( 'Time (s)' );
ylabel( 'Cepstrum index' );
title( 'Mel frequency cepstrum' );

% Set color map to grayscale
colormap( 1-colormap('gray') );

% Cluster the MFCC by using K-Means
% number of cluster

% Clear MFCCs from NaN's
MFCCs(:, any(isnan(MFCCs), 1)) = [];

% Remove the first component row from MFCCs
MFCCs(1,:) = [];

% differentiate
dif_1 = [MFCCs(1,:); diff(MFCCs)];

MFCCs = [MFCCs dif_1];

% K-Means
[idx,Centroids]  = kmeans(MFCCs',k);

%% Create Histogram
% To create histogram, I calculate the Euclidian Distance between each
% centroid of cluster and each frame in my image

Centroids = Centroids';

% Calculate the histogram of negative
hist_neg = zeros(k,no_of_negative);
% Train the negative samples
for neg = 1 : no_of_negative
    str =  sprintf([neg_name '(%d)' '.wav'], neg);
    
    [y_neg, Fs] = audioread(str);
    
    % ADD NOISE ---------------------------------------------
    y_neg = y_neg + randn().*sqrt(noise_var);
    % ADD NOISE --------------------------------------------- 
    
    % check if Fs is bigger than 16000, if so decimate it
    if Fs > 16000
        y_neg = decimate(y_neg(:,1), Fs/16000);
        
        Fs = 16000;
    end
    
    
    % Feature extraction (feature vectors as columns)
    [ MFCCs_neg, ~, ~] = ...
        mfcc( y_neg, Fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );
    
    % Remove the first component row from MFCCs
    MFCCs_neg(1,:) = [];
    % Clear MFCCs from NaN's
    MFCCs_neg(:, any(isnan(MFCCs_neg), 1)) = [];
    
    % differentiate
    dif_1 = [MFCCs_neg(1,:); diff(MFCCs_neg)];

    MFCCs_neg = [MFCCs_neg dif_1];
    
    
    for i = 1 : k
        for j = 1 : size(MFCCs_neg,2)
            
            % distance
            dist = norm(MFCCs_neg(:,j) - Centroids(:,i));
            
            % if distance is below than threshold
            if dist < thresh
                hist_neg(i, neg) = hist_neg(i, neg) + 1;
            end
        end
    end
    
    % Normalize the histogram
    normalization_const = sqrt(sum(hist_neg(:,neg).^2));
    
    hist_neg(:, neg) = hist_neg(:, neg)./normalization_const;
    
end

% Calculate the histogram of positive
hist_pos = zeros(k,no_of_positive);
% Train the negative samples
for pos = 1 : no_of_positive
    str =  sprintf([pos_name '(%d)' '.wav'], pos);
    
    [y_pos, Fs] = audioread(str);
    
    % ADD NOISE ---------------------------------------------
    y_pos = y_pos + randn().*sqrt(noise_var);
    % ADD NOISE --------------------------------------------- 
    
    % check if Fs is bigger than 16000, if so decimate it
    if Fs > 16000
        y_pos = decimate(y_pos(:,1), Fs/16000);
        
        Fs = 16000;
    end
    
    
    % Feature extraction (feature vectors as columns)
    [ MFCCs_pos, ~, ~] = ...
        mfcc( y_pos, Fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );
    
    % Remove the first component row from MFCCs
    MFCCs_pos(1,:) = [];
    % Clear MFCCs from NaN's
    MFCCs_pos(:, any(isnan(MFCCs_pos), 1)) = [];
    
    % differentiate
    dif_1 = [MFCCs_pos(1,:); diff(MFCCs_pos)];

    MFCCs_pos = [MFCCs_pos dif_1];
    
    
    for i = 1 : k
        for j = 1 : size(MFCCs_pos,2)
            
            % distance
            dist = norm(MFCCs_pos(:,j) - Centroids(:,i));
            
            % if distance is below than threshold
            if dist < thresh
                hist_pos(i, pos) = hist_pos(i, pos) + 1;
            end
        end
    end
    
    % Normalize the histogram
    normalization_const = sqrt(sum(hist_pos(:,pos).^2));
    
    hist_pos(:, pos) = hist_pos(:, pos)./normalization_const;
    
end

%
% figure
% bar([hist_pos', hist_neg', hist_org'])
% xlim([-1 101])
% title ('Histogram')
% legend('Positive', 'Negative', 'Original')

% Create group variable which defines the labels of features
Group = [ones(1,size(hist_pos,2)), zeros(1,size(hist_neg,2))]';

%% Support Vector Machine
svmStruct = svmtrain([hist_pos, hist_neg]',...
    Group, 'kernel_function', 'rbf', 'rbf_sigma', 50);

% Put all parameters in a struct
parameters.Tw = Tw;
parameters.Ts = Ts;
parameters.alpha = alpha;
parameters.M = M;
parameters.C = C;
parameters.L = L;
parameters.LF = LF;
parameters.HF = HF;
parameters.threshold = thresh;

clearvars -except svmStruct Centroids parameters
toc