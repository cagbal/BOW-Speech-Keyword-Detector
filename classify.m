function [ Group ] = classify( audio_file_name, svmStruct, Centroids,...
    parameters)
%classify function that tries to detect whether a word is included in given
% audio file or not 
%   This script uses bag of words model with a combination of MFCC. 
% Inputs: 
% audio_file_name = (string) name of the mp3 file
% svmStruct = (struct) returned by train script SVM info
% Centroids = (matrix) returned by train script Kmeans centroids
% parameters = (struct) returned by train script some required parameters
% Output:
% Group: (scalar) label of input data
%
% Written by Cagatay Odabasi
% E-Mail:    cagatayodabasi91@gmail.com
%            http://www.cagatayodabasi.com
%            Bogazici University
tic
% get the dictionary size
k = size(Centroids,2);
 
% initialize the histogram of test vector
hist_test = zeros(k,1);

[y_test, Fs] = audioread(audio_file_name);

% check if Fs is bigger than 16000, if so decimate it 
if Fs > 16000
    y_test = decimate(y_test(:,1), Fs/16000);
    
    Fs = 16000;
end

% Define variables
Tw = parameters.Tw;                % analysis frame duration (ms)
Ts = parameters.Ts;                % analysis frame shift (ms)
alpha = parameters.alpha;           % preemphasis coefficient
M = parameters.M;                 % number of filterbank channels
C = parameters.C;                 % number of cepstral coefficients
L = parameters.L;                 % cepstral sine lifter parameter
LF = parameters.LF;               % lower frequency limit (Hz)
HF = parameters.HF;              % upper frequency limit (Hz)
thresh = parameters.threshold;


% Feature extraction (feature vectors as columns)
[ MFCCs_test, ~, ~] = ...
    mfcc( y_test(:,1), Fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );

% Remove the first component row from MFCCs
MFCCs_test(1,:) = [];
% Clear MFCCs from NaN's
MFCCs_test(:, any(isnan(MFCCs_test), 1)) = [];

% differentiate
dif_1 = [MFCCs_test(1,:); diff(MFCCs_test)];

MFCCs_test = [MFCCs_test dif_1];


for i = 1 : k
    for j = 1 : size(MFCCs_test,2)
        
        % distance
        dist = norm(MFCCs_test(:,j) - Centroids(:,i));
        
        % if distance is below than threshold
        if dist < thresh
            hist_test(i, 1) = hist_test(i, 1) + 1;
        end
    end
end

% Normalize the histogram
normalization_const = sqrt(sum(hist_test(:,1).^2));

hist_test(:, 1) = hist_test(:, 1)./normalization_const;

% Classify 
Group = svmclassify(svmStruct,hist_test');

toc
end

