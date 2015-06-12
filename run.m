clear all
close all
clc


% This script is written to show user how to use the speech word detection
% system using our codes
% If you downloaded the entire project from Github, just run this script as
% a first step to learn how algorithm works.
%
%
% Written by Cagatay Odabasi
% E-Mail:    cagatayodabasi91@gmail.com
%            http://www.cagatayodabasi.com
%            Bogazici Universitysi



% Train the algorithm
train

disp('Training is done. It is time to test...');

%% Test
test_name = './time_data/philip/test ';

% read the the word we desire
[b,Fs ] = audioread('./time_data/philip/mic_1.wav');

% decimate the signal for speed
b = decimate(b,9);

% empty score and Groups array
scores = [];
groups = [];


for i = 1 : 20
    tic
    % read test file
    str =  sprintf([test_name '(%d)' '.wav'], i);

    % classify the signal
    [ Group ] = classify( str, svmStruct,Centroids, parameters);

    groups = [groups; Group];
    
    toc
end

disp('Test is done. Check out the results');

groups
