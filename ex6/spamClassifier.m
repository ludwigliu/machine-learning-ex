%% Machine Learning Online Class
%  Exercise 6 | Spam Classification with SVMs
%  
%  ------------------------------------------
% The exercise use a linear kernel instead of guassian kernel 
% to train SVM, why not? so i wrote this to train SVM with 
% gaussian kernel and check the accuracy. 

fprintf('\nPreprocessing sample email (pamTrain.mat)\n');

% Load the Spam Email dataset
% You will have X, y in your environment
load('spamTrain.mat');

fprintf('\nTraining SVM with RBF Kernel (Spam Classification)\n');

% needs to find the best values 
% here just use a initial value to test first
C = 1;
sigma = 0.3;

model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
p = svmPredict(model, X);

fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);

% Load the test dataset
% You will have Xtest, ytest in your environment
load('spamTest.mat');

fprintf('\nEvaluating the trained Linear SVM on a test set ...\n')

p = svmPredict(model, Xtest);

fprintf('Test Accuracy: %f\n', mean(double(p == ytest)) * 100);
pause;