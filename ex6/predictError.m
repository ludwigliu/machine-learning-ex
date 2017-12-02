function prediction_error = predictionError(X, y, Xval, yval, C, sigma)
	model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
	predictions = svmPredict(model, Xval);
	prediction_error = mean(double(predictions ~= yval));
end