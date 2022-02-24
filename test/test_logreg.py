"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""

import numpy as np
import pandas as pd
import random 
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler

X_train, X_val, y_train, y_val = utils.loadDataset(features=['Penicillin V Potassium 500 MG', 'Computed tomography of chest and abdomen', 
                                    'Plain chest X-ray (procedure)',  'Low Density Lipoprotein Cholesterol', 
                                    'Creatinine', 'AGE_DIAGNOSIS'], split_percent=0.8, split_state=42)
# Setting the seed
np.random.seed(10)

def test_updates():
	"""
	TODO:
	- Check that your gradient is being calculated correctly
	- Check that your loss function is correct and that you have reasonable losses at the end of training
	"""

	log_model_updates = logreg.LogisticRegression(num_feats=6, max_iter=10000, tol=0.001, learning_rate=0.0001, batch_size=12)
	log_model_updates.train_model(X_train, y_train, X_val, y_val)

	# Check gradient calculation
	gradient_values = log_model_updates.calculate_gradient(X_train, y_train)

	assert len(gradient_values) == X_train.shape[1]

	# Check for reasonable losses
	train_losses = log_model_updates.loss_function(X_train, y_train)
	val_losses = log_model_updates.loss_function(X_val, y_val)

	loss_difference = train_losses - val_losses

	assert abs(loss_difference) < 0.2

def test_predict():
	"""
	TODO:
	- Check that self.W is being updated as expected and produces reasonable estimates for NSCLC classification.
	- Check accuracy of model after training.
	"""
	log_model_predict = logreg.LogisticRegression(num_feats=6, max_iter=10000, tol=0.001, learning_rate=0.0001, batch_size=12)

	# checking that self.W is being updated
	pre_training_W = log_model_predict.W 
	
	log_model_predict.train_model(X_train, y_train, X_val, y_val)

	post_training_W = log_model_predict.W 

	assert sum(pre_training_W) != sum(post_training_W)

	# checking accuracy
	train_accuracy = log_model_predict.calculate_r2(X_train, y_train)
	val_accuracy = log_model_predict.calculate_r2(X_val, y_val)

	accuracy_difference = abs(train_accuracy-val_accuracy)

	assert accuracy_difference < 0.05
	