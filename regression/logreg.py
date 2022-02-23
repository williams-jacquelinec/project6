# importing dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BaseRegressor():
    def __init__(self, num_feats, learning_rate=0.1, tol=0.001, max_iter=100, batch_size=12):
        # initializing parameters
        self.W = np.random.randn(num_feats + 1).flatten()
        # assigning hyperparameters
        self.lr = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.num_feats = num_feats
        # defining list for storing loss history
        self.loss_history_train = []
        self.loss_history_val = []
        
    def calculate_gradient(self, X, y):
        pass
    
    def loss_function(self, y_true, y_pred):
        pass
    
    def make_prediction(self, X):
        pass
    
    def train_model(self, X_train, y_train, X_val, y_val):
        # Padding data with vector of ones for bias term
        X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
        X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
        # Defining intitial values for while loop
        prev_update_size = 1
        iteration = 1
        # Gradient descent
        while prev_update_size > self.tol and iteration < self.max_iter:
            # Shuffling the training data for each epoch of training
            shuffle_arr = np.concatenate([X_train, np.expand_dims(y_train, 1)], axis=1)
            # In place shuffle
            np.random.shuffle(shuffle_arr)
            X_train = shuffle_arr[:, :-1]
            y_train = shuffle_arr[:, -1].flatten()
            num_batches = int(X_train.shape[0]/self.batch_size) + 1
            X_batch = np.array_split(X_train, num_batches)
            y_batch = np.array_split(y_train, num_batches)
            # Generating list to save the param updates per batch
            update_size_epoch = []
            # Iterating through batches (full for loop is one epoch of training)
            for X_train, y_train in zip(X_batch, y_batch):
                # Making prediction on batch
                y_pred = self.make_prediction(X_train)
                # Calculating loss
                loss_train = self.loss_function(X_train, y_train)
                # Adding current loss to loss history record
                self.loss_history_train.append(loss_train)
                # Storing previous weights and bias
                prev_W = self.W
                # Calculating gradient of loss function with respect to each parameter
                grad = self.calculate_gradient(X_train, y_train)
                # Updating parameters
                new_W = prev_W - self.lr * grad 
                self.W = new_W
                # Saving step size
                update_size_epoch.append(np.abs(new_W - prev_W))
                # Validation pass
                loss_val = self.loss_function(X_val, y_val)
                self.loss_history_val.append(loss_val)
            # Defining step size as the average over the past epoch
            prev_update_size = np.mean(np.array(update_size_epoch))
            # Updating iteration number
            iteration += 1
    
    def plot_loss_history(self):
        """
        Plots the loss history after training is complete.
        """
        loss_hist = self.loss_history_train
        loss_hist_val = self.loss_history_val
        assert len(loss_hist) > 0, "Need to run training before plotting loss history"
        fig, axs = plt.subplots(2, figsize=(8,8))
        fig.suptitle('Loss History')
        axs[0].plot(np.arange(len(loss_hist)), loss_hist)
        axs[0].set_title('Training Loss')
        axs[1].plot(np.arange(len(loss_hist_val)), loss_hist_val)
        axs[1].set_title('Validation Loss')
        plt.xlabel('Steps')
        axs[0].set_ylabel('Train Loss')
        axs[1].set_ylabel('Val Loss')
        fig.tight_layout()
        

# import required modules
class LogisticRegression(BaseRegressor):
    def __init__(self, num_feats, learning_rate=0.1, tol=0.0001, max_iter=100, batch_size=12):
        super().__init__(num_feats, learning_rate, tol, max_iter, batch_size)
        
    def calculate_gradient(self, X, y): #-> np.ndarray:
        """
        TODO: write function to calculate gradient of the
        logistic loss function to update the weights 

        Params:
            X (np.ndarray): feature values
            y (np.array): labels corresponding to X

        Returns: 
            gradients for given loss function (np.ndarray)
        """
        self.X = X 
        self.y = y 

        y_pred = self.make_prediction(X)
        m = len(y)
        error = y - y_pred 

        # gradient of loss weights
        weight_grad = (1/m) * X.T.dot(error)

        other_grad = (1/m) * X.T.dot(error)

        # print('here is weight_grad')
        # print(weight_grad)
        # print('here is other_grad')
        # print(other_grad)
        # print('')

        return weight_grad
    
    def loss_function(self, X, y): #-> float:
        """
        TODO: get y_pred from input X and implement binary cross 
        entropy loss function. Binary cross entropy loss assumes that 
        the classification is either 1 or 0, not continuous, making
        it more suited for (binary) classification.

        Params:
            X (np.ndarray): feature values
            y (np.array): labels corresponding to X

        Returns: 
            average loss 
        """
        self.X = X 
        self.y = y 

        y_pred = self.make_prediction(X)

        y_pred[y_pred == 1] = 0.9999
        y_pred[y_pred == 0] = 0.0001

        bce_loss = -np.mean(y * (np.log(y_pred)) - (1 - y) * np.log(1 - y_pred))

        return bce_loss
    
    def make_prediction(self, X): #-> np.array:
        """
        TODO: implement logistic function to get estimates (y_pred) for input
        X values. The logistic function is a transformation of the linear model W.T(X)+b 
        into an "S-shaped" curve that can be used for binary classification

        Params: 
            X (np.ndarray): Set of feature values to make predictions for

        Returns: 
            y_pred for given X
        """

        # pass
        self.X = X 

        # adding bias term to X matrix if not already present
        if X.shape[1] == self.num_feats:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        y_pred = self.sigmoid(X.dot(self.W).flatten())

        return y_pred 

    def sigmoid(self, z):
        """
        Sigmoid function: converts input z (a real number) into a value between 0 and 1.
        """
        self.z = z 

        return 1.0/(1+np.exp(z)) 

    def calculate_r2(self, X, y):
        self.X = X 
        self.y = y 

        y_pred = self.make_prediction(X)
        rss = ((y - y_pred)**2).sum()
        # print("here is rss")
        # print(rss)
        tss = ((y - y.mean())**2).sum()
        # print("here is tss")
        # print(tss)
        # print("here is rss/tss")
        # print(rss/tss)
        r2 = 1 - (rss/tss)

        # print("here is the accuracy value")
        return abs(r2)
    
        

# import random 
# np.random.seed(10)

# from utils import loadDataset  
# from sklearn.preprocessing import StandardScaler

# x_mat, y_mat = loadDataset()
#print(x_mat.shape)

# calculate_gradient returns gradients for a given loss function (# of gradients = num.feats)
# print(LogisticRegression(num_feats = 6).calculate_gradient(x_mat, y_mat))

# loss_function returns average loss (1 number)
# print(LogisticRegression(num_feats = 6).loss_function(x_mat, y_mat))


# X_train, X_val, y_train, y_val = loadDataset(features=['Penicillin V Potassium 500 MG', 'Computed tomography of chest and abdomen', 
#                                 'Plain chest X-ray (procedure)',  'Low Density Lipoprotein Cholesterol', 
#                                 'Creatinine', 'AGE_DIAGNOSIS'], split_percent=0.85, split_state=42)

# # scale data since values vary across features
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_val = sc.transform (X_val)
# print("here is the shape of X (train & val) and the shape of y (train & val)")
# print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

# log_model = LogisticRegression(num_feats=6, max_iter=10000, tol=0.001, learning_rate=0.00001, batch_size=800)
# print("here is printing log_model.W before training")
# print(log_model.W)
# log_model.train_model(X_train, y_train, X_val, y_val)
# # What is self.W? How do I check it as it updates?
# print("here is printing log_model.W after training?")
# print(log_model.W)

# # calculating the gradient values
# print("here are the gradient values")
# print(log_model.calculate_gradient(X_train, y_train))
# print("length of gradient values =", len(log_model.calculate_gradient(X_train, y_train)))



# # Checking how the loss function looks
# print("here are loss function values with training data and validation data")
# print(log_model.loss_function(X_train, y_train))
# print(log_model.loss_function(X_val, y_val))

# loss_difference = log_model.loss_function(X_train, y_train) - log_model.loss_function(X_val, y_val)
# print("here is the difference in losses")
# print(loss_difference)

# # calculating accuracy
# print('here is accuracy of training and validation data')
# print(log_model.calculate_r2(X_train, y_train))
# print(log_model.calculate_r2(X_val, y_val))

# print("difference in accuracy values")
# print(log_model.calculate_r2(X_train, y_train) - log_model.calculate_r2(X_val, y_val))

# log_model.plot_loss_history()




















        # gradient of loss bias
        #bias_grad = (1/m) * np.sum(error)



        # grad = - X.T.dot(error)
        # print("this is what the linear regression grad formula gets:", grad)


        # print('')
        # print('testing the make_prediction formula')
        # print('here is the y_pred value')
        # print(y_pred)
        # print('and here is the length of y_pred')
        # print(len(y_pred))
        # print('')
        # print('here is what self.W is')
        # print(self.W)

                # print(X.shape)

        # print("here is self.W I think")
        # print(self.W)