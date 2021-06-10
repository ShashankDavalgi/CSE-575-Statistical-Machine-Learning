#!/usr/bin/env python
# coding: utf-8

import scipy.io
import statistics
import scipy.stats
import numpy as np

# Load the data

data = scipy.io.loadmat('D:\\Spring 21\\Statistical Machine Learning\\ProjectMilestone1\\fashion_mnist.mat')

#Splitting the data as training and testing

training_data_array = data['trX']
# print(len(training_data_array))
# print(len(training_data_array[0]))

training_label_array = data['trY']
# print(len(training_label_array))
# print(len(training_label_array[0]))

testing_data_array = data['tsX']
# print(len(testing_data_array))
# print(len(testing_data_array[0]))

testing_label_array = data['tsY']
# print(len(testing_label_array))
# print(len(testing_label_array[0]))

# Calculating the average and standard deviation for each image based on the pixel array for the training data

average = []
standard_deviation = []

for i in range(len(training_data_array)):
    average.append(sum(training_data_array[i])/len(training_data_array[0]))
    standard_deviation.append(statistics.stdev(training_data_array[i]))
# print(len(average))
# print(len(standard_deviation))

# Calculating the average and standard deviation for each image based on the pixel array for the testing data

testing_x1_average = []
testing_x2_std_dev = []

for i in range(len(testing_data_array)):
    testing_x1_average.append(sum(testing_data_array[i])/len(testing_data_array[0]))
    testing_x2_std_dev.append(statistics.stdev(testing_data_array[i]))
# print(len(testing_x1_average))
# print(len(testing_x2_std_dev))

# Creating X_train and y_train data

nd_list = [average, standard_deviation]

X = np.asarray(nd_list)
# print(X)

y = training_label_array
# print(type(y))

test_nd_list = [testing_x1_average, testing_x2_std_dev]

X_test = np.asarray(test_nd_list)

# Function to calculate the Accuracy

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# Function to calculate the sigmoid value

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Prediction Method

def predict(X):
    linear_model = np.dot(X, weights) + bias
    y_predicted = sigmoid(linear_model)
    
    y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
    return y_predicted_class

# gradient ascent model to fit the data

def fit(X, y):
    n_samples = len(average)
    learning_rate = 0.1
    epoch_iters = 1000
    weights = np.zeros(2)
    bias = 0


    for _ in range(epoch_iters):
        linear_model = np.dot(X, weights) + bias
        y_predicted = sigmoid(linear_model)

        dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
        db = (1 / n_samples) * np.sum(y_predicted - y)

        weights -= learning_rate * dw
        bias -= learning_rate * db


fit(X, y)
predictions = predict(X_test)

print("Accuracy:", accuracy(y_test, predictions))

