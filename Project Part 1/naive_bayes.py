#!/usr/bin/env python
# coding: utf-8


import scipy.io
import statistics
import scipy.stats

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

# Calculating the average and standard deviation for each image based on the pixel array for the testing data

testing_x1_average = []
testing_x2_std_dev = []

for i in range(len(testing_data_array)):
    testing_x1_average.append(sum(testing_data_array[i])/len(testing_data_array[0]))
    testing_x2_std_dev.append(statistics.stdev(testing_data_array[i]))

# X1 i.e. mean values of training data for each class

training_x1_dict = {}

for i in range(len(average)):
    class_label = int(training_label_array[0][i])
    if class_label not in training_x1_dict:
        training_x1_dict[class_label] = list()
    training_x1_dict[class_label].append(average[i])
    
# X2 i.e. mean values of training data for each class 
    
training_x2_dict = {}

for i in range(len(average)):
    class_label = int(training_label_array[0][i])
    if class_label not in training_x2_dict:
        training_x2_dict[class_label] = list()
    training_x2_dict[class_label].append(standard_deviation[i])
    
# X1 i.e. standard deviation values of testing data for each class
    
testing_x1_dict = {}

for i in range(len(testing_x1_average)):
    class_label = int(testing_label_array[0][i])
    if class_label not in testing_x1_dict:
        testing_x1_dict[class_label] = list()
    testing_x1_dict[class_label].append(testing_x1_average[i])
    
# X2 i.e. standard devation values of testing data for each class

testing_x2_dict = {}

for i in range(len(testing_x1_average)):
    class_label = int(testing_label_array[0][i])
    if class_label not in testing_x2_dict:
        testing_x2_dict[class_label] = list()
    testing_x2_dict[class_label].append(testing_x2_std_dev[i])
    

list_x1_class_0 = training_x1_dict[0]
list_x1_class_1 = training_x1_dict[1]

list_x2_class_0 = training_x2_dict[0]
list_x2_class_1 = training_x2_dict[1]

testing_list_x1_class_0 = testing_x1_dict[0]
testing_list_x1_class_1 = testing_x1_dict[1]

testing_list_x2_class_0 = testing_x2_dict[0]
testing_list_x2_class_1 = testing_x2_dict[1]

# Mean and Standard deviation for each variables x1 and x2 for both classes 0 and 1:

mean_x1_class_0 = sum(list_x1_class_0)/len(list_x1_class_0)
print("Mean of X1 for class 0: ", mean_x1_class_0)
std_x1_class_0 = statistics.stdev(list_x1_class_0)
print("Standard Deviation of X1 for Class 0: ", std_x1_class_0)
mean_x1_class_1 = sum(list_x1_class_1)/len(list_x1_class_1)
print("Mean of X1 for class 1: ", mean_x1_class_1)
std_x1_class_1 = statistics.stdev(list_x1_class_1)
print("Standard Deviation of X1 for Class 1: ", std_x1_class_1)
mean_x2_class_0 = sum(list_x2_class_0)/len(list_x2_class_0)
print("Mean of X2 for class 0: ", mean_x2_class_0)
std_x2_class_0 = statistics.stdev(list_x2_class_0)
print("Standard Deviation of X2 for Class 0: ", std_x2_class_0)
mean_x2_class_1 = sum(list_x2_class_1)/len(list_x2_class_1)
print("Mean of X2 for class 1: ", mean_x2_class_1)
std_x2_class_1 = statistics.stdev(list_x2_class_1)
print("Standard Deviation of X2 for Class 1: ", std_x2_class_1)

# Gaussian Distribution for x1 and x2 for both the classes 0 and 1:

gaussian_dist_x1_class_0 = scipy.stats.norm(mean_x1_class_0, std_x1_class_0)
gaussian_dist_x1_class_1 = scipy.stats.norm(mean_x1_class_1, std_x1_class_1)
gaussian_dist_x2_class_0 = scipy.stats.norm(mean_x2_class_0, std_x2_class_0)
gaussian_dist_x2_class_1 = scipy.stats.norm(mean_x2_class_1, std_x2_class_1)


# Naive Bayes calculation using Gaussian distribution

prob_class_0 = 0.5
prob_class_1 = 0.5

predicted_list_class_0 = []
predicted_list_class_1 = []

for i in range(len(testing_list_x1_class_0)):
    prob_class_0_x1_x2 = gaussian_dist_x1_class_0.pdf(testing_list_x1_class_0[i]) * gaussian_dist_x2_class_0.pdf(testing_list_x2_class_0[i]) * prob_class_0
#     print("Prob of 0 given x1= ", testing_list_x1_class_0[i], " x2= ", testing_list_x2_class_0[i], " is ", prob_class_0_x1_x2)
    prob_class_1_x1_x2 = gaussian_dist_x1_class_1.pdf(testing_list_x1_class_0[i]) * gaussian_dist_x2_class_1.pdf(testing_list_x2_class_0[i]) * prob_class_1
#     print("Prob of 1 given x1= ", testing_list_x1_class_0[i], " x2= ", testing_list_x2_class_0[i], " is ", prob_class_1_x1_x2)
    
    if(prob_class_0_x1_x2 > prob_class_1_x1_x2):
        predicted_list_class_0.append(0)
    else:
        predicted_list_class_0.append(1)
        

for i in range(len(testing_list_x1_class_1)):
    prob_class_0_x1_x2_class_1_data = gaussian_dist_x1_class_0.pdf(testing_list_x1_class_1[i]) * gaussian_dist_x2_class_0.pdf(testing_list_x2_class_1[i]) * prob_class_0
#     print("Prob of 0 given x1= ", testing_list_x1_class_1[i], " x2= ", testing_list_x2_class_1[i], " is ", prob_class_0_x1_x2_class_1_data)
    prob_class_1_x1_x2_class_1_data = gaussian_dist_x1_class_1.pdf(testing_list_x1_class_1[i]) * gaussian_dist_x2_class_1.pdf(testing_list_x2_class_1[i]) * prob_class_1
#     print("Prob of 1 given x1= ", testing_list_x1_class_1[i], " x2= ", testing_list_x2_class_1[i], " is ", prob_class_1_x1_x2_class_1_data)
    
    if(prob_class_0_x1_x2_class_1_data < prob_class_1_x1_x2_class_1_data):
        predicted_list_class_1.append(1)
    else:
        predicted_list_class_1.append(0)

# Predicting the labels and calculating the Accuracy

count_0 = 0
count_1 = 0
for i in range(len(predicted_list_class_0)):
    if predicted_list_class_0[i] == 0:
        count_0 += 1
    if predicted_list_class_1[i] == 1:
        count_1 += 1

# print(count_1)
print("Accuracy for the naive bayes algorithm: ", ((count_0+count_1)/len(testing_data_array)) * 100, "%")

