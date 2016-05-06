# nD k-Class Gausian Discriminant Analysis
import urllib2

import matplotlib.pyplot as plot
import numpy as np
import math
from math import log
from sklearn import datasets, linear_model
from numpy.linalg import inv
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

"""
# data2 = urllib2.urlopen("http://mlr.cs.umass.edu/ml/machine-learning-databases/iris/iris.data")
# print type(data2)

data2 = open('newfile.txt', 'r')
print type(data2)
# print file.read()

# For Iris data
inp_data2 = []
files = np.loadtxt(data2,dtype=str, delimiter=',')

inp_data2 = np.array(files[:,0:-1], dtype='float')
givenY = files[:,-1]

target2=np.zeros(len(givenY), dtype='int')
unique_y = np.unique(givenY)

for cls in range(len(givenY)):
    for x in range(len(unique_y)):
        if(givenY[cls] == unique_y[x]):
            target2[cls] = x

data_matrix = np.matrix(np.array(inp_data2))
target_matrix = np.array(target2)

print data_matrix
print target_matrix
"""

class NaiveBayesBernoulli():

    def __init__(self):
        print ""

# Calculate gradient alpha
    def membership_function(self, test_set, alpha, priorVal):
        value_1 = []
        value_2 = []
        for x in alpha:
            if x == 0.0 or x == 1.0:
                value_1.append(0.0)
                value_2.append(0.0)
            else:
                value_1.append(log(x))
                value_2.append(log(1 - x))

        gX = [(np.sum(((test_set[:,j][i]*value_1[j]) + ((1-test_set[:,j][i])*value_2[j])) for j in range(test_set.shape[1]-1))
              + math.log(priorVal)) for i in range(test_set.shape[0])]
        return gX

# Calculate prediction
    def discriminant_function(self, max_gX, clas):

        diff = []
        predicted = []
        array = []

        for index in range(len(clas)):
            array.append(max_gX[index])
        predicted_values = np.maximum.reduce(array)

        for x in range(len(predicted_values)):
            for y in max_gX:
                key = max_gX[y]
                if(key[x] == predicted_values[x]):
                    predicted.append(y)
        return predicted

# Find precision recall and F-measure
    def findOtherParameters(self, confusion_mat):

        list_diagonal = np.zeros(confusion_mat.shape[0])
        list_row_sum = np.zeros(confusion_mat.shape[0])
        list_column_sum=np.zeros(confusion_mat.shape[1])

        precision_value = []
        recall_value = []
        f_measure_value = []

        total = np.sum(confusion_mat)
        confuse_diagonal = 0

        for i in range(confusion_mat.shape[0]):
            for j in range(confusion_mat.shape[1]):
                list_row_sum[i] += confusion_mat[i][j]
                list_column_sum[i] += confusion_mat[j][i]
                if(i==j):
                    list_diagonal[i] = confusion_mat[i][j]
                    confuse_diagonal +=  confusion_mat[i][j]
        # print "Accuracy", float(confuse_diagonal)/total
        accuracy = float(confuse_diagonal)/total

        for index in range(len(list_row_sum)):
            if list_row_sum[index]==0:
                precision_value.append(0.0)
            else:
                precision_value.append((float)(list_diagonal[index]) / list_row_sum[index])

            if list_column_sum[index]==0:
                recall_value.append(0)
            else:
                recall_value.append((float)(list_diagonal[index]) / list_column_sum[index])

            if precision_value[index]==0 or recall_value[index]==0:
                f_measure_value.append(0)
            else:
                f_measure_value.append((float) (2 * precision_value[index] * recall_value[index]) / (precision_value[index] + recall_value[index]))

        return accuracy, precision_value, recall_value, f_measure_value

"""
print "\nK-Fold started "
max_gX = {}

maximum_gX = []
kf = KFold(data_matrix.shape[0], n_folds=10, shuffle=True)
print "No of folds = ",len(kf)

temp = 0
final_precision=0
final_recall = 0
final_fmeasure = 0
final_accuracy = 0

for train_index, test_index in kf:
    print kf
    X_Train_Data, X_Test_Data = data_matrix[train_index], data_matrix[test_index]
    Y_Train_Data, Y_Test_Data = target_matrix[train_index], target_matrix[test_index]

    # iterate data for each class
    for clas in np.unique(target_matrix):
        class_feature_matrix = X_Train_Data[Y_Train_Data==clas]
        print "Class=", clas, " length = ", len(class_feature_matrix)
        prior_array = len(class_feature_matrix)*1.0/len(X_Train_Data)
        print prior_array
        alpha = [(np.sum(class_feature_matrix[:,i])/len(class_feature_matrix)) for i in range(class_feature_matrix.shape[1])]
        gX = membership_function(X_Test_Data, alpha, prior_array)
        max_gX.update({int(clas): gX})

    # find discriminanent function
    disc_function = discriminant_function(max_gX, np.unique(target_matrix))
    print disc_function
    confusion_mat = confusion_matrix(Y_Test_Data, disc_function)
    print confusion_mat

    # find precision, recall , f-measure
    accuracy, precision_val, recall_val, f_measure_val = findOtherParameters(confusion_mat)

    if accuracy > temp:
        if (accuracy != 1):
            final_accuracy = accuracy
            final_precision = precision_val
            final_recall = recall_val
            final_fmeasure = f_measure_val
            temp = accuracy

    print "\n K-fold completed \n"

# print errors
print "Final Output : "
print "Accuracy = ", final_accuracy
print "Precision = ", final_precision
print "Recall = ", final_recall
print "F-Measure", final_fmeasure
print "\n Fold Completed \n"
"""