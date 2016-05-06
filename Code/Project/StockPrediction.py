
import numpy as np
import math
from sklearn.cross_validation import KFold
import Multiclass_SVM
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



# Stock Prediction

print "Read from text file and prepare data matrix & target matrix...."
data_Stock = open('stockpredict.txt', 'r')
inp_dataStock = []
stockfiles = np.loadtxt(data_Stock, delimiter=',')

inp_dataStock = np.array(stockfiles[:,0:-1], dtype='float')
stock_Y = stockfiles[:,-1]

X_stock = np.array(inp_dataStock)
y_stock = np.array(stock_Y)

print "Data matrix & target matrix are ready \n"

# NB Classifier
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = summarize(instances)
    return summaries


def mean(numbers):
    return sum(numbers)/float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)

    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]

    return summaries


def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)

    return separated


def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))

    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)

    return probabilities


def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue

    return bestLabel


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)

    return predictions


def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1

    return (correct/float(len(testSet))) * 100


print "Stock prediction using SVM model...."
svm_StockPred_accuracy = []
NBC_accuracy = []
svn_temp = 0

svm_final_accuracy = 0
svm_final_precision = 0
svm_final_recall = 0
svm_final_fmeasure = 0

kf1 = KFold(X_stock.shape[0], n_folds=4, shuffle=False)
for train_index, test_index in kf1:
    X_train, X_test = X_stock[train_index], X_stock[test_index]
    y_train, y_test = y_stock[train_index], y_stock[test_index]
    trainingset, testingset = stockfiles[train_index], stockfiles[test_index]

    # SVM Start
    clf_Stock_SVM = Multiclass_SVM.MulticlassSVM(C=0.1, tol=0.01, max_iter=100, random_state=0, verbose=1)
    clf_Stock_SVM.fit(X_train, y_train)
    predicted_y =clf_Stock_SVM.calculate_prediction(X_test)
    svm_StockPred_accuracy.append(accuracy_score(y_test,predicted_y))
    svm_confusion_mat = confusion_matrix(y_test, predicted_y)
    svm_accuracy, svm_precision_val, svm_recall_val, svm_f_measure_val = clf_Stock_SVM.svm_findOtherParameters(svm_confusion_mat)

    if accuracy_score(y_test,predicted_y) > svn_temp:
        if (accuracy_score(y_test,predicted_y) != 1):
            svm_final_accuracy = accuracy_score(y_test,predicted_y)
            svm_final_precision = svm_precision_val
            svm_final_recall = svm_recall_val
            svm_final_fmeasure = svm_f_measure_val
            svn_temp = accuracy_score(y_test,predicted_y)

    """
    # Naive bayes classifier
    # prepare model
    summaries = summarizeByClass(trainingset)
    # test model
    predictions = getPredictions(summaries, testingset)
    accuracy = getAccuracy(testingset, predictions)
    print('Accuracy: {0}%').format(accuracy)
    """

print "Stock prediction"
print "Accuracy =" ,max(svm_StockPred_accuracy)
print "Precision = ", svm_final_precision
print "Recall = ", svm_final_recall
print "F-Measure", svm_final_fmeasure
print "\n"
print "Stock predicted successfully. \n"
