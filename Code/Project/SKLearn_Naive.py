from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.naive_bayes import BernoulliNB

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC


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

X = np.array(inp_data2)
y = np.array(target2)

svm_accuracy = []
NB_accuracy = []
svm_SVCAccuracy = []
print "\n K-Fold"
kf = KFold(X.shape[0], n_folds= 10, shuffle=False)
print len(kf)
for train_index, test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #clf = BernoulliNB()
    clf = BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)

    clf.fit(X_train, y_train)

    predicted_y = clf.predict(X_test)

    #print "Bernoulli NB =" ,accuracy_score(y_test,predicted_y)
    NB_accuracy.append(accuracy_score(y_test,predicted_y))
    # print clf.score(X_test, y_test)

    clf_svm = OneVsRestClassifier(LinearSVC(random_state=0))
    clf_svm.fit(X_train,y_train)

    predicted_svmy = clf_svm.predict(X_test)
    svm_accuracy.append(accuracy_score(y_test,predicted_svmy))
    # print "Multiclass SVM =" ,accuracy_score(y_test,predicted_svmy)

    clf_svc = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)\

    clf_svc.fit(X_train, y_train)
    predicted_svc = clf_svc.predict(X_test)
    svm_SVCAccuracy.append(accuracy_score(y_test,predicted_svc))
    # print "SVC =" ,accuracy_score(y_test,predicted_svc)

print "Bernoulli NB =" ,max(NB_accuracy)
print "SVM =",max(svm_accuracy)
print "SVC =" ,max(svm_SVCAccuracy)



