import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression
from decision_tree import DecisionTree
from random_forest import RandomForest

def accuracy_score(Y_true, Y_predict):
    tp_plus_tn = 0
    N = len(Y_true)
    for index in range(N):
        if Y_true[index] == Y_predict[index]:
            tp_plus_tn += 1
    return tp_plus_tn / N


def evaluate_performance():
    '''
    Evaluate the performance of decision trees and logistic regression,
    average over 1,000 trials of 10-fold cross validation

    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of logistic regression
      stats[1,1] = std deviation of logistic regression accuracy

    ** Note that your implementation must follow this API**
    '''

    # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array(data[:, 0])
    n, d = X.shape

    all_accuracies_dt = []
    all_accuracies_lr = []
    all_accuracies_rf = []
    for trial in range(1):
        idx = np.arange(n)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]
        
        ind = np.arange(X.shape[0])
        classifier_dt = DecisionTree(9)
        classifier_lr = LogisticRegression(max_steps=10000, epsilon=1e-6, step_size = 1, l=3)
        classifier_rf = RandomForest(ratio_per_tree=0.5, num_trees = 100, max_tree_depth=8)
        scores_dt = []
        scores_lr = []
        scores_rf = []
        for i in range(10):
            test_ind = np.random.choice(ind, int(X.shape[0]/10), replace=False)
            ind = np.setdiff1d(np.arange(X.shape[0]), test_ind)
            X_train, Y_train, X_test, Y_test = X[ind], y[ind], X[test_ind], y[test_ind]
            # train the decision tree
            classifier_dt.fit(X_train, Y_train)
            accuracy_dt = accuracy_score(Y_true=Y_test, Y_predict=classifier_dt.predict(X_test))
            scores_dt.append(accuracy_dt)
            # train the logistic regression
            classifier_lr.fit(np.hstack((np.ones(len(X_train)).reshape(len(X_train), 1), X_train)), Y_train)
            accuracy_lr = accuracy_score(Y_true=Y_test, Y_predict=classifier_lr.predict( np.hstack((np.ones(len(X_test)).reshape(len(X_test), 1), X_test))))
            scores_lr.append(accuracy_lr)
            # train the random forest
            classifier_rf.fit(X_train, Y_train)
            accuracy_rf = accuracy_score(Y_true=Y_test, Y_predict=classifier_rf.predict(X_test)[0])
            scores_rf.append(accuracy_rf)
        all_accuracies_dt.append(np.mean(scores_dt))
        all_accuracies_lr.append(np.mean(scores_lr))
        all_accuracies_rf.append(np.mean(scores_rf))
       


    # compute the training accuracy of the model
    meanDecisionTreeAccuracy = np.mean(all_accuracies_dt)
    stddevDecisionTreeAccuracy = np.std(all_accuracies_dt)
    meanLogisticRegressionAccuracy = np.mean(all_accuracies_lr)
    stddevLogisticRegressionAccuracy = np.std(all_accuracies_lr)
    meanRandomForestAccuracy = np.mean(all_accuracies_rf)
    stddevRandomForestAccuracy = np.std(all_accuracies_rf)

    # make certain that the return value matches the API specification
    stats = np.zeros((3, 2))
    stats[0, 0] = meanDecisionTreeAccuracy
    stats[0, 1] = stddevDecisionTreeAccuracy
    stats[1, 0] = meanRandomForestAccuracy
    stats[1, 1] = stddevRandomForestAccuracy
    stats[2, 0] = meanLogisticRegressionAccuracy
    stats[2, 1] = stddevLogisticRegressionAccuracy
    return stats


# Do not modify from HERE...
if __name__ == "__main__":
    stats = evaluate_performance()
    print("Decision Tree Accuracy = ", stats[0, 0], " (", stats[0, 1], ")")
    print("Random Forest Tree Accuracy = ", stats[1, 0], " (", stats[1, 1], ")")
    print("Logistic Reg. Accuracy = ", stats[2, 0], " (", stats[2, 1], ")")
# ...to HERE.
