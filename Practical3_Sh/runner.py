
import numpy as np
import matplotlib.pyplot as plt
from decision_tree import DecisionTree
from random_forest import RandomForest


def accuracy_score(Y_true, Y_predict):
    
    """
    accuracy_score a function that returns the accuracy of the data
        
    :param Y_true: 1 dimensional python list or numpy 1 dimensional array with true labels
    :param Y_predict: 1 dimensional python list or numpy 1 dimensional array with predicted labels     
        
    """
    
    true = 0

    for i in range(len(Y_predict)):
        if Y_true[i] == Y_predict[i]:
            true+=1
    acc = true/float(len(Y_true))* 100.0
    return acc


def evaluate_performance():
    
    '''
    evaluate_performance a function that evaluates the performance of decision trees and logistic regression,
    averages over 1,000 trials of 10-fold cross validation

    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of logistic regression
      stats[1,1] = std deviation of logistic regression accuracy
      
    '''

    # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n, d = X.shape

    
    accuracies_tree = []
    accuracies_forest = []
    #accuracies_logistic = []
    for trial in range(1000):
        if trial%100 == 0:
            print(trial)
        # TODO: shuffle for each of the trials.
        # the following code is for reference only.
        idx = np.arange(n)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # TODO: write your own code to split data (for cross validation)
        # the code here is for your reference.
        Xtrain = X[0:100, :]  # train on first 100 instances
        Xtest = X[100:, :]
        ytrain = y[0:100, :]  # test on remaining instances
        ytest = y[100:, :]

        
        train = (np.hstack((Xtrain, ytrain))).tolist()
        # train the decision tree
        classifier = DecisionTree(100)
        tree = classifier.fit(train)

        # output predictions on the remaining data
        y_pred_tree = classifier.predict(Xtest, tree, [])
        accuracy_tree = accuracy_score(ytest, y_pred_tree)
        accuracies_tree.append(accuracy_tree)

        clt = RandomForest(10, 100)
        forest = clt.fit(Xtrain, ytrain)
                    
        y_pred_forest, conf = clt.predict(Xtest, forest)
        accuracy_forest = accuracy_score(ytest, y_pred_forest)
        accuracies_forest.append(accuracy_forest)
        


    # compute the training accuracy of the model
    meanDecisionTreeAccuracy = np.mean(accuracies_tree)
    stddevDecisionTreeAccuracy = np.std(accuracies_tree)
    #meanLogisticRegressionAccuracy = 0
    #stddevLogisticRegressionAccuracy = 0
    meanRandomForestAccuracy =  np.mean(accuracies_forest)
    stddevRandomForestAccuracy = np.std(accuracies_forest)

    # make certain that the return value matches the API specification
    stats = np.zeros((2, 2))
    stats[0, 0] = meanDecisionTreeAccuracy
    stats[0, 1] = stddevDecisionTreeAccuracy
    stats[1, 0] = meanRandomForestAccuracy
    stats[1, 1] = stddevRandomForestAccuracy
    #stats[2, 0] = meanLogisticRegressionAccuracy
    #stats[2, 1] = stddevLogisticRegressionAccuracy
    return stats


# Do not modify from HERE...
if __name__ == "__main__":
    stats = evaluate_performance()
    print ("Decision Tree Accuracy = ", stats[0, 0], " (", stats[0, 1], ")")
    print ("Random Forest Tree Accuracy = ", stats[1, 0], " (", stats[1, 1], ")")
    #print "Logistic Reg. Accuracy = ", stats[2, 0], " (", stats[2, 1], ")"
# ...to HERE.