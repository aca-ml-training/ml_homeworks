from decision_tree import DecisionTree
import numpy as np
from collections import Counter

class RandomForest(object):
    """
    RandomForest a class, that represents Random Forests.

    :param num_trees: Number of trees in the random forest
    :param max_tree_depth: maximum depth for each of the trees in the forest.
    :param ratio_per_tree: ratio of points to use to train each of
        the trees.
    """
    def __init__(self, num_trees = 7, max_tree_depth=5, ratio_per_tree=0.5):
        self.num_trees = num_trees
        self.max_tree_depth = max_tree_depth
        self.ratio_per_tree = ratio_per_tree
        self.trees = None

    def fit(self, X, Y):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array
        """
        self.trees = []
        ind = np.arange(X.shape[0])
        for _ in range(self.num_trees):
            train_ind = np.random.choice(ind, int(X.shape[0]*self.ratio_per_tree), replace=False)
            tree_clf = DecisionTree(self.max_tree_depth)
            tree_clf.fit(X[train_ind], Y[train_ind])
            self.trees.append(tree_clf)
        

    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: (Y, conf), tuple with Y being 1 dimension python
        list with labels, and conf being 1 dimensional list with
        confidences for each of the labels.
        """
        Y_list = []
        for tree in self.trees:
            Y_list.append(tree.predict(X))
        Y = []
        conf = []
        for i in range(len(X)):
            Y_i_counter = Counter(get_columns(Y_list, i))
            y = max(Y_i_counter, key=lambda key: Y_i_counter[key])
            Y.append(y)
            conf.append(Y_i_counter[y]/self.num_trees)
        return (Y, conf)
    
    
def get_columns(list_2D, cols):
    if type(cols) == int or type(cols) == float:
        return [item[cols] for item in list_2D]
    return [[item[col] for col in cols] for item in list_2D]


