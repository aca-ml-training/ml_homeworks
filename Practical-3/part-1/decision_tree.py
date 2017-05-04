import decision_tree_helper
from decision_tree_helper import build_tree

class DecisionTree(object):
    """
    DecisionTree class, that represents one Decision Tree

    :param max_tree_depth: maximum depth for this tree.
    """
    def __init__(self, max_tree_depth):
        self.max_depth = max_tree_depth

    def fit(self, X, Y):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array
        """
        data = X.tolist()
        for index in range(len(X)):
            data[index].append(Y[index])
        self.tree = build_tree(data, max_depth = self.max_depth)

    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: Y - 1 dimension python list with labels
        """
        X = X.tolist()
        Y = []
        for row in X:
            current_node = self.tree     
            while(current_node != None and not current_node.is_leaf):
                feature_val = row[current_node.column] 
                if type(feature_val) == int or type(feature_val) == float:
                    if feature_val >= current_node.value:
                        current_node = current_node.true_branch
                    else: 
                        current_node = current_node.false_branch
                if type(feature_val) == str:
                    if feature_val == current_node.value:
                        current_node = current_node.true_branch
                    else:
                        current_node = current_node.false_branch
            Y.append(current_node.result)
        return Y
