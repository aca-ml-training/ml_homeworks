import numpy as np
from collections import defaultdict
import builtins



class DecisionNode(object):
    
    """
    DecisionNode is a python class representing a  node in our decision tree
    
    """

    
    def __init__(self,
                 column=None,
                 value=None,
                 false_branch=None,
                 true_branch=None,
                 current_results=None,
                 is_leaf=False,
                 data = None,
                 results=None,
                 curent_dept = 0):
        
        
        self.column = column
        self.current_dept = curent_dept
        self.value = value
        self.false_branch = false_branch
        self.true_branch = true_branch
        self.current_results = current_results
        self.is_leaf = is_leaf
        self.results = results
        self.data = data

class DecisionTree(object):
    """
    DecisionTree class, that represents one Decision Tree

    :param max_tree_depth: maximum depth for this tree.
    """
    def __init__(self, max_tree_depth):
        self.max_tree_depth = max_tree_depth
        
    def dict_of_values(self,data):

        """
        param data: a 2D Python list representing the data. Last column of data is Y.
        return: returns a python dictionary showing how many times each value appears in Y

        """
        
        self.data = data
        results = defaultdict(int)
        for row in data:
            r = row[len(row) - 1]
            results[r] += 1
        return builtins.dict(results)


    def divide_data(self, data, feature_column, feature_val):

        """
        divide_data a function takes the data and divides it in two parts by a line. A line
        is defined by the feature we are considering (feature_column) and the target 
        value. The function returns a tuple (data1, data2) which are the desired parts of the data.
        For int or float types of the value, data1 have all the data with values >= feature_val
        in the corresponding column and data2 should have rest.
        For string types, data1 should have all data with values == feature val and data2 should 
        have the rest.

        param data: a 2D Python list representing the data. Last column of data is Y.
        param feature_column: an integer index of the feature/column.
        param feature_val: can be int, float, or string
        return: a tuple of two 2D python lists
        """
        
        self.data = data
        self.feature_column = feature_column
        self.feature_val = feature_val
        
        data1 = []
        data2 = []
        r = [row[feature_column] for row in data]

        item = [i for i,j in enumerate(r) if j>=feature_val] #i is row index, j is value
        for i in item:
            data1.append(data[i])
        item = [i for i,j in enumerate(r) if j<feature_val]
        for i in item:
            data2.append(data[i])

        return data1, data2



    def gini_impurity(self, data1, data2):
        
        """
        Given two 2D lists of compute their gini_impurity index. 
        Remember that last column of the data lists is the Y
        Lets assume y1 is y of data1 and y2 is y of data2.
        gini_impurity shows how diverse the values in y1 and y2 are.
        gini impurity is given by 

        N1*sum(p_k1 * (1-p_k1)) + N2*sum(p_k2 * (1-p_k2))

        where N1 is number of points in data1
        p_k1 is fraction of points that have y value of k in data1
        same for N2 and p_k2


        param data1: A 2D python list
        param data2: A 2D python list
        return: a number - gini_impurity 
        """
        
        self.data1 = data1
        self.data2 = data2

        d1 = self.dict_of_values(data1)
        N1 = d1.get(1.0, 0) + d1.get(1.0, 0)
        #N1 = sum(d1.values())
        if N1!=0  :
            P1 = [d1.get(1.0, 0)/N1, d1.get(0, 0)/N1]
        else:
            P1 = [0,0]
            N1 = 0

        d2 = self.dict_of_values(data2)
        N2 = d2.get(1.0, 0) + d2.get(1.0, 0)
        if N2!=0 :
            P2 = [d2.get(1.0, 0)/N2, d2.get(0, 0)/N2]
        else:
            P2 = [0,0]
            N2 = 0


        gini_impurity = 0

        for i in range(2):
            gini_impurity += N1*P1[i]*(1-P1[i]) + N2*P2[i]*(1-P2[i])

        return gini_impurity
    
    
        
    def fit(self, data, current_depth = 0):
        
        max_tree_depth = self.max_tree_depth
        self.data = data
        
        
        if len(data) == 0:
            return DecisionNode(is_leaf=True)

        elif(current_depth == max_tree_depth):
            return DecisionNode(current_results=self.dict_of_values(data))

        elif(len(self.dict_of_values(data)) == 1):
            return DecisionNode(current_results=self.dict_of_values(data), is_leaf=True)
        else:

            #This calculates gini number for the data before dividing 
            self_gini = self.gini_impurity(data, [])

            #Below are the attributes of the best division that you need to find. 
            #You need to update these when you find a division which is better
            best_gini = 1e100
            best_column = None
            best_value = None
            #best_split is tuple (data1,data2) which shows the two datas for the best divison so far
            best_split = None

            for c in range(len(data[0])-1):
                val = list(set(row[c] for row in data)) 
                for value in val:
                    data1, data2 = self.divide_data(data, c, value)
                    gini = self.gini_impurity(data1, data2)
                    if (gini < best_gini):
                        best_gini = gini
                        best_column = c
                        best_value = value
                        best_split = (data1, data2)

        #if best_gini is no improvement from self_gini, we stop and return a node.
        if abs(self_gini - best_gini) < 1e-10:
            return DecisionNode(current_results=self.dict_of_values(data), is_leaf=True)
        else:
            
            t1 = self.fit(best_split[1], current_depth+1)  
            t2 = self.fit(best_split[0], current_depth+1) 
        
        return DecisionNode(true_branch=t1, false_branch=t2, value = best_value, column=best_column, data = data)

    
    
    def predict(self, X, node, Y = []):
        
        self.node = node
        self.X = X
        
        if type(X[0]) == int or type(X[0]) == float or type(X[0]) == np.float64:
            if node.is_leaf:   #next - to take rows one by one   
                best = max(np.array(list(node.current_results.values())))
                Y.append(1 - next((i for i, j in node.current_results.items() if j == best)))
            else:
                if X[node.column]>= node.value:
                    return self.predict(X, node.true_branch, Y)
                else:
                    return self.predict(X, node.false_branch, Y)
        else:
            for i in range(len(X)):
                if node.is_leaf: 
                    best = max(np.array(list(node.current_results.values()))) 
                    Y.append(1 - next((i for i, j in node.current_results.items() if j == best)))
                else:
                    if X[i][node.column]>= node.value:
                        self.predict(X[i], node.true_branch, Y)
                    else:
                        self.predict(X[i], node.false_branch, Y)
        return Y