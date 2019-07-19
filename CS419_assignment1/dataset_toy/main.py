#code to build and train the regression tree
import numpy as np 
from argparse import ArgumentParser
import pandas as pd
import sys

#taking input parameters from the command line
parser = ArgumentParser()
parser.add_argument('train_data', type=str, help='Give the training data')
parser.add_argument('test_data', type=str, help='Give the test data')
parser.add_argument('min_leaf_size', type=int, help='Minimum no of elements in leaf')
parser.add_argument('error', type=str, help='Give the test data')
args = parser.parse_args()

#some utility due to os for changing the derictory
training_data= pd.read_csv(args.train_data)
testing_data= pd.read_csv(args.test_data)
#error=args.error
#print(training_data["Angle"][1])
#print(training_data.loc[1,"Angle"])
parameters=['Speed', 'Angle']

class node:
    def __init__(self, data, split_parameter, split_point, error):
        self.split_data=data
        self.split_point=split_point
        self.split_parameter=split_parameter
        self.error=error
        self.count=len(data)
        self.leftChild = None
        self.rightChild = None

class decision_tree:
    def __init__(self):
        self.root=None

    def split(self, currentNode, training_data, parameters):
        if(self.root is None):
            split_citeria= self.best_parameter(training_data, parameters)
            self.root = node(training_data, split_citeria[0], split_citeria[1], split_citeria[2])
            self.split(self.root, training_data, parameters)
            return
        else:
            if(currentNode.count < 20 or currentNode.count == 20):
                return
            else:
                split_citeria= currentNode.best_parameter(training_data, parameters)
                data1= pd.DataFrame(columns=parameters)
                data2= pd.DataFrame(columns=parameters)
                for x in range (0,14):
                    if(training_data[split_citeria[0]][x] <= training_data[split_citeria[0]][split_citeria[1]]):
                        data1= data1.append(pd.DataFrame(training_data[[x]], columns=parameters), ignore_index=True)
                    else:
                        data2= data2.append(pd.DataFrame(training_data[[x]], columns=parameters), ignore_index=True)
                currentNode.leftChild= node(data1, split_citeria[0], split_citeria[1], split_citeria[2])
                currentNode.rightChild= node(data2, split_citeria[0], split_citeria[1], split_citeria[2])
                split(currentNode.leftChild, data1, parameters)
                split(currentNode.rightChild, data2, parameters)
                return

    def absolute_error(self, data):
        n=len(data)
        if n==0:
            return 0
        average= sum(data)/n
        error=0
        for x in range (0,n-1):
            error= error + abs(data[x]-average)
        return error	

    def mean_squared_error(self, data):
        n=len(data)
        if n==0:
            return 0
        average= sum(data)/n
        error=0
        for x in range (0,n-1):
            error= error + (data[x]-average)**2
        return error	


    def error_function(self, training_data, ch): #needs correction
        error_type=args.error
        split_point=[None, None]
        if(error_type=='absolute'):
            for x in range (0,14):
                data1=[]
                data2=[]
                for y in range (0,14):
                    if(training_data[ch][y] <= training_data[ch][x]):
                        data1.append(training_data[ch][y])
                    else:
                        data2.append(training_data[ch][y])
                error= self.absolute_error(data1) + self.absolute_error(data2)
                if(split_point[1]==None or split_point[1] > error):
                    split_point[0]=training_data[ch][x]
                    split_point[1]=error
            return split_point
        elif(error_type=='mean_squared'):
            for x in range (0,14):
                data1=[]
                data2=[]
                for y in range (0,14):
                    if(training_data[ch][x] <= training_data[ch][y]):
                        data1.append(training_data[ch][y])
                    else:
                       data2.append(training_data[ch][y])
                error= self.mean_squared_error(data1) + self.mean_squared_error(data2)
                if(split_point[1]==None or split_point[1] > error):
                    split_point[0]=training_data[ch][x]
                    split_point[1]=error
            return split_point

    def best_parameter(self, training_data, parameters):
        split_citeria= ['\0', float("inf"), float("inf")]
        for ch in parameters:
            test= self.error_function(training_data, ch)
            if(test[0] < split_citeria[1]):
                split_citeria[0]= ch
                split_citeria[1]= test[0]
                split_citeria[2]= test[1]
        return split_citeria

    def train(self, training_data, parameters):
        self.split(self.root, training_data, parameters)

    def test(self, test_data):
        pass

def prune(s):
    pass

if __name__ == '__main__':
    s=decision_tree()
    s.train(training_data,parameters)
    s.test(testing_data)
    prune(s)
    s.test(testing_data)	