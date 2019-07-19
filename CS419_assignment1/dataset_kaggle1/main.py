#code to build and train the regression tree
import numpy as np 
from argparse import ArgumentParser
import pandas as pd
import os

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
parameters=['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'output']
test_headers=['predection']

class node:
    def __init__(self, data, parameters):
        self.split_data=data
        self.predicted_value=data['output'].mean()
        self.split_citeria= self.best_parameter(data, parameters) #paramater, point, error
        self.count=len(data)
        self.leftChild = None
        self.rightChild = None
        self.parent= None

    def prediction(self):
        return self.predicted_value

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


    def error_function(self, data, ch):
        error_type=args.error
        split_point=[float("inf"), float("inf")]
        if(error_type=='absolute'):
            for x in range (0,len(data)):
                data1=[]
                data2=[]
                for y in range (0,len(data)):
                    if(data[ch][y] <= data[ch][x]):
                        data1.append(data['output'][y])
                    else:
                        data2.append(data['output'][y])
                error= self.absolute_error(data1) + self.absolute_error(data2)
                if(split_point[1] > error):
                    split_point[0]=data[ch][x]
                    split_point[1]=error
            return split_point
        elif(error_type=='mean_squared'):
            for x in range (0,len(data)):
                data1=[]
                data2=[]
                for y in range (0,len(data)):
                    if(data[ch][y] <= data[ch][x]):
                        data1.append(data['output'][y])
                    else:
                       data2.append(data['output'][y])
                error= self.mean_squared_error(data1) + self.mean_squared_error(data2)
                if(split_point[1] > error):
                    split_point[0]=data[ch][x]
                    split_point[1]=error
            return split_point

    def best_parameter(self, data, parameters):
        split_citeria= ['\0', float("inf"), float("inf")]
        for ch in parameters:
            if ch=='output':
                break
            test= self.error_function(data, ch)
            if(test[1] < split_citeria[2]):
                split_citeria[0]= ch
                split_citeria[1]= test[0]
                split_citeria[2]= test[1]
        #print(split_citeria)
        return split_citeria



class decision_tree:
    def __init__(self,training_data, parameters):
        self.root= node(training_data, parameters)
        self.split(self.root, training_data, parameters)

    def split(self, currentNode, training_data, parameters):
        #print(currentNode.count)
        if(currentNode.count <= 40):
            return
        if(currentNode.parent != None and currentNode.count==currentNode.parent.count):
            return
        data1= pd.DataFrame(columns=parameters)
        data2= pd.DataFrame(columns=parameters)
        for x in range (0, currentNode.count):
            if(training_data[currentNode.split_citeria[0]][x] <= currentNode.split_citeria[1]):
                data1= data1.append(pd.DataFrame(training_data.loc[[x]], columns=parameters), ignore_index=True)
            else:
                data2= data2.append(pd.DataFrame(training_data.loc[[x]], columns=parameters), ignore_index=True)
        currentNode.leftChild= node(data1, parameters)
        currentNode.rightChild= node(data2, parameters)
        currentNode.leftChild.parent = currentNode
        currentNode.rightChild.parent = currentNode
        self.split(currentNode.leftChild, data1, parameters)
        self.split(currentNode.rightChild, data2, parameters)

    def test(self, test_data):
        pass

def prune(s):
    pass

def add_data(p, value, i, test_results):
    if(p.leftChild==None and p.rightChild==None):
        test_results.loc[i]=[p.predicted_value]
        return test_results
    elif(value <= p.split_citeria[1]):
        return add_data(p.leftChild, value, i, test_results)
    else:
        return add_data(p.rightChild, value, i, test_results)

def test(p,testing_data):
    test_results= pd.DataFrame(columns=test_headers)
    for i in range (0,len(testing_data)):
        test_results=add_data(p, testing_data[p.split_citeria[0]][i], i, test_results)
    if(args.error=='absolute'):
    	test_results.to_csv('output_absolute.csv', sep=',', encoding='utf-8')
    else:
    	test_results.to_csv('output_meanSquared.csv', sep=',', encoding='utf-8')
    #print(test_results)


if __name__ == '__main__':
    s=decision_tree(training_data,parameters)
    #prune(s)
    p=s.root
    test(p, testing_data)