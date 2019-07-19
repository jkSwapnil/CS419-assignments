#for task2
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
#parser.add_argument('error', type=str, help='Give the test data')
args = parser.parse_args()

#some utility due to os for changing the derictory
training_data= pd.read_csv(args.train_data)
testing_data= pd.read_csv(args.test_data)
#error=args.error
#print(training_data["Angle"][1])
#print(training_data.loc[1,"Angle"])
parameters=['fixedacidity', 'volatileacidity', 'citricacid', 'residualsugar', 'chlorides', 'freesulfurdioxide', 'totalsulfurdioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
test_headers=['output']
test_results= pd.DataFrame(columns=test_headers)

class node:
    def __init__(self, data, parameters):
        self.split_data=data
        self.predicted_value=self.prediction(data)
        self.entropy=self.node_entropy(data)
        self.split_citeria= self.best_parameter(data, parameters) #paramater, point, error
        #print('a')
        self.count=len(data)
        self.leftChild = None
        self.rightChild = None
        #self.parent= None

    def prediction(self, data):
        n=len(data)
        if n==0:
            return None
        count=[0]*11
        for i in range (0,n):
            count[data['quality'][i]]+=1
        return count.index(max(count))

    def node_entropy(self, data):
        n=len(data)
        if n==0:
            return 0.0
        from math import log
        count=[0]*11
        for i in range (0,n):
            count[data['quality'][i]]+=1
        entropy=0.0
        for x in range (0,11):
            if(count[x]!=0):
                y=float(count[x])/n
                entropy-=y*log(y)
        return entropy

    def _entropy(self, data):
        n=len(data)
        if n==0:
            return 0.0
        from math import log
        count=[0]*11
        for i in range (0,n):
            count[data[i]]+=1
        entropy=0.0
        for x in range (0,11):
            if(count[x]!=0):
                y=float(count[x])/n
                entropy-=(y)*log(y)
        return entropy       

    def entropy_function(self, data, ch):
        split_point=[float("inf"), float("inf")]
        for x in range (0,len(data)):
            data1=[]
            data2=[]
            for y in range (0,len(data)):
                if(data[ch][y] <= data[ch][x]):
                    data1.append(data['quality'][y])
                else:
                    data2.append(data['quality'][y])
            child_entropy= (self._entropy(data1)*len(data1) + self._entropy(data2)*len(data2))/len(data)
            #child_entropy= self._entropy(data1) + self._entropy(data2)
            if(split_point[1] > child_entropy):#and (entropy-child_entropy)>0
                split_point[0]=data[ch][x]
                split_point[1]=child_entropy
        return split_point

    def best_parameter(self, data, parameters):
        split_citeria= ['\0', float("inf"), float("inf")]
        for ch in parameters:
            if ch=='quality':
                break
            test= self.entropy_function(data, ch)
            if(test[1] < split_citeria[2]):
                split_citeria[0]= ch
                split_citeria[1]= test[0]
                split_citeria[2]= test[1]
        return split_citeria



class decision_tree:
    def __init__(self,training_data, parameters):
        self.root= node(training_data, parameters)
        self.split(self.root, training_data, parameters)

    def split(self, currentNode, training_data, parameters):
        print(currentNode.count)
        if(currentNode.count <= 100):
            return
        # if(currentNode.parent != None and currentNode.count==currentNode.parent.count):
        #     return
        data1= pd.DataFrame(columns=parameters)
        data2= pd.DataFrame(columns=parameters)
        for x in range (0, currentNode.count):
            if(training_data[currentNode.split_citeria[0]][x] <= currentNode.split_citeria[1]):
                data1= data1.append(pd.DataFrame(training_data.loc[[x]], columns=parameters), ignore_index=True)
            else:
                data2= data2.append(pd.DataFrame(training_data.loc[[x]], columns=parameters), ignore_index=True)
        currentNode.leftChild= node(data1, parameters)
        currentNode.rightChild= node(data2, parameters)
        #currentNode.leftChild.parent = currentNode
        #currentNode.rightChild.parent = currentNode
        self.split(currentNode.leftChild, data1, parameters)
        self.split(currentNode.rightChild, data2, parameters)

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
    test_results.to_csv('output.csv', sep=',', encoding='utf-8')
    #print(test_results)


if __name__ == '__main__':
    s=decision_tree(training_data,parameters)
    #prune(s)
    p=s.root
    test(p, testing_data)
