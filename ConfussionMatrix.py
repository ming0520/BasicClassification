import pandas as pd
import numpy as np
import math
from anytree import Node, RenderTree, find, Walker,DoubleStyle,LevelOrderIter,findall

class ConfussionMatrix:
    def __init__(self,tp=0,fp=0,fn=0,tn=0,support=0,label=None):
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn
        self.support = support
        self.label=label
        
    def get_p(self):
        self.p = self.tp + self.fn
        return self.p
    
    def get_n(self):
        self.n = self.fp + self.tn
        return self.n
    
    def accuracy(self):
        p = self.get_p()
        n = self.get_n()
        if p == 0.0 or n == 0.0:
            return 0.0
        return float(self.tp+self.tn)/(p+n)
    
    def error_rate(self):
        p = self.get_p()
        n = self.get_n()
        if p == 0.0 or n == 0.0:
            return 0.0
        return float(self.fp+self.fn)/(p+n)
    
    def recall(self):
        p = self.get_p()
        if p == 0.0:
            return 0.0
        return float(self.tp)/(p)
    
    def specificity(self):
        n = self.get_n()
        if n == 0.0:
            return 0.0
        return float(self.tn)/(n)
    
    def percision(self):
        divider = self.tp + self.fp
        if(divider == 0.0):
            return 0.0
        return float(self.tp)/(divider)
    
    def f1(self):
        percision = self.percision()
        recall = self.recall()
        total = percision + recall
        if(total == 0.0):
            return 0.0
        return float(2*percision*recall)/(percision+recall)
    
    def weighted_f1(self):
        return float(self.f1()) * self.support
    
    def weighted_recall(self):
        return float(self.recall()) * self.support
    
    def weighted_percision(self):
        return float(self.percision()) * self.support
    
    def weighted_error(self):
        return float(self.error_rate()) * self.support 
    
    def display_report(self,name):
#         print('%5s %5s %5s %5s %5s' % ('Name','Accuracy','Percision','Recall','F1'))
        print('%7s'% name,end =' ')
        print('%7.2f' % self.accuracy(),end=' ')
        print('%7.2f' %self.percision(),end=' ')
        print('%7.2f' %self.recall(),end=' ')
        print('%7.2f' %self.f1(),end=' ')
        print('%7d' %self.support)
    def display_matrix(self):
        print(f'TP = {self.tp} FP = {self.fp}')
        print(f'FN = {self.fn} TP = {self.tn}')

class Report:
    def __init__(self):
        pass
    def create_cm_list(self,actualList,predictionList,labels):
        cm_list = {}
        for label in labels:
            TP = 0
            FN = 0
            FP = 0
            TN = 0
            mask = np.isin(actualList,label)
            support = mask.sum()
            for index,predict in enumerate(predictionList):
                actual = actualList[index]
                if(predict == label == actual):
                    TP +=1
                elif(predict == actual and predict != label):
                    TN +=1
                elif(predict != actual and predict == label):
        #             FN +=1
                    FP += 1
                elif(predict != actual and actual != label):
                    TN +=1
                elif(predict != actual and actual == label):
        #             FP +=1
                    FN+=1
            cm = ConfussionMatrix(TP,FP,FN,TN,support,label)
            cm_list[label] = cm
        return cm_list

    def create_report(self,cm_list,labels):
        totalTP =0
        totalFP =0
        totalFN =0
        totalSupport =0
        totalF1 =0
        totalRecall =0
        totalPercision =0
        totalWeightedRecall =0
        totalWeightedPercision =0
        totalWeightedF1=0
        totalError=0
        totalWeightedError=0
        for label in labels:
            cm = cm_list[label]
            totalTP += cm.tp
            totalFP += cm.fp
            totalFN += cm.fn
            totalF1 += cm.f1()
            totalWeightedF1 += cm.weighted_f1()
            totalSupport += cm.support
            totalRecall += cm.recall()
            totalPercision += cm.percision()
            totalWeightedRecall += cm.weighted_recall()
            totalWeightedPercision += cm.weighted_percision()
            totalError += cm.error_rate()
            totalWeightedError += cm.weighted_error()
            print('-----------------------------')
            print(f'Label:{cm.label}')
            print(f'Accuracy: {cm.accuracy():.2f}')
            print(f'Error: {cm.error_rate():.2f}')
            print(f'Specificity: {cm.specificity():.2f}')
            print(f'Percision: {cm.percision():.2f}')
            print(f'Recall: {cm.recall():.2f}')
            print(f'F1-score: {cm.f1():.2f}')
            print(f'support: {cm.support:d}')
            cm.display_matrix()
            
        print('-----------------------------')
        nol = len(labels)
        P = totalTP+totalFN
        micro_f1 = totalTP/P
        macro_f1 = (totalF1)/nol
        weigthed_f1 = totalWeightedF1 / totalSupport
        if(nol > 2):
            print(f'Micro f1/Accuracy: {micro_f1:.2f}')
        print(f'Macro f1: {macro_f1:.2f}')
        print(f'Weighted f1: {weigthed_f1:.2f}')
        
        micro_error = (totalFP+totalFN)/totalSupport
        macro_error = (totalError)/nol
        weigthed_error = totalWeightedError / totalSupport
        if(nol > 2):
            print(f'Micro error: {micro_error:.2f}')
        print(f'Macro error: {macro_error:.2f}')
        print(f'Weighted error: {weigthed_error:.2f}')     
        
        micro_percision = (totalTP)/(totalTP+totalFP)
        macro_percision = totalPercision / nol
        weighted_percision = totalWeightedPercision / totalSupport
        if(nol > 2):
            print(f'Micro percision: {micro_percision:.2f}')
        print(f'Macro percision: {macro_percision:.2f}')
        print(f'Weighted percision: {weighted_percision:.2f}')

        micro_recall = totalTP/P
        macro_recall = totalRecall / nol
        weighted_recall = totalWeightedRecall / totalSupport
        if(nol > 2):
            print(f'Micro recall: {micro_recall:.2f}')
        print(f'Macro recall: {macro_recall:.2f}')
        print(f'Weighted recall: {weighted_recall:.2f}')   