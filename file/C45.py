import pandas as pd
import numpy as np
import math
from anytree import Node, RenderTree, find, Walker,DoubleStyle,LevelOrderIter,findall

class Feature:
    def __init__(self, name=None, unique=None,info=0.0,
                 df=None,gain=0.0,split_info=0.0):
        
        self.name = name
        self.unique = unique
        self.info = info
        self.gain = gain
        self.split_info = split_info
        self.gain_ratio = 0.0
        self.dataset = df

class C45:
    def __init__(self):
        self.except_features = []
        self.feature_list={}
        self.selected_feature=[]
        self.tree = None
        self.ROOT = 'root'
        self.LABEL = 'label'
        self.DECISION = 'class'
        self.VALUE = 'value'        
        
    def read_csv(self,filename):
        df = pd.read_table(filename, sep=';', engine='python')
        self.label_name = df.columns[-1]
        self.number_of_entries = len(df)
        self.df = df
        
    def remove_feature(self,feature):
        if feature not in self.except_features:
            self.except_features.append(feature)
            self.features = [item for item in self.df.columns if item not in self.except_features]
#             self.df = self.df.drop(feature,axis=1)
        else:
            print(f'{feature} is removed!')
        
    def identify_feature(self):
        except_features = self.except_features
        df = self.df
        for col in df:
            if col not in except_features:
                feature = Feature(name=col,unique=df[col].unique())
                self.feature_list[col] = feature
#         for key in self.feature_list:
#             subdf = self.df[[key,self.label_name]]
#             self.feature_list[key].dataset = subdf
    
    def log2(self,x):
        if x == 0:
            return 0
        else:
            return math.log(x,2)
        
    def calc_info(self,featureObj,labelObj,df):
#         print(f'Processing {featureObj.name}')
        number_of_entries = len(df)
        sum_info = 0.0
        classP = 0.0
        if featureObj == labelObj:
            info = 0.0
            for label_value in labelObj.unique:
                idxs = df[(df[labelObj.name]==label_value)].index
                occur = len(idxs)
#                 print('occur:',occur)
                valueP = float(occur)/number_of_entries
                info = info - (valueP * ( self.log2(valueP) ) )
            return info
        
        for feature_value in featureObj.unique:
            info = 0.0
            idxs = df[(df[featureObj.name]==feature_value)].index
            Dj = len(idxs)
#             print(f'Occurance: {Dj}, ClassP: {classP}')
            classP = float(Dj)/number_of_entries
            for label_value in labelObj.unique:
                idxs = df[(df[featureObj.name]==feature_value) & (df[labelObj.name]==label_value)].index
                occur = len(idxs)
#                 print(f'{feature_value} {label_value} {occur}/{Dj}')
                if(Dj != 0.0):
                    valueP = float(occur)/Dj
                else:
                    valueP = 0.0
                info = info - (valueP * ( self.log2(valueP) ) )
            split_info = classP * info
            sum_info = sum_info + split_info
#         print(f'Info {featureObj.name}(D) = {sum_info}')
#         print('========================================================')
        return sum_info

    def find_feature(self,dataset):
        feature_list = {}
        for col in dataset:
            if col not in self.except_features:
                feature = Feature(name=col,unique=dataset[col].unique())
                feature_list[col] = feature
        return feature_list

    def best_feature(self,feature_list):
        bestGain = 0.0
        for key in feature_list: 
            featureObj = feature_list[key]
            if featureObj.gain_ratio > bestGain:
                bestFeature = featureObj
                bestGain = featureObj.gain_ratio
        if bestGain == 0.0:
            return -99
        return bestFeature
    
    def find_best_features(self,feature_list,df):
        labelObj = feature_list[self.label_name]
        for key in feature_list:
            featureObj = feature_list[key]
            featureObj.info = self.calc_info(featureObj,labelObj,df)
            featureObj.split_info = self.calc_info(featureObj,featureObj,df)
        labelObj = feature_list[self.label_name]
        for key in feature_list:
            featureObj = feature_list[key]
            if featureObj == labelObj:
                continue
            featureObj.gain = labelObj.info - featureObj.info
            if(featureObj.gain != 0.0):
                featureObj.gain_ratio = featureObj.gain / featureObj.split_info
            else:
                featureObj.gain_ratio = 0.0
            print(f'{featureObj.name} info={featureObj.info:.4f} gain={featureObj.gain:.4f} split_info={featureObj.split_info:.4f} gain_ratio={featureObj.gain_ratio:.4f}')
        return feature_list

    def find_best_label(self,labelObj,df):
        count = 0
        bestLabel = labelObj.unique[0]
        for value in labelObj.unique:
            idxs = df[(df[labelObj.name]==value)].index
            newCount = len(idxs)
            if newCount > count:
                bestLabel = value
        return bestLabel
    
    def split_dataset(self,name,value,dataset):
        dataset = dataset.loc[(dataset[name]==value)]
        dataset = dataset.drop(name,axis=1)
        return dataset
    
    def create_value_node(self,feature,df,currentNode):
        for value in feature.unique:
            dataset = self.split_dataset(feature.name,value,df)
            newNode = Node(value,parent=currentNode,dataset=dataset,type=self.VALUE)
            
    def create_tree(self):
        # feature_list = model.feature_list
        print('Identifing first feature...')
        feature_list = self.find_feature(self.df)
        feature_list = self.find_best_features(feature_list,self.df)
        bestFeature = self.best_feature(feature_list)
        root = Node(bestFeature.name,type=self.ROOT)
        print(f'Best feature: {bestFeature.name}')
        for value in bestFeature.unique:
            dataset = self.split_dataset(bestFeature.name,value,self.df)
        #     dataset = model.df.loc[(model.df[bestFeature.name]==value)]
        #     dataset = dataset.drop(bestFeature.name,axis=1)
            newNode = Node(value,parent=root,dataset=dataset,type=self.VALUE)
        self.tree = root
        self.display_tree()
        for node in LevelOrderIter(root):
            print('=================================')
            print(f'Node: {node.name} Type:{node.type}')
            if node != root and node.type != self.LABEL and node.type != self.DECISION:
        #         print(f'Node: {node.name}')
                print(node.dataset)
                feature_list = self.find_feature(node.dataset)
#                 print(f'Length: {len(feature_list)}')
                feature_list = self.find_best_features(feature_list,node.dataset)
                bestFeature = self.best_feature(feature_list)
                if(bestFeature != -99):
                    print(f"Best feature: {bestFeature.name}")
                    newNode = Node(bestFeature.name,parent=node,type=self.DECISION)
                    self.create_value_node(bestFeature,node.dataset,newNode)
                else:
                    labelObj = feature_list[self.label_name]
                    best_label = self.find_best_label(labelObj,node.dataset)
                    print(f'Selected label: {best_label}')
                    newNode = Node(best_label,parent=node,type=self.LABEL)
                self.display_tree()
            else:
                print(f'Skip {node.name} {node.type}')
                continue
        return root
    
    def read_testset(self,file):
        df = pd.read_table(file, sep=';', engine='python')
        label_name = df.columns[-1]
        number_of_entries = len(df)
        features = [item for item in df.columns if item != label_name]
        return label_name,number_of_entries,df,features
    
    def check_value(self,currentNode,data):
        found = False
        if currentNode.type == self.LABEL:
            return currentNode
        for child in currentNode.children:
    #         print(f'Data:{str(data[currentNode.name])} type({type(str(data[currentNode.name]))}) compare child {str(child.name)} type({type(str(child.name))})')
            if(str(data[currentNode.name]) == str(child.name)):
                found = True
                return child
        if not found:
            return currentNode

    def get_label(self,data):
        #initialize with root of tree
        currentNode = self.tree
        #start to find prediction
        while True:
    #         print(currentNode.name,currentNode.type)
    #         if it is label mean leaf
            if currentNode.type == self.LABEL:
                return currentNode.name
            #keep decending
            valueNode = self.check_value(currentNode,data)
            if(currentNode == valueNode):
#                 print(f'Missing id={data.values[0]} data={data[currentNode.name]}')
                return self.missing_get_label(currentNode)
            # if it is leaf return result
    #         if valueNode.type == model.LABEL:
    #             return valueNode.name
            #go to next node
            currentNode = valueNode.children[0]
            
    def missing_get_label(self,currentNode):
        labels = []
        for node in LevelOrderIter(currentNode):
    #         print(node.name,node.type)
            if(node.type == self.LABEL):
                labels.append(node.name)
        labels = np.array(labels)
        unique_label = np.unique(labels)
        label_list = {}
    #     print(labels)
    #     print(unique_label)
        for item in unique_label:
            mask = np.isin(labels,item)
            label_list[item] = mask.sum()
    #         print(item,len(idxs))
        bestLabel = self.missing_best_label(label_list)
        return bestLabel
            
    def missing_best_label(self,label_list):
        key_list = list(label_list)
    #     print(key_list)
        bestLabel = label_list[key_list[0]]
        bestCount = 0
        for key in key_list:
            labelCount = label_list[key]
            if(labelCount > bestCount):
                bestCount = labelCount
    #             print(key,bestCount)
        return key
            
    def predict_file(self,file):
        label_name,number_of_entries,dataset,features = self.read_testset(file)
        predictions = []
        print(f'Number of entries: {number_of_entries}')
        print(f'Label: {label_name}')
        print(f'Features: {features}')
        for index in dataset.index:
            data = dataset.loc[index]
            label = self.get_label(data)
            predictions.append(label)
#             print(f'Index: {index}')
#             print(f'Actual: {data[model.label_name]}')
#             print(f'Predicted: {label}')
        return predictions,dataset

    def display_predictions(self,predictions,dataset):
#         print(dataset.columns[0],self.label_name,'Predictions')
        df = pd.DataFrame(columns=[dataset.columns[0],self.label_name,'Predictions'])
        df['Predictions'] = predictions
        df[dataset.columns[0]]=dataset[dataset.columns[0]]
        df[self.label_name]=dataset[self.label_name]
        print(df)

    def display_tree(self):
        for pre,_,node in RenderTree(self.tree,DoubleStyle):
                print("%s%s" % (pre, node.name))    
        
    def display_feature_list(self):
        for key in feature_list:
            print('======================================')
            print(f'Feature name: {feature_list[key].name}')
            print(f'Unique: {feature_list[key].unique}')
            print(f'Info Value: {feature_list[key].entropy}')
            print(f'Dataset: {feature_list[key].dataset}')
            print('======================================')
                
    def info(self):
        self.features = [item for item in self.df.columns if item not in self.except_features]
        print(f'Remove feature: {self.except_features}')
        print(f'Available feature: {self.features}')
        print(f'Number of entries: {self.number_of_entries}')