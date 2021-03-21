import pandas as pd
import numpy as np
from ConfussionMatrix import Report,ConfussionMatrix
from collections import defaultdict

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

class Prior:
    def __init__(self, feature=None, label=None,feature_support=0,label_support=0,label_count=0):
        self.feature = feature
        self.label = label
        self.feature_support = feature_support
        self.label_support = label_support
        self.label_count = label_count
        self.split_prob = 0
    
    def probability(self):
        feature_support = self.feature_support
        label_support = self.label_support
        if (label_support == 0):
            return 0
        elif(feature_support == 0):
            feature_support += 1
            label_support += self.label_count
        return feature_support/label_support
            
class NaiveBayesian:
    def __init__(self,verbose=True):
        self.except_features = []
        self.feature_list={}
        self.verbose = verbose
        self.isLoad = False
        
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
            
    def find_feature(self,dataset):
        feature_list = {}
        for col in dataset:
            if col not in self.except_features:
                feature = Feature(name=col,unique=dataset[col].unique())
                feature_list[col] = feature
        return feature_list
    
    def create_model(self):
        feature_list = self.find_feature(self.df)
        key_list = list(feature_list)
        label_dict = {}
        labelObj = feature_list[self.label_name]
        for label in labelObj.unique:
            label_idxs = self.df[(self.df[labelObj.name]==label)].index
            label_dict[label] = len(label_idxs)

        self.label_dict=label_dict

        prior_dict = {}
        labelObj = feature_list[self.label_name]  
        for key in feature_list:
            featureObj = feature_list[key]
            value_dict={}
            for unique in featureObj.unique: 
                unique_list={}
                for label in labelObj.unique:
                    feature_idxs = self.df[((self.df[featureObj.name]==unique)&(self.df[labelObj.name]==label))].index
                    feature_count = len(feature_idxs)
                    label_count = label_dict[label]
                    prior = Prior(unique, label,feature_count,label_count,len(labelObj.unique))
                    prior.split_prob = self.label_dict[label]/self.number_of_entries
                    unique_list[label] = prior
                value_dict[unique] = unique_list
            prior_dict[key] = value_dict
        self.prior_dict = prior_dict
        self.label_dict = label_dict
        self._feature_list = feature_list
        
    def get_model(self):
        data = []
        for key in self.prior_dict:
            value_dict = self.prior_dict[key]
            for value_key in value_dict:
                label_dict = value_dict[value_key]
                for label_key in label_dict:
                    prior = label_dict[label_key]
                    ls = {'feature':key,
                          'X':prior.feature,
                          'C':prior.label,
                          'Xi':prior.feature_support,
                          'Ci':prior.label_support,
                          'FeatureType':prior.label_count,
                          'Probability':prior.probability(),
                          'SplitProb': prior.split_prob
                         }
                    data.append(ls)
        df = pd.DataFrame(data)
        self.print_full(df)
        return df
    
    def save(self,file):
        df = self.get_model()
        df.to_csv(file)
        
    def read_testset(self,file):
        df = pd.read_table(file, sep=';', engine='python')
        label_name = df.columns[-1]
        number_of_entries = len(df)
        features = [item for item in df.columns if item != label_name]
        return label_name,number_of_entries,df,features        
        
    def predict(self,data):
        if self.verbose:
            print('-----------------------Predict for-----------------------')
            print(data)
            print()
        key_list = list(self.prior_dict)
        label_result = {}
        for label_key in self.label_dict:
            pX = 1.0
            if self.verbose:
                print(f'Label: {label_key}')
            for class_key in key_list:
                feature = data[class_key]
                try:
                    feature_dict = self.prior_dict[str(class_key)][str(feature)][str(label_key)]
                    if self.verbose:
                        print(feature,feature_dict.probability())
                    pX *= feature_dict.probability()
                except:
                    print(f'Missing data for {data[0]}:{class_key}: {feature} => {label_key}')
            try:
                if self.isLoad:
                    pC = float(pX) * feature_dict.split_prob
                else:
                    pC = float(pX) * (self.label_dict[label_key]/self.number_of_entries)
            except:
                pC = 0.0
            label_result[label_key] = pC
            if self.verbose:
                print(f"P({label_key}|X): {pC} / P(X)")
                print()

            bestPc = 0.0
            bestKey = list(label_result)[0]
            for label_key in label_result:
                if(label_result[label_key] > bestPc):
                    bestPc = label_result[label_key]
                    bestKey = label_key
        if self.verbose:
            print(f'Selected label: {bestKey}')
            print('-----------------------End predict-----------------------')
        return bestKey
        
    def predict_file(self,file,verbose=None):
        label_name,number_of_entries,dataset,features = self.read_testset(file)
        if self.isLoad:
            self.label_name = dataset.columns[-1]
            self.df = dataset
            # self.df[self.label_name] = dataset[self.label_name].unique()
        predictions = []
        if verbose == None:
            verbose =self.verbose
        if verbose:
            print(f'Number of entries: {number_of_entries}')
            print(f'Label: {label_name}')
            print(f'Features: {features}')
        for index in dataset.index:
            data = dataset.loc[index]
            label = self.predict(data)
            predictions.append(label)
        return predictions,dataset
    
    def display_predictions(self,predictions,dataset):
#         print(dataset.columns[0],self.label_name,'Predictions')
        df = pd.DataFrame(columns=[dataset.columns[0],self.label_name,'Predictions'])
        df['Predictions'] = predictions
        df[dataset.columns[0]]=dataset[dataset.columns[0]]
        df[self.label_name]=dataset[self.label_name]
        self.print_full(df)

    def print_full(self,x):
        pd.set_option('display.max_rows', len(x))
        print(x)
        pd.reset_option('display.max_rows')
        
    def multi_level_dict(self):
        return defaultdict(self.multi_level_dict)
        
    def load(self,file):
        df = pd.read_csv(file)
        data = []
        model_list = self.multi_level_dict()
        for index in df.index:
            feature_class= df['feature'][index]
            feature_value = df['X'][index]
            label = df['C'][index]
            feature_suppot_count = df['Xi'][index]
            label_support_count = df['Ci'][index]
            label_count = df['FeatureType'][index]
            split_prob = df['SplitProb'][index]
            prior = Prior(feature_value, label,feature_suppot_count,label_support_count,label_count)
            prior.split_prob = split_prob
            model_list[str(feature_class)][str(feature_value)][str(label)] = prior
        # print(type(feature_class),type(feature_value),type(label))
        self.prior_dict = model_list
        label_dict = {}
        for item in df['C'].unique():
            label_dict[item] = 0  
        self.label_dict = label_dict
        self.print_full(df)

        # for class_key in self.prior_dict:
        #     class_value = model_list[class_key]
        #     for label_value in class_value:
        #         objs = class_value[label_value]
        #         for key in objs:
        #             obj = objs[key]
        #             print(obj.feature,obj.label,obj.feature_support,obj.label_support,obj.label_count,obj.probability())
        return df
    
    def info(self):
        self.features = [item for item in self.df.columns if item not in self.except_features]
        print(f'Remove feature: {self.except_features}')
        print(f'Available feature: {self.features}')
        print(f'Number of entries: {self.number_of_entries}')    