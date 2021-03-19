from C45 import C45
from NaiveBayesian import NaiveBayesian
from ConfussionMatrix import Report,ConfussionMatrix
import numpy as np
import os
import sys

def stop_working():
    print('Program stopped!')

def create_report(model,predictions,dataset):
    while True:
        print('Do you want to create report? (y/n)')
        try:
            choice = str(input())
            choice = choice[0].lower()
        except:
            continue
        if (choice =='y'):
            break
        elif (choice == 'n'):
            stop_working()
            return
        else:
            print('Invalid choice!')
    model.display_predictions(predictions,dataset)
    actualList = dataset[model.label_name]
    predictionList = np.array(predictions)
    labels = model.df[model.label_name].unique()
    report = Report()
    newList = report.create_cm_list(actualList,predictionList,labels)
    report.create_report(newList,labels)

def create_model_question():
    while True:
        print('Do you want to create model? (y/n)')
        try:
            choice = str(input())
            choice = choice[0].lower()
        except:
            continue
        if (choice =='y'):
            return True
        elif (choice == 'n'):
            stop_working()
            sys.exit()
        else:
            print('Invalid choice!')    
    print('============================================')    

def call_c45(train,test):
    model = C45()
    model.read_csv(train)
    model.remove_feature(model.df.columns[0])
    model.info()
    create_model_question()
    model.create_tree()
    model.display_tree()
    predictions,dataset = model.predict_file(test)
    model.display_predictions(predictions,dataset)
    actualList = dataset[model.label_name]
    create_report(model,predictions,dataset)


def call_naive(train,test):
    model = NaiveBayesian(verbose=False)
    model.read_csv(train)
    model.remove_feature(model.df.columns[0])
    model.info()
    create_model_question()
    model.create_model()
    print(model.get_model())
    predictions,dataset = model.predict_file(test,verbose=True)
    create_report(model,predictions,dataset)

def main():
    while True:
        print('Enter train file (.txt): ')
        try:
            train = str(input())
        except:
            continue
        if (os.path.exists(train)):
            break
        else:
            print('File does not exits')

    while True:
        print('Enter test file (.txt): ')
        try:
            test = str(input())
        except:
            continue
        if (os.path.exists(test)):
            break
        else:
            print('File does not exits')

    print('Select a method to analyze the frequent pattern')
    print('1) C4.5')
    print('2) Naive Bayesian')

    while True:
        print('Enter your choice:')
        choice = int(input())
        if (choice >= 1 and choice <= 2):
            break
        else:
            print('Invalid choice!')

    if(choice == 1):
        call_c45(train,test)
    elif(choice == 2):
        call_naive(train,test)
    else:
        print('Invalid choice! Exit program')

if __name__ == '__main__':
    main()
   
# name = 'student'
# train = name + '_train.txt'
# test = name + '_test.txt'
# call_c45(train,test)
# call_naive(train,test)