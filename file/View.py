from C45 import C45
from NaiveBayesian import NaiveBayesian
from ConfussionMatrix import Report,ConfussionMatrix
import numpy as np
import os
import sys

def stop_working():
    print('Program stopped!')
    sys.exit()

def get_file():
    while True:
        print('Enter file name (.txt): or (no)')
        try:
            filename = str(input())
        except:
            print('Invalid input!')
            continue
        if (filename == 'no'):
            stop_working()
        if (os.path.exists(filename)):
            return filename
        else:
            print('File does not exits')

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
        else:
            print('Invalid choice!')    
    print('============================================')

def call_c45(train):
    model = C45()
    model.read_csv(train)
    model.remove_feature(model.df.columns[0])
    model.info()
    create_model_question()
    model.create_tree()
    model.display_tree()
    print('Please enter your TESTING file')
    test = get_file()
    predictions,dataset = model.predict_file(test)
    model.display_predictions(predictions,dataset)
    actualList = dataset[model.label_name]
    create_report(model,predictions,dataset)


def call_naive(train):
    model = NaiveBayesian(verbose=False)
    model.read_csv(train)
    model.remove_feature(model.df.columns[0])
    model.info()
    create_model_question()
    model.create_model()
    print(model.get_model())
    while True:
        print('Enter name to save prior probability (.csv) or (no): ')
        try:
            input_name = str(input())
        except:
            continue
        
        if(input_name == 'no'):
            break

        if (not os.path.exists(input_name)):
            model.save(input_name)
            break
        else:
            print('File exits!')
    print('Please enter your TESTING file')
    test = get_file()
    predictions,dataset = model.predict_file(test,verbose=True)
    create_report(model,predictions,dataset)

def main():
    print('Please enter your training file')
    train = get_file()

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
        call_c45(train)
    elif(choice == 2):
        call_naive(train)
    else:
        print('Invalid choice! Exit program')

if __name__ == '__main__':
    while True:
        main()
        print('Do you want exit program? (y/n)')
        choice = str(input())
        choice = choice[0].lower() 
        if (choice =='y'):
            break
        elif (choice == 'n'):
            continue
        else:
            print('Invalid choice!')
   
# name = 'student'
# train = name + '_train.txt'
# test = name + '_test.txt'
# call_c45(train,test)
# call_naive(train,test)