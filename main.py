from C45 import C45
from ConfussionMatrix import Report,ConfussionMatrix
import numpy as np
import os

train = 'student_train.txt'
test = 'student_test.txt'

def call_c45(train,test):
    model = C45()
    model.read_csv(train)
    model.remove_feature(model.df.columns[0])
    model.info()
    print('============================================')
    model.create_tree()
    model.display_tree()
    predictions,dataset = model.predict_file(test)
    model.display_predictions(predictions,dataset)
    actualList = dataset[model.label_name]
    predictionList = np.array(predictions)
    labels = model.df[model.label_name].unique()
    report = Report()
    newList = report.create_cm_list(actualList,predictionList,labels)
    report.create_report(newList,labels)

while True:
    print('Enter train file (.txt): ')
    train = str(input())
    if (os.path.exists(train)):
        break
    else:
        print('File does not exits')

while True:
    print('Enter test file (.txt): ')
    test = str(input())
    if (os.path.exists(test)):
        break
    else:
        print('File does not exits')

call_c45(train,test)