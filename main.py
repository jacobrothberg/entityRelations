from CorpusReader import *
from FeatureExtractor import *
import pandas as pd
from DataTransform import transform
from LSTM import train_model
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np
from utils import *
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    CR_train = CorpusReader()
    dataset_train = CR_train.read('semeval_train.txt')
    # df.to_csv('test_dataset.csv')

    featureExtractor_train = FeatureExtractor()
    new_dataset_train = featureExtractor_train.getFeatures(dataset_train)

    # new_dataset.to_csv('test_features.csv')
    CR_test = CorpusReader()
    dataset_test = CR_test.read('semeval_test.txt')

    featureExtractor_test = FeatureExtractor()
    new_dataset_test = featureExtractor_test.getFeatures(dataset_test)

    labelMappings = {'Other': 0,'Instrument-Agency(e1,e2)': 1, 'Instrument-Agency(e2,e1)': 2,
                     'Cause-Effect(e1,e2)': 3, 'Cause-Effect(e2,e1)': 4,'Member-Collection(e1,e2)': 5,
                     'Member-Collection(e2,e1)' : 6,'Entity-Destination(e1,e2)': 7, 'Entity-Destination(e2,e1)': 8,
                     'Content-Container(e1,e2)': 9, 'Content-Container(e2,e1)': 10,'Message-Topic(e1,e2)': 11,
                     'Message-Topic(e2,e1)': 12, 'Product-Producer(e1,e2)': 13,'Product-Producer(e2,e1)': 14,
                     'Entity-Origin(e1,e2)': 15, 'Entity-Origin(e2,e1)': 16,'Component-Whole(e1,e2)': 17,
                     'Component-Whole(e2,e1)': 18}

    classToLabel = {0: ('Other','NA'),1: ('Instrument-Agency','(e1,e2)'), 2: ('Instrument-Agency','(e2,e1)'),
                    3: ('Cause-Effect','(e1,e2)'),4: ('Cause-Effect','(e2,e1)'),5: ('Member-Collection','(e1,e2)'),
                    6: ('Member-Collection','(e2,e1)'),7: ('Entity-Destination','(e1,e2)'),8: ('Entity-Destination','(e2,e1)'),
                    9: ('Content-Container','(e1,e2)'),10: ('Content-Container','(e2,e1)'),11: ('Message-Topic','(e1,e2)'),
                    12: ('Message-Topic','(e2,e1)'),13: ('Product-Producer','(e1,e2)'),14: ('Product-Producer','(e2,e1)'),
                    15: ('Entity-Origin','(e1,e2)'),16: ('Entity-Origin','(e2,e1)'),17: ('Component-Whole','(e1,e2)'),
                    18: ('Component-Whole','(e2,e1)')}

    label_train = [labelMappings[ele] for ele in dataset_train['labels']]
    label_train = dense_to_one_hot(np.array(label_train), len(labelMappings))
    x_label = np.asarray(label_train)

    label_test = [labelMappings[ele] for ele in dataset_test['labels']]
    label_test = dense_to_one_hot(np.array(label_test), len(labelMappings))
    y_label = np.asarray(label_test)

    dependency_list = new_dataset_train['dependents']
    a, b = np.unique(dependency_list, return_counts=True)
    a_sorted = a[np.argsort(b)[::-1]]
    major_dep = a_sorted[:33]
    major_deps = {}
    i = 0
    for j in range(33):
        major_deps[major_dep[j]] = i
        i += 1

    train_dependency_list = new_dataset_train['dependents']
    test_dependency_list = new_dataset_test['dependents']
    x_dependency_list_filter = dependency_encoder(major_deps,train_dependency_list)
    y_dependency_list_filter = dependency_encoder(major_deps,test_dependency_list)
    tokenizer,embedding_matrix, max_length, major_dep, word_index, x_text_seq, x_mut_ancestors_list = transform(labelMappings,new_dataset_train,train = True)
    (_,_,_,_,_,y_text_seq,y_mut_ancestors_list) = transform(labelMappings,new_dataset_test,tokenizer,max_length,major_dep,word_index,train=False)



    model, history = train_model(x_text_seq, x_mut_ancestors_list,
                                  x_dependency_list_filter, x_label,
                                  y_text_seq, y_mut_ancestors_list,
                                  y_dependency_list_filter, y_label,
                                  embedding_matrix,
                                  max_length,
                                  len(word_index) + 1)
    model.save('LSTM')
    prediction = model.predict(
        [y_text_seq,y_mut_ancestors_list,y_dependency_list_filter],
        batch_size=1000)
    class_pred = np.argmax(prediction, axis=1)
    class_true = np.argmax(y_label, axis=1)
    conf = confusion_matrix(class_true, class_pred)
    precision,recall,fscore,support = precision_recall_fscore_support(class_true, class_pred, average='macro')
    print("precision: ",precision*100,"\nrecall: ",recall*100,"\nfscore: ",fscore*100)

    y_true = [classToLabel[x] for x in class_true]
    y_pred = [classToLabel[x] for x in class_pred]

    cor_rel_cor_edge, cor_rel_wr_edge = 0, 0
    for i in range(len(y_true)):
        if y_true[i][0] == y_pred[i][0]:
            if y_true[i][1] == y_pred[i][1]:
                cor_rel_cor_edge += 1
            else:
                cor_rel_wr_edge += 1

    print("Correct relation,correct edges: ",100 * cor_rel_cor_edge/ len(y_true))
    print("Correct relation, wrong edge: ", 100 * cor_rel_wr_edge / len(y_true))

