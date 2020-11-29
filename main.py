from CorpusReader import *
from FeatureExtractor import *
import pandas as pd
from DataTransform import transform
from LSTM import train_model
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np
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

    i = 0
    labelMappings = {}
    for label in dataset_train['labels']:
        if label not in labelMappings.keys():
            labelMappings[label] = i
            i += 1

    print(len(labelMappings))
    print(labelMappings)

    tokenizer,embedding_matrix, max_length, major_dep, word_index, x_text_seq, x_mut_ancestors_list, x_dependency_list_filter, x_label = transform(labelMappings,new_dataset_train,train = True)
    (_,_,_,_,_,y_text_seq,y_mut_ancestors_list,y_dependency_list_filter,y_label) = transform(labelMappings,new_dataset_test,tokenizer,max_length,major_dep,word_index,train=False)

    print("X: ", x_text_seq.shape,x_mut_ancestors_list.shape,x_dependency_list_filter.shape,x_label.shape)
    print("Y: ", y_text_seq.shape, y_mut_ancestors_list.shape, y_dependency_list_filter.shape, y_label.shape)

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
    precision_recall_fscore = precision_recall_fscore_support(class_true, class_pred, average='macro')