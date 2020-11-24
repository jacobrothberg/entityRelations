from CorpusReader import *
from FeatureExtractor import *
import pandas as pd


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
    corpusReader = CorpusReader()
    corpusReader.read('semeval_train.txt')
    data = corpusReader.get_data()

    
    line = 16016

    print(corpusReader.data[line])

    featureExtractor = FeatureExtractor(corpusReader.data[line][0])

    print(featureExtractor.tokens)

    print(featureExtractor.lemmas)

    print(featureExtractor.pos_tags)

    print(featureExtractor.synset_dict)

    print("Entity labels: ", featureExtractor.entity_labels)

    """

    CR = CorpusReader()
    df = CR.read('semeval_train.txt')
    print(df)

