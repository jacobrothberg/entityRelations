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
    CR = CorpusReader()
    df = CR.read('semeval_train.txt')

    """          processed_data = pd.DataFrame({'tokens': self.tokens, 'lemmas': self.lemmas,
                                       'pos_tags': self.pos_tags, 'ner_tags': self.entity_labels})
    """

    tokens = []
    lemmas = []
    pos_tags = []
    entity_labels = []
    dependency_parse = []

    for i in range(len(df)):
        featureExtractor = FeatureExtractor(df.iloc[i]['text'])
        tokens.append(featureExtractor.tokens)
        lemmas.append(featureExtractor.lemmas)
        pos_tags.append(featureExtractor.pos_tags)
        entity_labels.append(featureExtractor.entity_labels)
        dependency_parse.append(featureExtractor.parse_tree)

    processed_data = pd.DataFrame({'tokens': tokens, 'lemmas': lemmas,
                                   'pos_tags': pos_tags, 'ner_tags': entity_labels, 'parse_tree': dependency_parse})
    # print(processed_data)
    processed_data.to_csv("task2.csv")