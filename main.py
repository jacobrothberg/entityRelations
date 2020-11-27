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
    df = CR.read('semeval_test.txt')
    df.to_csv('test_dataset.csv')


    """          processed_data = pd.DataFrame({'tokens': self.tokens, 'lemmas': self.lemmas,
                                       'pos_tags': self.pos_tags, 'ner_tags': self.entity_labels})
    """

    tokens = []
    lemmas = []
    pos_tags = []
    entity_labels = []
    dependency_parse = []
    sentence = []
    synsets = []
    ancestors = []
    deps = []
    features = []
    entities = []
    unique_labels = dict()
    i = 0
    for label in df['labels']:
        if label not in unique_labels.keys():
            unique_labels[label] = i
            i += 1



    for i in range(len(df)):
        featureExtractor = FeatureExtractor(df.iloc[i]['text'], df.iloc[i]['entity_indices'], df.iloc[i]['entities'])
        sentence.append(featureExtractor.sentence)
        tokens.append(featureExtractor.tokens)
        lemmas.append(featureExtractor.lemmas)
        pos_tags.append(featureExtractor.pos_tags)
        entity_labels.append(featureExtractor.entity_labels)
        dependency_parse.append(featureExtractor.parse_tree)
        synsets.append(featureExtractor.both_synsets)
        ancestors.append(featureExtractor.ancestors)
        deps.append(featureExtractor.deps)
        entities.append(featureExtractor.entities)
        features.append(featureExtractor.features)

    processed_data = pd.DataFrame({'text': sentence,'entities': entities,'relations': df['relations'], 'tokens': tokens, 'lemmas': lemmas,
                                      'pos_tags': pos_tags, 'ner_tags': entity_labels, 'parse_tree': dependency_parse, 'synsets': synsets, 'ancestors': ancestors, 'deps': deps,'features':features})
    processed_data.to_csv("task2_test.csv")

    # print(unique_labels)
    # print(len(unique_labels))