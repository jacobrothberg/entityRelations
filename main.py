from CorpusReader import *
from FeatureExtractor import *


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    corpusReader = CorpusReader()
    corpusReader.read('train.txt')
    corpusReader.get_data()

    line = 1000

    print(corpusReader.data[line])

    featureExtractor = FeatureExtractor(corpusReader.data[line][0])

    print(featureExtractor.tokens)

    print(featureExtractor.lemmas)

    print(featureExtractor.pos_tags)

    print(featureExtractor.synset_dict)

    print(featureExtractor.entity_labels)