from FeatureExtractor import *
from CorpusReader import *


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

    print(corpusReader.data[0])
    print(read_sentence("Hi my <e1>name </e1> is <e2> Poorna </e2>."))

    featureExtractor = FeatureExtractor()
    #print(corpusReader.data[0][0],corpusReader.data[0][1][0],corpusReader.data[0][1][1])
    featureExtractor.shortest_path_tree(corpusReader.data[0][0],'Thom','Radiohead')


    self.synset_list = {**self.senset_list , token:wordnet.synsets(token) for token in self.tokens}

