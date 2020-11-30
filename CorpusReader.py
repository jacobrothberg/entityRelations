"""
Usage :
from CorpusReader import *
corpusReader = CorpusReader()
corpusReader.read('semeval_train.txt')

training_data = corpusReader.get_data()
tags = corpusReader.get_tags()
mapping = corpusReader.get_entity_relationship()

"""

import re
import pandas as pd
from ModifiedDictionary import ModifiedDictionary


class CorpusReader:
    
    def __init__(self):
        
        """
        Initialize variable to hold processed data
        """
        # stores the list of e1 and e2 separately
        self.entities = list()

        # store the indices of e1 and e2
        self.entity1_index = list()

        self.entity2_index = list()

        # stores a list of dictionaries which captures the entities, types and the relation(directionality)
        self.relationships = list()
        
        # stores the extracted text data (X)
        self.text_data = list()
        
        # stores the direction of edges
        self.edge = list()

        self.labels = list()

        self.parsed_data = pd.DataFrame()

    def extract_relations(self,labels):

        for label in labels:
            label = label.split("\n")[0]
            self.labels.append(label)
            entity_labels = label.split("(")[0]
            self.relationships.append(entity_labels.split(":"))
            if label == 'Other':
                self.edge.append("NA")
            else:
                self.edge.append(label[label.index("("):])

        return True

    def extract_text(self,sentences):

        for sentence in sentences:
            sentence = re.split('\t|\n', sentence)[0].replace('\"', "").replace(".", "").replace(",", "") \
                .replace("?", "").replace("!", "").replace(")", "").replace("(", "").replace("\'s", "") \
                .replace("\'ve", "").replace("\'t", "").replace("\'re", "").replace("\'d", "") \
                .replace("\'ll", "").replace("'", "").replace(";", "").replace(":", "")
            words = sentence.split()
            for word in words:
                if "</e1>" in word:
                    entity1 = word
                if "</e2>" in word:
                    entity2 = word

            pos1 = words.index(entity1)
            pos2 = words.index(entity2)

            self.entity1_index.append(pos1)
            self.entity2_index.append(pos2)

            results = re.findall(r'<(\w+)>(.*)</\1>', sentence)
            sentence = re.sub(r'<\w+>', "", sentence)
            sentence = re.sub(r'</\w+>', "", sentence)

            self.text_data.append(sentence)
            r = list()
            for res in results:
                word = res[1].strip()
                r.append(word)
            self.entities.append(r)
        return True

    def read(self, filename):

        labels = list()
        sentences = list()
        with open(filename, 'r') as file:
            file.seek(0)
            for lines in file.readlines():
                if lines == '\n':
                    pass
                elif lines == 'Other\n':
                    labels.append(lines)
                elif re.fullmatch(r'\w+(\W*\w+)?-\w+(\W*\w+)?\(\w+,\w+\)\s?', lines):
                    labels.append(lines)
                elif re.search(r'".*"', lines) is not None:
                    if re.search(r'<e[12]>|</e[12]>', lines) is not None:
                        sentence = lines.split("\t")[1]
                        sentences.append(sentence)
                    else:
                        pass
                else:
                    pass

        self.extract_text(sentences)
        self.extract_relations(labels)
        self.parsed_data = pd.DataFrame(data={'text': self.text_data, 'entity1_index': self.entity1_index,
                                              'entity2_index':self.entity2_index, 'entities': self.entities,
                                              'labels': self.labels,'relations': self.relationships, 'edges': self.edge},
                                        columns=['text', 'entity1_index','entity2_index', 'entities','labels','relations', 'edges'])

        return self.parsed_data

