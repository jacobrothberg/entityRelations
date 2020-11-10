"""
Usage :
from CorpusReader import *
corpusReader = CorpusReader()
corpusReader.read('train.txt')

training_data = corpusReader.get_data()
tags = corpusReader.get_tags()
mapping = corpusReader.get_entity_relationship()

"""

import re
from ModifiedDictionary import ModifiedDictionary


class CorpusReader:
    
    def __init__(self):
        
        """
        Initialize variable to hold processed data
        """
        # stores the list of e1 and e2 separately
        self.entityList = ModifiedDictionary()
        
        # stores the list of entity type separately
        self.entityTypeList = ModifiedDictionary()
        
        # stores a list of dictionaries which captures the entities, types and the relation(directionality)
        self.entity_relationships = list()
        
        # stores the extracted text data (X)
        self.text = list()
        
        # stores the entity type and relation(directionality) (Y)
        self.labels = set()
        
        # stores the unique tags present in the corpus
        self.tags = list()
        
        # stores e1 and e2 for each sentence in the corpus
        self.entities = list()
        
        # stores entity tags, entity type and relation(directionality)
        self.relations = list()
        
        self.data = list()
        
    def get_entity_list(self):

        return self.entityList
    
    def get_entity_type_list(self):
        
        return self.entityTypeList
    
    def get_entity_relationship(self):
        
        return self.entity_relationships

    def get_unique_relations(self):
        
        return self.labels
    
    def get_tags(self):
        
        return self.tags
    
    def get_entities(self):
        
        return self.entities
    
    def get_relations(self):
        
        return self.relations    
    
    def get_text(self):
        
        return self.text
        
    def capture_entities(self, sentence):
        """
        returns annotated entities from the corpus
        """
        
        tag = list()
        entity = dict()
        results = re.findall(r'<(\w+)>(.*)</\1>', sentence)
        new_sentence = re.sub(r'<\w+>', "", sentence)
        new_sentence = re.sub(r'</\w+>', "", new_sentence)
        self.text.append(new_sentence.strip())
        for captures in results:
            entity = {**entity, captures[0]: captures[1].strip()}
            if captures[0] not in tag:
                tag.append(captures[0])
                
        return entity, tuple(tag)
    
    def capture_relations(self, label, tags):
        """
        returns entity type and relation from the corpus
        """
        
        relation = dict()
        
        if label == 'no_relation':
            self.labels.add(label)
            for t in tags:
                relation = {**relation, t: 'NA'}
            relation = {**relation, 'edge': 'NA'}
        else:
            relation_type = label.split("(")[0].split(":")
            edge = re.findall(r'\((\w+),(\w+)\)', label)[0]
            self.labels.add((tuple(relation_type)))
            for t in tags:
                relation = {**relation, t: relation_type[edge.index(t)]}
            relation = {**relation, 'edge': edge}
        
        return relation

    @staticmethod
    def map_entity_relations(ce, cr, tag, n):
        """
        returns a dictionary with mapping of entity tags(e1,e2), the named entities, the relation they hold 
        and direction of relation
        """
        
        entity_relations = []
        for i in range(n):
            er = {'entities': {}, 'relation': ()}
            for t in tag:
                er['entities'] = {**er['entities'], t: {'entity_name': ce[i][t], 'entity_type': cr[i][t]}}
            er['relation'] = cr[i]['edge']
            entity_relations.append(er)
            
        return entity_relations     
        
    def read(self, filename):
        """
        reads the corpus to represent information in a convenient fashion
        """
        
        with open(filename, 'r') as file:
            file.seek(0)
            for lines in file.readlines():
                
                if lines != '\n':
                    line = lines.split("\t")
                    if len(line) >= 2:
                        sentence = line[1].split('"')[1]
                        (ce, self.tags) = self.capture_entities(sentence)
                        self.entities.append(ce)
                        for tag1 in self.tags:
                            self.entityList[tag1].append(ce[tag1])
                    else:
                        label = line[0].split("\n")
                        cr = self.capture_relations(label[0], self.tags)
                        self.relations.append(cr)
                        for tag in self.tags:
                            self.entityTypeList[tag].append(cr[tag])
                    
        self.entity_relationships = self.map_entity_relations(self.entities, self.relations,
                                                              self.tags, len(self.entities))
    
        return True

    def get_data(self):
        
        for i in range(len(self.text)):
            self.data.append((self.text[i], (self.entities[i][self.tags[0]], self.entities[i][self.tags[1]]),
                              (self.relations[i][self.tags[0]], self.relations[i][self.tags[1]]),
                              self.relations[i]['edge']))

        return self.data
