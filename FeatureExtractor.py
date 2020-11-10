import nltk
import spacy
import networkx as nx
nlp = spacy.load("en_core_web_sm")

class FeatureExtractor:

    def __init__(self):

        self.tokens = list()

    def tokenize(self, sentence):
        self.tokens = nltk.word_tokenize(sentence)

    def get_tokens(self, sentence):

        return self.tokens

    def shortest_path_tree(self,sentence,entity1,entity2):
        edges = []
        doc = nlp(sentence)
        for token in doc :
            for child in token.children:
                edges.append(('{0}'.format(token.lower_),'{0}'.format(child.lower_)))

        print(edges)
        #graph = nx.Graph(edges)
        #print(nx.shortest_path_length(graph, source=entity1, target=entity2))
        #print(nx.shortest_path(graph, source=entity1, target=entity2))




