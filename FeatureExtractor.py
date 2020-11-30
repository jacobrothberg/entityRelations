import spacy
from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import pandas as pd
wnl = WordNetLemmatizer()

nlp = spacy.load("en_core_web_sm")


class FeatureExtractor:
    def __init__(self):

        self.entities = list()

        self.tokens = list()

        self.lemmas = list()

        self.pos_tags = list()

        self.synset_list = list()

        self.ner_tags = list()

        self.ancestor_a = list()

        self.ancestor_b = list()

        self.parse_tree = list()

        self.dependents = list()

        self.both_synsets = list()

    def Entities(self, entities):


        for entity in entities:
            ent = [wnl.lemmatize(x, pos='v') for x in entity]
            self.entities.append(ent)

        # print("entities: ",len(self.entities))
        return True

    def Tokenize(self, texts):

        for text in texts:
            self.tokens.append(word_tokenize(text))

        # print("tokens: ",len(self.tokens))
        return True

    def Lemmatize(self, tokens):

        for token in tokens:
            lemma = [wnl.lemmatize(x, pos='v') for x in token]
            self.lemmas.append(lemma)

        # print("lemmas: ",len(self.lemmas))
        return True

    def PosTags(self, tokens):

        for token in tokens:
            self.pos_tags.append(pos_tag(token))

        # print("pos_tags: ",len(self.pos_tags))
        return True

    def Synsets(self, tokens):

        for token in tokens:
            synset = {}
            for t in token:
                synset = {**synset, t: wn.synsets(t)}
            self.synset_list.append(synset)

        # print("synsets: ",len(self.synset_list))
        return True

    def nltkFeatures(self, texts, entity1_indices,entity2_indices, entities):

        for (text, ent1_index,ent2_index, entity) in zip(texts, entity1_indices,entity2_indices, entities):
            doc = nlp(text)
            entity_labels = {}
            for ent in doc.ents:
                entity_labels = {**entity_labels, ent.text: ent.label_}
            self.ner_tags.append(entity_labels)


            both_ancestors = []
            both_dep = ""
            for pos, token in enumerate(doc):
                if pos == ent1_index or pos == ent2_index:
                    anc = [i.text for i in token.ancestors]
                    both_ancestors.append(anc)
                    both_dep += token.dep_
            self.dependents.append(both_dep)

            e1_is_ancestor_e2 = 0
            e2_is_ancestor_e1 = 0

            if entity[0] in both_ancestors[1]:
                e1_is_ancestor_e2 = 1
            elif entity[1] in both_ancestors[0]:
                e2_is_ancestor_e1 = 1

            self.ancestor_a.append(e1_is_ancestor_e2)
            self.ancestor_b.append(e2_is_ancestor_e1)


            tree = list()
            for chunk in doc.noun_chunks:
                tree.append((chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text))
            self.parse_tree.append(tree)

        # print("ner_tags: ",len(self.ner_tags))
        # print("ancestors: ",len(self.ancestor_a), len(self.ancestor_b))
        # print("dependents: ",len(self.dependents))
        # print("parse_tree: ",len(self.parse_tree))
        return True

    def wordnetFeatures(self):

        for i in range(len(self.entities)):
            entity = self.entities[i]
            for ent in entity:
                synsets = wn.synsets(ent)
                list_hypernyms = []
                list_hyponyms = []
                list_meronyms = []
                list_holonyms = []
                synset_dict = {}
                for synset in synsets:
                    for hypernym in synset.hypernyms():
                        for name in hypernym.lemma_names():
                            list_hypernyms.append(name)
                    for hyponym in synset.hyponyms():
                        for name in hyponym.lemma_names():
                            list_hyponyms.append(name)
                    for meronym in synset.member_meronyms():
                        for name in meronym.lemma_names():
                            list_meronyms.append(name)
                    for holonym in synset.member_holonyms():
                        for name in holonym.lemma_names():
                            list_holonyms.append(name)

                synset_dict[ent] = {'hypernyms': list_hypernyms, 'hyponyms': list_hyponyms, 'meronyms': list_meronyms,
                                    'holonyms': list_holonyms}

            self.both_synsets.append(synset_dict)

        # print("both synsets: ",len(self.both_synsets))
        return True

    def getFeatures(self, df):

        self.Entities(df['entities'])
        self.Tokenize(df['text'])
        self.Lemmatize(self.tokens)
        self.PosTags(self.tokens)
        self.Synsets(self.tokens)
        self.nltkFeatures(df['text'], df['entity1_index'],df['entity2_index'], df['entities'])
        self.wordnetFeatures()


        features = pd.DataFrame(
            data={'text': df['text'], 'entity1_index': df['entity1_index'],'entity2_index': df['entity2_index'], 'entities': self.entities,
                  'tokens': self.tokens, 'lemmas': self.lemmas, 'pos_tags': self.pos_tags, 'ner_tags': self.ner_tags,
                  'ancestor_a': self.ancestor_a,'ancestor_b': self.ancestor_b,
                  'parse_tree': self.parse_tree, 'dependents': self.dependents, 'both_synsets': self.both_synsets},
            columns=['text', 'entity1_index','entity2_index', 'entities', 'tokens', 'lemmas', 'pos_tags', 'ner_tags',
                     'ancestor_a','ancestor_b', 'parse_tree', 'dependents', 'both_synsets'])

        # features['entity_indices'] = features['entity_indices'].astype(('int64','int64'))
        return features



