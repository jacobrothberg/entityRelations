# meaningless comment 1
import spacy
from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import geonamescache
nlp = spacy.load("en_core_web_sm")
from nltk.util import ngrams
wnl = WordNetLemmatizer()

gc = geonamescache.GeonamesCache()
countries = gc.get_countries()
cities = gc.get_cities()
states = gc.get_us_states()

def ngramToList(ngram):
    words = []
    for c in ngram:
        word = ""
        for w in c:
            word += w
        words.append(word)
    return words
def extract_gpe_type(gpe_type, entity):

    if isinstance(gpe_type, dict):
        for key, value in gpe_type.items():
            if key == entity:
                yield value

            if isinstance(value, (dict, list)):
                yield from extract_gpe_type(value, entity)

    elif isinstance(gpe_type, list):
        for gpe in gpe_type:
            yield from extract_gpe_type(gpe, entity)


cities_list = [*extract_gpe_type(cities, 'name')]
countries_list = [*extract_gpe_type(countries, 'name')]
states_list = [*extract_gpe_type(states, 'name')]


class FeatureExtractor:

    def __init__(self, sentence, entity_indices, entities):

        self.sentence = sentence

        self.entities = [wnl.lemmatize(entity, pos = 'v') for entity in entities]

        self.tokens = word_tokenize(self.sentence)

        self.lemmas = [wnl.lemmatize(token, pos='v') for token in self.tokens]

        self.pos_tags = pos_tag(self.tokens)

        self.synset_dict = dict()
        for token in self.tokens:
            self.synset_dict = {**self.synset_dict, token: wn.synsets(token)}

        self.entity_labels = dict()
        doc = nlp(self.sentence)
        for ent in doc.ents:
            if ent.label_ == 'GPE':
                if ent.text in countries_list:
                    self.entity_labels = {**self.entity_labels, ent.text: "Country"}
                elif ent.text in cities_list:
                    self.entity_labels = {**self.entity_labels, ent.text: "City"}
                elif ent.text in states_list:
                    self.entity_labels = {**self.entity_labels, ent.text: "State"}
                else:
                    self.entity_labels = {**self.entity_labels, ent.text: ent.label_}
            else:
                self.entity_labels = {**self.entity_labels, ent.text: ent.label_}

        both_ancestors = []
        both_dep = ""
        for pos, token in enumerate(doc):
            if pos in entity_indices:
                ancestors = [i.text for i in token.ancestors]
                both_ancestors.append(ancestors)
                both_dep += token.dep_

        e1_is_ancestor_e2 = 0
        e2_is_ancestor_e1 = 0
        if entities[0] in both_ancestors[1]:
            e1_is_ancestor_e2 = 1
        elif entities[1] in both_ancestors[0]:
            e2_is_ancestor_e1 = 1

        self.ancestors = ((e1_is_ancestor_e2, e2_is_ancestor_e1))
        self.deps = both_dep

        for chunk in doc.noun_chunks:
            self.parse_tree = (chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text)

        self.both_synsets = {}

        for entity in self.entities:
            synsets = wn.synsets(entity)
            list_hypernyms = []
            list_hyponyms = []
            list_meronyms = []
            list_holonyms = []

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
            self.both_synsets[entity] = {'hypernyms': list_hypernyms, 'hyponyms': list_hyponyms, 'meronyms': list_meronyms, 'holonyms': list_holonyms}

            self.features = list()
            feature = []
            unigrams = self.tokens
            feature.extend(unigrams)
            feature.extend(ngramToList(ngrams(unigrams, 2)))
            feature.extend(ngramToList(ngrams(unigrams, 2)))
            feature.extend(ngramToList(ngrams(unigrams, 3)))
            feature.extend(ngramToList(ngrams(unigrams, 3)))
            feature.extend(ngramToList(ngrams(unigrams, 3)))
            feature = ' '.join(feature)
            self.features.append(feature)