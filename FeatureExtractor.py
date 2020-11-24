## meaningless comment 1
import spacy
from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import geonamescache
nlp = spacy.load("en_core_web_sm")

wnl = WordNetLemmatizer()

gc = geonamescache.GeonamesCache()
countries = gc.get_countries()
cities = gc.get_cities()
states = gc.get_us_states()


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

    def __init__(self, sentence):

        self.sentence = sentence

        self.tokens = word_tokenize(self.sentence)

        self.lemmas = [wnl.lemmatize(token) for token in self.tokens]

        self.pos_tags = pos_tag(self.tokens)

        self.dict_of_synsets = {}

        self.synset_dict = dict()
        for token in self.tokens:
            self.synset_dict = {**self.synset_dict, token : wordnet.synsets(token)}

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
                    self.entity_labels = {**self.entity_labels, ent.text : ent.label_}
            else:
                self.entity_labels = {**self.entity_labels, ent.text: ent.label_}

        for chunk in doc.noun_chunks:
            self.parse_tree = (chunk.text,chunk.root.text, chunk.root.dep_, chunk.root.head.text)