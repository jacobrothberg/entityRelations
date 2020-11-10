from nltk import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

wnl = WordNetLemmatizer()


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
