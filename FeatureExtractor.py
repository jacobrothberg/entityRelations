import nltk

class FeatureExtractor:

    def __init__(self):

        self.tokens = list()

    def tokenize(self, sentence):
        self.tokens = nltk.word_tokenize(sentence)

    def get_tokens(self, sentence):

        return self.tokens