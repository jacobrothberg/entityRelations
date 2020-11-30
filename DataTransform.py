import numpy as np
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer




def loadGloveModel(File):
    print("Loading Glove Model")
    f = open(File,'r',encoding='utf-8')
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    print(len(gloveModel)," words loaded!")
    return gloveModel

w2vModel = loadGloveModel('glove.42B.300d.txt')


def transform(labelMappings, df, tokenizer=None, max_length=0, major_dep=None, word_index=None,train=False):
    num_classes = len(labelMappings)
    text = list()
    pos1 = [int(x) for x in df['entity1_index']]
    pos2 = [int(x) for x in df['entity2_index']]
    mut_ancestors_list = list()
    for i in range(len(df)):
        split_sentence = df.iloc[i]['text'].split(' ')
        sentence = " ".join(split_sentence[pos1[i]:pos2[i] + 1])
        text.append(sentence)
        mut_a = int(df.iloc[i]['ancestor_a'])
        mut_b = int(df.iloc[i]['ancestor_b'])
        mut_ancestors_list.append((mut_a,mut_b))


    if (train):
        tokenizer = Tokenizer(num_words=25000, lower=True, split=' ', char_level=False)
        tokenizer.fit_on_texts(text)
        word_index = tokenizer.word_index

        sentence_seq = tokenizer.texts_to_sequences(text)
        max_length = np.max([len(i) for i in sentence_seq])

        text_seq = sequence.pad_sequences(sentence_seq, maxlen=max_length)


    else:
        sentence_seq = tokenizer.texts_to_sequences(text)
        text_seq = sequence.pad_sequences(sentence_seq, maxlen=max_length)

    embedding_size = 300
    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, embedding_size))
    for word, i in word_index.items():
        if word in w2vModel.keys():
            embedding_matrix[i] = w2vModel[word]


    return tokenizer,embedding_matrix, max_length, major_dep, word_index, np.asarray(text_seq), np.asarray(mut_ancestors_list)
