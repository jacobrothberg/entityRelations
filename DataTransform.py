import numpy as np
# from ast import literal_eval as make_tuple
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

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

    dependency_list = df['dependents']

    if (train):
        tokenizer = Tokenizer(num_words=25000, lower=True, split=' ', char_level=False)
        tokenizer.fit_on_texts(text)
        word_index = tokenizer.word_index

        sentence_seq = tokenizer.texts_to_sequences(text)
        max_length = np.max([len(i) for i in sentence_seq])

        text_seq = sequence.pad_sequences(sentence_seq, maxlen=max_length)

        a, b = np.unique(dependency_list, return_counts=True)
        a_sorted = a[np.argsort(b)[::-1]]
        major_dep = a_sorted[:33]

    else:
        sentence_seq = tokenizer.texts_to_sequences(text)
        text_seq = sequence.pad_sequences(sentence_seq, maxlen=max_length)

    embedding_size = 300
    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, embedding_size))
    for word, i in word_index.items():
        if word in w2vModel.keys():
            embedding_matrix[i] = w2vModel[word]

    dependency_list_filter = []
    for dep in dependency_list:
        if dep in major_dep:
            dependency_list_filter.append(dep)
        else:
            dependency_list_filter.append("other")

    lb = LabelBinarizer()
    dependency_list_filter = lb.fit_transform(dependency_list_filter)


    label = [labelMappings[ele] for ele in df['labels']]
    label = dense_to_one_hot(np.array(label), num_classes)

    return tokenizer,embedding_matrix, max_length, major_dep, word_index, np.asarray(text_seq), np.asarray(mut_ancestors_list), np.asarray(dependency_list_filter), np.asarray(label)
