import re
import pandas as pd
import numpy as np

def read_sentence(sentence):
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

    results = re.findall(r'<(\w+)>(.*)</\1>', sentence)
    sentence = re.sub(r'<\w+>', "", sentence)
    new_sent = re.sub(r'</\w+>', "", sentence)
    entities = []
    ent = [x[1] for x in results]
    entities.append(ent)
    parsed_sentence = pd.DataFrame(
        data={'text': new_sent.strip(), 'entity1_index': pos1, 'entity2_index': pos2, 'entities': (entities)},
        columns=['text', 'entity1_index', 'entity2_index', 'entities'])
    return parsed_sentence

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def dependency_encoder(major_deps, dependency_list):
    dependency_filter_list = np.zeros((len(dependency_list), len(major_deps) + 1))
    for i in range(len(dependency_list)):
        if dependency_list[i] not in major_deps.keys():
            j = 33
        else:
            j = major_deps[dependency_list[i]]

        dependency_filter_list[i][j] = 1

    return dependency_filter_list