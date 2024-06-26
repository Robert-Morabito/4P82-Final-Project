import gensim  # NOTE: For gensim to work you need scipy v1.12 or lower as it needs triu, v1.13 won't work
import numpy as np
from scipy.spatial import distance

# Global model variable
w2v_model = None


def train_word2vec(dataset):
    """
    Trains a word2vec model given a dataset of words
    :param dataset: Training data
    """
    global w2v_model
    w2v_model = gensim.models.Word2Vec(dataset, vector_size=10, window=5, min_count=1, workers=16, epochs=20)


def w2v_vectorize(dataset):
    """
    Vectorizes an entire dataset of sentence with individual words
    :param dataset: Dataset to be vectorized
    :return: Vectorized dataset
    """
    return np.array([[w2v_model.wv[word] for word in sentence if word in w2v_model.wv] for sentence in dataset])


def unvectorize(vectors):
    """
    Finds the most similar word from a pretrained model to the vectorized word provided
    :param vectors: Vectorized word
    :return: Word most similar to vector
    """
    words = []
    for vector in vectors:
        max_sim = -1
        closest_word = None
        for word in w2v_model.wv.key_to_index:
            sim = 1 - distance.cosine(vector, w2v_model.wv[word])
            if sim > max_sim:
                max_sim = sim
                closest_word = word
        words.append(closest_word)
    return words
