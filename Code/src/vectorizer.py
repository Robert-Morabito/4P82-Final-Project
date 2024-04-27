import gensim  # NOTE: For gensim to work you need scipy v1.12 or lower as it needs triu, v1.13 won't work
from scipy.spatial import distance

# Global model variable
model = None

def train_word2vec(dataset):
    global model
    model = gensim.models.Word2Vec(dataset, vector_size=10, window=5, min_count=1, workers=4)

def vectorize(dataset):
    """
    Vectorizes an entire dataset of sentence with individual words
    :param dataset: Dataset to be vectorized
    :return: Vectorized dataset
    """
    return [[model.wv[word] for word in sentence] for sentence in dataset]


def unvectorize(vectors):
    """
    Finds the most similar word from a pretrained model to the vectorized word provided
    :param vector: Vectorized word
    :return: Word most similar to vector
    """
    words = []
    for vector in vectors:
        max_sim = -1
        closest_word = None
        for word in model.wv.key_to_index:
            sim = 1 - distance.cosine(vector, model.wv[word])
            if sim > max_sim:
                max_sim = sim
                closest_word = word
        words.append(closest_word)
    return words
