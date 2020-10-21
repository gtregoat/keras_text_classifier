import pandas as pd
import numpy as np
import os
from typing import Tuple


def load_data(phase: str, store_path: str) -> pd.DataFrame:
    """Loads the data provided for training and testing. If the data hasn't been
    downloaded, it will be downloaded in store_path/dataset_name (e.g. data/trainSet.csv).
    If it has already beed downloaded in the directory 'store_path', it will be read from there
    directly, avoiding the need for another download.

    :param phase: str, "fit" for the training data, "predict" for the test set.
    :param store_path: path to the directory where to download the data, and where to reload it from the next time.
    :return: training data as a pandas DataFrame. The training set will have two columns: ['text', 'label'] when
        the test set will only have ['text'].
    """
    if phase == "fit":
        path, save_needed = resolve_path("trainSet.csv", store_path)
        df = pd.read_csv(path, names=["text", "label"])
    elif phase == "predict":
        path, save_needed = resolve_path("candidateTestSet.txt", store_path)
        df = pd.read_csv(path, names=["text"])
    else:
        raise Exception("Available phases: fit, predict")

    if save_needed:
        df.to_csv(os.path.join(store_path, os.path.basename(path)),
                  index=False,
                  header=False)

    return df


def resolve_path(filename: str, store_path: str) -> Tuple[str, bool]:
    """Resolves if downloading data is needed. This not only speeds up the process but avoids being blocked
    from the server.

    :param filename: str, file to open, either trainSet.csv or candidateTestSet.txt.
    :param store_path: path to the directory where to save data.
    :return: (path, save_needed). path is a string, either a path to the download or to the local
        storage. save_needed is a boolean that will be used by the load_data function to know
        whether to save the data locally or not.
    """
    save_needed = False
    if filename in os.listdir(store_path):
        path = os.path.join(store_path, filename)
    else:
        path = f"https://s3-eu-west-1.amazonaws.com/adthena-ds-test/{filename}"
        save_needed = True
    return path, save_needed


def load_glove_embeddings(path: str, embedding_dim: int, word_index: np.array) -> np.array:
    """Loads GloVe embeddings for weight initialization in the embedding layer.
    The goal is to use this to load the GloVe word embeddings to pass to the model in model.py. That model
    will use them in the Keras Embedding layer.

    This needs to be run after tokenizing the text with the tokenizer from Keras. That tokenizer
    will have word_index as an attribute, that needs to be passed to this function so it can make
    a correspondence between the encoded id by the tokenizer and the GloVe word embedding.

    :param path: path to glove embedding matrix.
    :param embedding_dim: dimension of the GloVe embedding matrix.
    :param word_index: the word index obtained using tokenizer.word_index
    :return: the embedding matrix that can be used to initialize the embedding layer for
      NLP in Keras. e.g.:
      model.add(Embedding(input_dim=len(word_index) + 1, output_dim=dim, weights=[embeddings],
      input_length=max_len, trainable=True))
    """
    embedding_index = {}
    f = open(path, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    f.close()
    return create_embedding_matrix(word_index=word_index, embedding_index=embedding_index, embedding_dim=embedding_dim)


def create_embedding_matrix(word_index, embedding_index, embedding_dim):
    """
    Creates the embedding matrix that can be used to initialize the embedding layer in Keras (e.g. initialize it with
    pre-trained GloVe or FastText word vectors). The vectors must have been read first.

    :param word_index: the word index obtained using tokenizer.word_index
    :param embedding_index: embeddings index computed in load_glove_embeddings
    :param embedding_dim: int, dimension of the word embeddings
    :return:
    """
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
