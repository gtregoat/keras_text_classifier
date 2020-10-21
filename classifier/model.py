from tensorflow.keras import layers
from tensorflow.keras import Model
import numpy as np


class TextClassifier:
    def __init__(self,
                 sequence_length: int,
                 input_dim: int,
                 label_dim: int,
                 embeddings=None,
                 vector_dim=None):
        """
        Wrapper to define a Keras text classification model.

        :param sequence_length: Integer, maximum number of words to keep in a sequence
        :param input_dim: Integer. Size of the vocabulary, i.e. maximum integer index + 1.
        :param label_dim: Integer, number of classes
        :param embeddings: Numpy array, matrix of embeddings
        :param vector_dim: int, output dimension of the embedding layer. If embeddings are provided, corresponds
            to the dimension of the pre-trained embeddings.
        """
        self.model = self.build(
            sequence_length=sequence_length,
            input_dim=input_dim,
            label_dim=label_dim,
            embeddings=embeddings,
            vector_dim=vector_dim)

    def fit(self, x: np.array, y: np.array, **kwargs) -> None:
        """
        Fits the model to the training data x and its associated labels y. The model will
        be recorded in self.model.
        :param x: numpy array, training data (must be integers).
        :param y: numpy array, training labels (one-hot encoded)
        :param kwargs: any arguments to pass to Keras' fit method, e.g. epochs, batch_size.
        """
        self.model.fit(x, y, **kwargs)

    def predict(self, x):
        """
        Generates predictions using the previously trained model with the fit method.
        :param x: numpy array, prediction data (must be integers).
        :return: numpy array, (x_samples, n_classes)
        """
        return self.model.predict(x)

    @staticmethod
    def build(sequence_length: int,
              input_dim: int,
              label_dim: int,
              embeddings: np.array = None,
              vector_dim: int = None) -> Model:
        """Builds and returns the Keras classification model.
        The model is a CNN LSTM with the attention mechanism. It can integrate pre-trained word embeddings.
        Loss is categorical cross-entropy and the Adam algorithm is used to minimize it. The accuracy on
        the training data is recorded at each epoch.

        :param sequence_length: int, refers to the maximum number of words in the input. If x was the training data,
            this would be x.shape[1].
        :param input_dim: int, number of words in the embeddings, i.e. the highest word id after tokenizing words.
        :param label_dim: int, number of classes.
        :param embeddings: np.array, pre-trained word embeddings (e.g. GloVe). If set to None, the embeddings
            will be trained, otherwise they will remain fixed.
        :param vector_dim: int, dimension of the word embeddings. If embeddings are provided, this will be
            automatically set to embeddings.shape[1]
        :return: Keras model
        """
        inputs = layers.Input(shape=(sequence_length,))
        # Pre-trained embeddings
        if embeddings is not None:
            vector_dim = embeddings.shape[1]
            embedded = layers.Embedding(
                input_dim=input_dim,
                output_dim=vector_dim,
                weights=[embeddings],
                input_length=sequence_length,
                trainable=False)(inputs)  # The embedding weights will remain fixed as there isn't much data per class
        else:
            assert vector_dim is not None, "If not using pretrained embeddings, an embedding dimension has " \
                                           "to be provided."
            embedded = layers.Embedding(
                input_dim=input_dim,
                output_dim=vector_dim,
                input_length=sequence_length,
                trainable=True)(inputs)  # In this case there are no pre-trained embeddings so training is required
        conv = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(embedded)
        pool = layers.MaxPooling1D(pool_size=2)(conv)
        pool = layers.BatchNormalization()(pool)
        recurrent = layers.LSTM(units=100, return_sequences=True)(pool)
        # compute importance for each step (attention mechanism)
        attention = layers.Dense(1, activation='tanh')(recurrent)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(100)(attention)
        attention = layers.Permute([2, 1])(attention)
        # Complete text representation
        representation = layers.Multiply()([recurrent, attention])
        representation = layers.Flatten()(representation)

        # Classify
        classification = layers.Dense(500, activation="relu")(representation)
        classification = layers.Dropout(0.4)(classification)
        classification = layers.BatchNormalization()(classification)
        classification = layers.Dense(200, activation="relu")(classification)
        classification = layers.Dropout(0.4)(classification)
        classification = layers.BatchNormalization()(classification)
        classification = layers.Dense(100, activation="relu")(classification)
        classification = layers.Dropout(0.4)(classification)
        classification = layers.Dense(10, activation="relu")(classification)
        classification = layers.Dense(label_dim, activation="softmax")(classification)

        # Create the model
        model = Model([inputs], classification)

        # Compile
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
        return model
