import pytest
from classifier import utils
from classifier import model
from classifier import preprocessing
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Constants
CLASSIFIER_ARGS = {
    "sequence_length": 10,
    "input_dim": 18,
    "label_dim": 1,
    "embeddings": None,
    "vector_dim": 50
}
DATA = [
        "Lorem ipsum dolor sit amet",
        "hello consectetur adipiscing elit. Curabitur congue",
        "world consequat lorem a cursus. Aliquam sollicitudin"
    ]


@pytest.fixture
def classifier():
    return model.TextClassifier(**CLASSIFIER_ARGS)


@pytest.fixture
def text_transformer():
    transformer = preprocessing.TextFormatting(top_words=100000,
                                               max_len=10)
    transformer.fit_transform(DATA)
    return transformer


@pytest.fixture
def training_set():
    return pd.DataFrame({"text": DATA}), pd.Series([0, 1, 0])


def test_classifier_build(classifier):
    m = classifier.build(**CLASSIFIER_ARGS)
    print(m.summary())


def test_fit(classifier, text_transformer, training_set):
    preprocessed_text = text_transformer.fit_transform(training_set[0].loc[:, "text"].tolist())
    label_encoder = LabelEncoder()
    training_labels = label_encoder.fit_transform(training_set[1])
    classifier.fit(preprocessed_text, training_labels, batch_size=1)


def test_fit_predict(classifier, text_transformer, training_set):
    preprocessed_text = text_transformer.fit_transform(training_set[0].loc[:, "text"].tolist())
    label_encoder = LabelEncoder()
    training_labels = label_encoder.fit_transform(training_set[1])
    classifier.fit(preprocessed_text, training_labels, batch_size=1)
    assert len(classifier.predict(text_transformer.transform([["hello world"]]))) == 1
