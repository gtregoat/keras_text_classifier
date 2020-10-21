import pytest
import classifier as clf
import pandas as pd

DATA = [
        "Lorem ipsum dolor sit amet",
        "hello consectetur adipiscing elit. Curabitur congue",
        "world consequat lorem a cursus. Aliquam sollicitudin"
    ]


@pytest.fixture
def pipeline():
    return clf.TextClassificationPipeline(sequence_length=15,
                                          embeddings_path=None,
                                          embeddings_dim=50)


@pytest.fixture
def training_set():
    return pd.DataFrame({"text": DATA}), pd.Series([0, 1, 0])


def test_fit(pipeline, training_set):
    pipeline.fit(training_set[0].loc[:, "text"], training_set[1], batch_size=1)
    assert pipeline.model.model.history.history['acc'][0] > 0  # If so the model has been fitted


def test_fit_predict(pipeline, training_set):
    pipeline.fit(training_set[0].loc[:, "text"], training_set[1], batch_size=1)
    assert len(pipeline.predict(["hello world"])) == 1


def test_cv(pipeline, training_set):
    n_splits = 2
    scores = pipeline.cv(training_set[0].loc[:, "text"], training_set[1], batch_size=1, n_splits=n_splits)
    assert len(scores) == n_splits
