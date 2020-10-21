import pytest
from classifier import utils
import pandas as pd
import os

DIR_PATH = os.path.dirname(__file__)
TRAIN_PATH = os.path.join(DIR_PATH, "trainSet.csv")
TEST_PATH = os.path.join(DIR_PATH, "candidateTestSet.txt")


@pytest.fixture(scope="module", autouse=True)
def cleanup():
    try:
        os.remove(TRAIN_PATH)
        os.remove(TEST_PATH)
    except OSError:
        pass
    yield
    os.remove(TRAIN_PATH)
    os.remove(TEST_PATH)


def test_data_loading():
    df = utils.load_data("fit", store_path=DIR_PATH)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (606823, 2)
    assert os.path.exists(TRAIN_PATH)
    df = utils.load_data("predict", store_path=DIR_PATH)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (67424, 1)
    assert os.path.exists(TEST_PATH)
