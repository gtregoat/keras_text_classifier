import classifier as clf
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PREDICTION_PATH = os.path.join(DATA_DIR, "predictions.csv")
N_SPLITS = 4
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

if __name__ == '__main__':
    df = pd.read_csv(TRAIN_PATH).loc[:, ["text", "label"]]
    pipeline = clf.TextClassificationPipeline(sequence_length=25,
                                              embeddings_path="data/glove/glove.6B.300d.txt",
                                              embeddings_dim=300)
    # Evaluate using cross - validation
    scores = pipeline.cv(df.loc[:, "text"], df.loc[:, "label"], n_splits=N_SPLITS, epochs=15, refit=True, shuffle=True)
    print(f"Average loss:{sum([i[0] for i in scores]) / N_SPLITS}",
          f"\nAverage accuracy: {sum([i[1] for i in scores]) / N_SPLITS}")
    # Generate predictions on the test set
    df = pd.read_csv(TEST_PATH)
    df.loc[:, "target"] = pipeline.predict(df.loc[:, "text"])
    df.loc[:, ["id", "target"]].to_csv(PREDICTION_PATH, index=False)
