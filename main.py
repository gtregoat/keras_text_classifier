import classifier as clf
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PREDICTION_PATH = os.path.join(DATA_DIR, "predictions.csv")
N_SPLITS = 4

if __name__ == '__main__':
    df = clf.load_data("fit", store_path=DATA_DIR)
    pipeline = clf.TextClassificationPipeline(sequence_length=15,
                                              embeddings_path="data/glove/glove.6B.50d.txt",
                                              embeddings_dim=50)
    # Evaluate using cross - validation
    scores = pipeline.cv(df.loc[:, "text"], df.loc[:, "label"], n_splits=N_SPLITS, epochs=20, refit=True)
    print(f"Average loss:{sum([i[0] for i in scores]) / N_SPLITS}",
          f"\nAverage accuracy: {sum([i[1] for i in scores]) / N_SPLITS}")
    # Generate predictions on the test set
    df = clf.load_data("predict", DATA_DIR)
    df.loc[:, "predictions"] = pipeline.predict(df.loc[:, "text"])
    df.to_csv(PREDICTION_PATH, index=False)
