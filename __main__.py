import argparse
import pandas as pd
from classifier import pipeline


parser = argparse.ArgumentParser()
parser.add_argument("--training_data",
                    help="path to training data",
                    required=False,
                    default=None)
# parser.add_argument("--output_model",
#                     help="path to store the model",
#                     required=False,
#                     default=None)
parser.add_argument("--prediction_data",
                    help="path to prediction data",
                    required=False,
                    default=None)
parser.add_argument("--output_predictions",
                    help="path where to output the predictions",
                    required=False,
                    default=None)
parser.add_argument("--sequence_length",
                    help="maximum number of words taken into account by the model",
                    required=False,
                    default=15)
parser.add_argument("--embeddings_path",
                    help="Path to pretrained glove embeddings.",
                    required=False,
                    default=None)
parser.add_argument("--embeddings_dim",
                    help="Dimension of the embeddings. If the embeddings path is given, this number must match the "
                         "dimension of the embeddings",
                    required=False,
                    default=50)
parser.add_argument("--batch_size",
                    help="Batch size when training the model",
                    required=False,
                    default=32)
parser.add_argument("--epochs",
                    help="Number of epochs to train the model.",
                    required=False,
                    default=1)


args = parser.parse_args()

pipeline = pipeline.TextClassificationPipeline(sequence_length=args.sequence_length,
                                               embeddings_path=args.embeddings_path,
                                               embeddings_dim=args.embeddings_dim)


def train():
    df = pd.read_csv(args.training_data,
                     names=["text", "label"])
    pipeline.fit(df.loc[:, "text"], df.loc[:, "label"], epochs=int(args.epochs), batch_size=int(args.batch_size))


def predict():
    if not pipeline.fitted:
        raise Exception("the model needs to be fitted before generating predictions.")
    df = pd.read_csv(args.prediction_data)
    df.loc[:, "predictions"] = pipeline.predict(df)
    df.to_csv(args.output_predictions, index=False)


if args.training_data is not None:
    train()

if args.prediction_data is not None:
    predict()
