"""
Runs a script to interact with a model using the shell.
"""
import os

import pandas as pd

from test_tube import HyperOptArgumentParser
from src.cores.longformer_classifier import LONGFORMERClassifier
from collections import Counter


def load_model_from_experiment(experiment_folder: str):
    """ Function that loads the model from an experiment folder.
    :param experiment_folder: Path to the experiment folder.
    Return:
        - Pretrained model.
    """
    tags_csv_file = experiment_folder + "/meta_tags.csv"
    tags = pd.read_csv(tags_csv_file, header=None, index_col=0, squeeze=True).to_dict()
    checkpoints = [
        file
        for file in os.listdir(experiment_folder + "/checkpoints/")
        if file.endswith(".ckpt")
    ]
    checkpoint_path = experiment_folder + "/checkpoints/" + checkpoints[-1]

    model = LONGFORMERClassifier.load_from_metrics(weights_path=checkpoint_path, tags_csv=tags_csv_file)

    # Make sure model is in prediction mode
    model.eval()
    model.freeze()

    return model


if __name__ == "__main__":
    parser = HyperOptArgumentParser(description="Minimalist BERT Classifier", add_help=True)
    # parser.add_argument("--experiment", default='../../data/experiments/lightning_logs/version_29-04-2020--22-14-10',
    #                     type=str, help="Path to the experiment folder.")

    parser.add_argument("--experiment", default='../../data/experiments/lightning_logs/version_30-04-2020--16-32-26',
                        type=str, help="Path to the experiment folder.")
    hparams = parser.parse_args()
    print("Loading model...")
    model = load_model_from_experiment(hparams.experiment)
    print(model)

    print("Please write a movie review or quit to exit the interactive shell:")
    # Get input sentence

    test_path = '../../data/input/test.csv'
    test = pd.read_csv(test_path, encoding='utf-8')

    labels = []
    for index in range(test.shape[0]):
        if index % 1000 == 0:
            print(index)

        # prediction_list = []
        # text = test.iloc[index, 0]
        # while text:
        #     prediction = model.predict(sample={"text": text[:500]})['predicted_label']
        #     prediction_list.append(prediction)
        #     text = text[500:]
        #     if len(text) < 500:
        #         if len(text) < 10:
        #             break
        #         prediction = model.predict(sample={"text": text})['predicted_label']
        #         prediction_list.append(prediction)
        #         break
        #
        # word_count = Counter(prediction_list)
        # labels.append(word_count.most_common(1)[0][0])

        labels.append(model.predict(sample={'text': test.iloc[index, 0]})['predicted_label'])

    test['label'] = labels
    test['label'].to_csv('../../data/output/keys_longformer.csv', header=None, encoding='utf-8')
