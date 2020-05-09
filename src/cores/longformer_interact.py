"""
Runs a script to interact with a model using the shell.
"""
from __future__ import absolute_import, division, print_function
import os
import sys

sys.path.append('/content/2020-AI-Financial-User-Review-Categories')
os.chdir(sys.path[0])

import torch
import pandas as pd

from test_tube import HyperOptArgumentParser
from src.cores.longformer_classifier import LONGFORMERClassifier

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
    parser.add_argument("--experiment",
                        default='../../data/experiments/lightning_logs/version_09-05_adversarial_validation',
                        type=str, help="Path to the experiment folder.")
    hparams = parser.parse_args()
    print("Loading model...")
    model = load_model_from_experiment(hparams.experiment).to(device=DEVICE)

    print("Please write a movie review or quit to exit the interactive shell:")
    # Get input sentence

    adversarial_validation = True
    if adversarial_validation:
        train_path = '../../data/adversarial_validation/train.csv'
        test_path = '../../data/adversarial_validation/test.csv'
        test = pd.read_csv(test_path, encoding='utf-8')
        train = pd.read_csv(train_path, encoding='utf-8')

        data = pd.concat([train, test], ignore_index=True)
        data = data[data['label'] == 1]
        data.reset_index(drop=True, inplace=True)

        labels = []
        for index in range(0, data.shape[0], 4):
            if index % 1000 == 0:
                print(index)

            try:
                label = model.predict(
                    samples=[{'text': data.iloc[index + 0, 0]},
                             {'text': data.iloc[index + 1, 0]},
                             {'text': data.iloc[index + 2, 0]},
                             {'text': data.iloc[index + 3, 0]}],
                    probability=True)['predicted_label']
            except Exception as e:
                print(e)
                label = model.predict(
                    samples=[{'text': data.iloc[index + 0, 0]},
                             {'text': data.iloc[index + 1, 0]}],
                    probability=True)['predicted_label']

            labels.extend(label)

        data['label_after'] = labels
        data.sort_values(by='label_after', inplace=True, ascending=False)
        data.reset_index(drop=True, inplace=True)
        data['label'] = data['label_before']
        dev_rate_value = int(data.shape[0] * 0.2)
        dev = data.loc[:dev_rate_value]
        train = data.loc[dev_rate_value + 1:]

        dev[['text', 'label']].to_csv('../../data/output/dev.csv', index=None, encoding='utf-8')
        train[['text', 'label']].to_csv('../../data/output/train.csv', index=None, encoding='utf-8')

    else:
        test_path = '../../data/output/test.csv'
        test = pd.read_csv(test_path, encoding='utf-8')

        labels = []
        for index in range(0, test.shape[0], 4):
            if index % 1000 == 0:
                print(index)

            try:
                label = model.predict(
                    samples=[{'text': test.iloc[index + 0, 0]},
                             {'text': test.iloc[index + 1, 0]},
                             {'text': test.iloc[index + 2, 0]},
                             {'text': test.iloc[index + 3, 0]}])['predicted_label']
            except:
                label = model.predict(
                    samples=[{'text': test.iloc[index + 0, 0]},
                             {'text': test.iloc[index + 1, 0]}])['predicted_label']

            labels.extend(label)

        test['label'] = labels
        test['label'].to_csv('../../data/output/keys_longformer.csv', header=None, encoding='utf-8')
