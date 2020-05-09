# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
import sys

os.chdir(sys.path[0])
from datetime import datetime
from sklearn.model_selection import train_test_split

import random
import os
import torch
import re
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

from pytorch_lightning.logging import TestTubeLogger

random.seed(42)


def setup_testube_logger() -> TestTubeLogger:
    """ Function that sets the TestTubeLogger to be used. """
    try:
        job_id = os.environ["SLURM_JOB_ID"]
    except Exception:
        job_id = None

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y--%H-%M-%S")

    return TestTubeLogger(save_dir="../../data/experiments/", version=job_id + "_" + dt_string if job_id else dt_string,
                          name="lightning_logs")


def mask_fill(fill_value: float, tokens: torch.tensor, embeddings: torch.tensor, padding_index: int, ) -> torch.tensor:
    """
    Function that masks embeddings representing padded elements.
    :param fill_value: the value to fill the embeddings belonging to padded tokens.
    :param tokens: The input sequences [bsz x seq_len].
    :param embeddings: word embeddings [bsz x seq_len x hiddens].
    :param padding_index: Index of the padding token.
    """
    padding_mask = tokens.eq(padding_index).unsqueeze(-1)

    return embeddings.float().masked_fill_(padding_mask, fill_value).type_as(embeddings)


def replace_typical_misspell(text):
    mispell_dict = {'colour': 'color', 'centre': 'center', 'didnt': 'did not', 'doesnt': 'does not',
                    'isnt': 'is not', 'shouldnt': 'should not', 'favourite': 'favorite', 'travelling': 'traveling',
                    'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',
                    'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize',
                    'instagram': 'social medium',
                    'whatsapp': 'social medium', 'snapchat': 'social medium', "ain't": "is not",
                    "aren't": "are not", "can't": "cannot", "'cause": "because", "could've": "could have",
                    "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't": "do not",
                    "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
                    "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you",
                    "how'll": "how will", "how's": "how is", "I'd": "I would", "I'd've": "I would have",
                    "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have",
                    "i'd": "i would", "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have",
                    "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                    "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not",
                    "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
                    "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
                    "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",
                    "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                    "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
                    "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
                    "she's": "she is", "should've": "should have", "shouldn't": "should not",
                    "shouldn't've": "should not have",
                    "so've": "so have", "so's": "so as", "this's": "this is", "that'd": "that would",
                    "that'd've": "that would have", "that's": "that is",
                    "there'd": "there would", "there'd've": "there would have", "there's": "there is",
                    "here's": "here is", "they'd": "they would", "they'd've": "they would have",
                    "they'll": "they will", "they'll've": "they will have",
                    "they're": "they are", "they've": "they have", "to've": "to have",
                    "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
                    "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                    "we've": "we have", "weren't": "were not", "what'll": "what will",
                    "what'll've": "what will have", "what're": "what are", "what's": "what is",
                    "what've": "what have", "when's": "when is", "when've": "when have",
                    "where'd": "where did", "where's": "where is", "where've": "where have",
                    "who'll": "who will", "who'll've": "who will have", "who's": "who is",
                    "who've": "who have", "why's": "why is", "why've": "why have",
                    "will've": "will have", "won't": "will not", "won't've": "will not have",
                    "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                    "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",
                    "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
                    "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                    "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center',
                    'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling',
                    'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',
                    'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize',
                    'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What',
                    'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are',
                    'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do',
                    'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does',
                    'mastrubation': 'masturbation', 'mastrubate': 'masturbate',
                    "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum',
                    'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018',
                    'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess',
                    "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization',
                    'demonitization': 'demonetization', 'demonetisation': 'demonetization'
                    }

    def _replace(match):
        return mispellings[match.group(0)]

    def _get_mispell(mispell_dict):
        mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
        return mispell_dict, mispell_re

    mispellings, mispellings_re = _get_mispell(mispell_dict)

    return mispellings_re.sub(_replace, text.lower())


def deal_text(text):
    """
    处理text
    :param text:
    :return:
    """
    # 去除其中的X值
    text = re.sub('X ', 'X %$', text)
    text = re.sub('X+', 'X', text)
    text = re.sub('/X+', 'X ', text)
    text = re.sub('X, ', 'X ', text)
    text = text.lower().strip()
    text = re.sub('x,+', 'x', text)
    text = re.sub('x+', 'x', text)
    text = re.sub('x%\$+', 'x%$', text)
    text = re.sub('%\$+', ' ', text)
    text = re.sub(' +', ' ', text)
    text = re.sub('x x x', 'x', text)
    text = re.sub('x x x', 'x', text)
    text = re.sub(' x x ', ' x ', text)
    text = re.sub('\!+', '!', text)
    text = re.sub('\?+', '?', text)
    text = re.sub('\*+', '*', text)

    text = text.replace(" n't", "n't")
    text = text.replace(" 's", "'s")
    text = text.replace(" 've", "'ve")
    text = text.replace(" 'm", "'m")
    text = text.replace(" 're", "'re")
    text = text.replace(" 'd", "'d")
    text = text.replace(" 'll", "'ll")

    text = replace_typical_misspell(text=text)

    return text


def generate_train_dev_csv(input_train_path, output_train_path, output_dev_path):
    """
    生成训练/验证/测试/标签 json文件
    :return:
    """
    if os.path.exists(output_train_path) and os.path.exists(output_dev_path) and False:
        train = pd.read_csv(output_train_path, encoding='utf-8')
        dev = pd.read_csv(output_dev_path, encoding='utf-8')
    else:
        # text, label
        data = pd.read_csv(filepath_or_buffer=input_train_path, encoding='utf-8')

        data['text'] = data['text'].apply(lambda x: x.replace('\n', ' '))
        data['text'] = data['text'].apply(lambda x: x.replace('\t', ' '))
        data.drop_duplicates(inplace=True)
        data.reset_index(inplace=True, drop=True)

        data_count = data.groupby(by=['label'], as_index=False)['label'].agg({'number': 'count'})
        data_count = dict(zip(data_count['label'], data_count['number']))
        data_count = dict(sorted(data_count.items(), key=lambda item: item[1]))
        count_average = int(data.shape[0] // (len(data_count)))
        # 1646
        print('count_average: {}'.format(count_average))

        # 不用处理的label
        no_deal_label = []
        deal_label = []
        deal_label_number = []
        for key, value in data_count.items():
            if value > count_average:
                no_deal_label.append(key)
            else:
                deal_label.append(key)
                deal_label_number.append(count_average // data_count[key])

        labels = []
        texts = []
        for index in range(data.shape[0]):
            text = deal_text(data.iloc[index, 0])
            label = data.iloc[index, 1]

            # 截断
            # while len(text) > 0:
            #     texts.append(text[:500])
            #     text = text[500:]
            #
            #     # 原始的数据
            #     labels.append(label)
            #
            #     if len(text) < 500:
            #         break

            # 不截断
            texts.append(text)
            labels.append(label)

            if len(text) < 5:
                continue

        data = pd.DataFrame({'text': texts, 'label': labels})

        train_index = []
        dev_index = []
        for item in data['label'].unique():
            item_data = data[data['label'] == item]
            item_data = item_data.index.tolist()
            item_data_len = len(item_data)
            train_count = int(item_data_len * 0.8)
            train_count_data = random.sample(item_data, train_count)
            train_index.extend(train_count_data)
            for value in train_count_data:
                item_data.remove(value)
            dev_index.extend(item_data)

        train = data.iloc[np.asarray(train_index)]
        dev = data.iloc[np.asarray(dev_index)]

        train = shuffle(train, random_state=42)
        dev = shuffle(dev, random_state=42)

        train.reset_index(inplace=True, drop=True)
        dev.reset_index(inplace=True, drop=True)

        train.to_csv(output_train_path, encoding='utf-8', index=None)
        dev.to_csv(output_dev_path, encoding='utf-8', index=None)

    return train, dev


def generate_test_csv(input_test_path, output_test_path):
    if os.path.exists(output_test_path) and False:
        test = pd.read_csv(output_test_path, encoding='utf-8')
    else:
        # text
        test = pd.read_csv(filepath_or_buffer=input_test_path, encoding='utf-8')

        test['text'] = test['text'].apply(lambda x: x.replace('\n', ' '))
        test['text'] = test['text'].apply(lambda x: x.replace('\t', ' '))
        test['text'] = test['text'].apply(lambda x: deal_text(x))

        test.to_csv(output_test_path, encoding='utf-8', index=None)

    return test


def generate_adversarial_validaion(input_train_path, input_dev_path, input_test_path, output_test_path,
                                   output_train_path):
    # text, label
    train = pd.read_csv(input_train_path)
    # text, label
    dev = pd.read_csv(input_dev_path)
    # text
    test = pd.read_csv(input_test_path)

    train['label_before'] = train['label'].astype(int)
    dev['label_before'] = dev['label'].astype(int)

    train['label'] = 1
    dev['label'] = 1
    test['label'] = 0

    data = pd.concat([train, dev, test], ignore_index=True)

    train_X, test_X, train_y, test_y = train_test_split(data[['text', 'label_before']].values,
                                                        data['label'].values,
                                                        test_size=0.2,
                                                        random_state=0)

    train = pd.DataFrame({"text": train_X[:, 0], "label": train_y, "label_before": train_X[:, 1]})
    test = pd.DataFrame({"text": test_X[:, 0], "label": test_y, "label_before": test_X[:, 1]})

    train.to_csv(output_train_path, index=None, encoding='utf-8')
    test.to_csv(output_test_path, index=None, encoding='utf-8')


if __name__ == '__main__':
    # generate_train_dev_csv(input_train_path='../../data/input/train.csv',
    #                        output_train_path='../../data/output/train.csv',
    #                        output_dev_path='../../data/output/dev.csv')
    # generate_test_csv(input_test_path='../../data/input/test.csv',
    #                   output_test_path='../../data/output/test.csv')
    generate_adversarial_validaion(input_train_path='../../data/output/train.csv',
                                   input_dev_path='../../data/output/dev.csv',
                                   input_test_path='../../data/output/test.csv',
                                   output_test_path='../../data/adversarial_validation/test.csv',
                                   output_train_path='../../data/adversarial_validation/train.csv')
