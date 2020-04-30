# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
import sys

os.chdir(sys.path[0])
from datetime import datetime

import random
import os
import torch
import json
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

from pytorch_lightning.logging import TestTubeLogger


def setup_testube_logger() -> TestTubeLogger:
    """ Function that sets the TestTubeLogger to be used. """
    try:
        job_id = os.environ["SLURM_JOB_ID"]
    except Exception:
        job_id = None

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y--%H-%M-%S")

    return TestTubeLogger(
        save_dir="../../data/experiments/",
        version=job_id + "_" + dt_string if job_id else dt_string,
        name="lightning_logs",
    )


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


def generate_train_dev_csv(input_train_path, output_train_path, output_dev_path):
    """
    生成训练/验证/测试/标签 json文件
    :return:
    """
    if os.path.exists(output_train_path) and os.path.exists(output_dev_path):
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
            text = data.iloc[index, 0]
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


if __name__ == '__main__':
    generate_train_dev_csv(input_train_path='../../data/input/train.csv',
                           output_train_path='../../data/output/train.csv',
                           output_dev_path='../../data/output/dev.csv')
