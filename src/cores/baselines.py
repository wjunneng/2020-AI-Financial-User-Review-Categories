# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
import sys

os.chdir(sys.path[0])

import pandas as pd
from simpletransformers.classification import ClassificationModel

train = pd.read_csv('../../data/output/train.csv')
test = pd.read_csv('../../data/output/test.csv')
model = ClassificationModel('roberta', 'roberta-large', num_labels=11,
                            args={'sliding_window': True, 'reprocess_input_data': True, 'overwrite_output_dir': True,
                                  'logging_steps': 5, 'train_batch_size': 8, 'gradient_accumulation_steps': 16,
                                  'stride': 0.6, 'max_seq_length': 512, 'num_train_epochs': 4})
model.train_model(train)
predictions, _ = model.predict([row['text'] for _, row in test.iterrows()])
pd.DataFrame({'ID': test.index, 'label': predictions}).to_csv('../../data/output/keys_longformer.csv', index=False,
                                                              header=False)
