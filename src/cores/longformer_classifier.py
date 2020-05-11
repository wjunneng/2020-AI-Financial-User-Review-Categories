# -*- coding: utf-8 -*-
import logging as log
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoModel

import pytorch_lightning as pl

from test_tube import HyperOptArgumentParser
from torchnlp.encoders import LabelEncoder
from torchnlp.utils import collate_tensors, lengths_to_mask
from src.longformer.longformer import Longformer

from src.libs.utils import mask_fill
from src.libs.dataloader import sentiment_analysis_dataset
from src.cores.longformer_tokenizer import LONGFORMERTextEncoder

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class LONGFORMERClassifier(pl.LightningModule):
    """
    Sample model to show how to use LongFormer to classify sentences.

    :param hparams: ArgumentParser containing the hyperparameters.
    """

    def __init__(self, hparams) -> None:
        super(LONGFORMERClassifier, self).__init__()
        self.hparams = hparams
        self.batch_size = hparams.batch_size

        # build model
        self.__build_model()

        # Loss criterion initialization.
        self.__build_loss()

        if hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False
        self.nr_frozen_epochs = hparams.nr_frozen_epochs

    def __build_model(self) -> None:
        """ Init BERT model + tokenizer + classification head."""
        # self.bert = AutoModel.from_pretrained(
        #     self.hparams.encoder_model, output_hidden_states=True, num_labels=11
        # )
        self.bert = Longformer.from_pretrained(self.hparams.encoder_model)

        # set the number of features our encoder model will return...
        if self.hparams.encoder_model == "google/bert_uncased_L-2_H-128_A-2":
            self.encoder_features = 128
        else:
            self.encoder_features = 768

        # Tokenizer
        # self.tokenizer = LONGFORMERTextEncoder('../../data/roberta-base-pytorch/')
        self.tokenizer = LONGFORMERTextEncoder('../../data/roberta-large-pytorch/')

        # Label Encoder
        self.label_encoder = LabelEncoder(self.hparams.label_set.split(","), reserved_labels=[])
        self.label_encoder.unknown_index = None

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.encoder_features, self.encoder_features * 2),
            nn.Tanh(),
            nn.Linear(self.encoder_features * 2, self.encoder_features),
            nn.Tanh(),
            nn.Linear(self.encoder_features, self.label_encoder.vocab_size),
        )

    def __build_loss(self):
        """ Initializes the loss function/s. """
        self._loss = nn.CrossEntropyLoss()

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            log.info(f"\n-- Encoder model fine-tuning")
            for param in self.bert.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.bert.parameters():
            param.requires_grad = False
        self._frozen = True

    def predict(self, samples: dict, probability: bool = False) -> dict:
        """ Predict function.
        :param sample: dictionary with the text we want to classify.

        Returns:
            Dictionary with the input text and the predicted label.
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            model_tokens = []
            model_lengths = []
            for sample in samples:
                model_input, _ = self.prepare_sample([sample], prepare_target=False)
                model_tokens.append(model_input['tokens'])
                model_lengths.append(model_input['lengths'].numpy()[0])

            model_inputs = dict(
                {'tokens': torch.ones((len(samples), max(model_lengths))).to(torch.int64).to(device=DEVICE)})
            for token_index in range(len(model_tokens)):
                current_tokens = model_tokens[token_index].flatten()

                model_inputs['tokens'][token_index][:len(current_tokens)] = current_tokens.to(torch.int64).to(
                    device=DEVICE)

            model_inputs['lengths'] = torch.Tensor(np.asarray(model_lengths)).to(torch.int64).to(device=DEVICE)
            model_out = self.forward(**model_inputs)
            logits = model_out["logits"].cpu().numpy()
            if probability:
                predicted_labels = logits[:, 1]
            else:
                predicted_labels = [
                    self.label_encoder.index_to_token[prediction]
                    for prediction in np.argmax(logits, axis=1)
                ]
            sample["predicted_label"] = predicted_labels

        return sample

    def forward(self, tokens, lengths):
        """ Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        :param lengths: source lengths [batch_size]

        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        tokens = tokens[:, : lengths.max()]
        # When using just one GPU this should not change behavior
        # but when splitting batches across GPU the tokens have padding
        # from the entire original batch
        mask = lengths_to_mask(lengths, device=tokens.device)
        # Run BERT model.
        word_embeddings = self.bert(tokens, mask)[0]

        # Average Pooling
        word_embeddings = mask_fill(
            0.0, tokens, word_embeddings, self.tokenizer.padding_index
        )
        sentemb = torch.sum(word_embeddings, 1)
        sum_mask = mask.unsqueeze(-1).expand(word_embeddings.size()).float().sum(1)
        sentemb = sentemb / sum_mask

        return {"logits": self.classification_head(sentemb)}

    def loss(self, predictions: dict, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x 1] with model predictions
        :param labels: Label values [batch_size]

        Returns:
            torch.tensor with loss value.
        """
        return self._loss(predictions["logits"], targets["labels"])

    def prepare_sample(self, sample: list, prepare_target: bool = True) -> (dict, dict):
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.

        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        sample = collate_tensors(sample)
        tokens, lengths = self.tokenizer.batch_encode(sample["text"])

        inputs = {"tokens": tokens, "lengths": lengths}

        if not prepare_target:
            return inputs, {}

        # Prepare target:
        try:
            targets = {"labels": self.label_encoder.batch_encode(sample["label"])}
            return inputs, targets
        except RuntimeError:
            raise Exception("Label encoder found an unknown label.")

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """
        Runs one training step. This usually consists in the forward function followed
            by the loss function.

        :param batch: The output of your dataloader.
        :param batch_nb: Integer displaying which batch this is

        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_val = self.loss(model_out, targets)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        tqdm_dict = {"train_loss": loss_val}
        output = OrderedDict(
            {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.

        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_val = self.loss(model_out, targets)

        y = targets["labels"]
        y_hat = model_out["logits"]

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)

        output = OrderedDict({"val_loss": loss_val, "val_acc": val_acc, })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def test_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the testing step but with the model in test mode.

        Returns:
            - dictionary passed to the test_end function.
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_test = self.loss(model_out, targets)

        y = targets["labels"]
        y_hat = model_out["logits"]

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        test_acc = torch.tensor(test_acc)

        if self.on_gpu:
            test_acc = test_acc.cuda(loss_test.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_test = loss_test.unsqueeze(0)
            test_acc = test_acc.unsqueeze(0)

        output = OrderedDict({"test_loss": loss_test, "test_acc": test_acc, })

        # can also return just a scalar instead of a dict (return loss_test)
        return output

    def test_end(self, outputs: list) -> dict:
        """ Function that takes as input a list of dictionaries returned by the test_step
        function and measures the model performance accross the entire test set.

        Returns:
            - Dictionary with metrics to be added to the lightning logger.
        """
        test_loss_mean = 0
        test_acc_mean = 0
        for output in outputs:
            test_loss = output["test_loss"]

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                test_loss = torch.mean(test_loss)
            test_loss_mean += test_loss

            # reduce manually when using dp
            test_acc = output["test_acc"]
            if self.trainer.use_dp or self.trainer.use_ddp2:
                test_acc = torch.mean(test_acc)

            test_acc_mean += test_acc

        test_loss_mean /= len(outputs)
        test_acc_mean /= len(outputs)
        tqdm_dict = {"test_loss": test_loss_mean, "test_acc": test_acc_mean}
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "test_loss": test_loss_mean,
        }
        return result

    def validation_end(self, outputs: list) -> dict:
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.

        Returns:
            - Dictionary with metrics to be added to the lightning logger.
        """
        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = output["val_loss"]

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output["val_acc"]
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {"val_loss": val_loss_mean, "val_acc": val_acc_mean}
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "val_loss": val_loss_mean,
        }
        return result

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {"params": self.classification_head.parameters()},
            {
                "params": self.bert.parameters(),
                "lr": self.hparams.encoder_learning_rate,
            },
        ]
        optimizer = optim.Adam(parameters, lr=self.hparams.learning_rate)
        return [optimizer], []

    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()

    def __retrieve_dataset(self, train=True, val=True, test=True):
        """ Retrieves task specific dataset """
        return sentiment_analysis_dataset(self.hparams, train, val, test)

    @pl.data_loader
    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        self._train_dataset = self.__retrieve_dataset(val=False, test=False)[0]
        return DataLoader(
            dataset=self._train_dataset,
            sampler=RandomSampler(self._train_dataset),
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    @pl.data_loader
    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        self._dev_dataset = self.__retrieve_dataset(train=False, test=False)[0]
        return DataLoader(
            dataset=self._dev_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    @pl.data_loader
    def test_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        self._test_dataset = self.__retrieve_dataset(train=False, val=False)[0]
        return DataLoader(
            dataset=self._test_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    @classmethod
    def add_model_specific_args(
            cls, parser: HyperOptArgumentParser
    ) -> HyperOptArgumentParser:
        """ Parser for Estimator specific arguments/hyperparameters.
        :param parser: HyperOptArgumentParser obj

        Returns:
            - updated parser
        """
        parser.add_argument(
            "--encoder_model",
            default="bert-base-uncased",
            type=str,
            help="Encoder model to be used.",
        )
        parser.add_argument(
            "--encoder_learning_rate",
            default=1e-05,
            type=float,
            help="Encoder specific learning rate.",
        )
        parser.add_argument(
            "--learning_rate",
            default=3e-05,
            type=float,
            help="Classification head learning rate.",
        )
        parser.opt_list(
            "--nr_frozen_epochs",
            default=1,
            type=int,
            help="Number of epochs we want to keep the encoder model frozen.",
            tunable=True,
            options=[0, 1, 2, 3, 4, 5],
        )
        # Data Args:
        parser.add_argument(
            "--label_set",
            default="pos,neg",
            type=str,
            help="Classification labels set.",
        )
        parser.add_argument(
            "--train_csv",
            default="data/imdb_reviews_train.csv",
            type=str,
            help="Path to the file containing the train data.",
        )
        parser.add_argument(
            "--dev_csv",
            default="data/imdb_reviews_test.csv",
            type=str,
            help="Path to the file containing the dev data.",
        )
        parser.add_argument(
            "--test_csv",
            default="data/imdb_reviews_test.csv",
            type=str,
            help="Path to the file containing the dev data.",
        )
        parser.add_argument(
            "--loader_workers",
            default=8,
            type=int,
            help="How many subprocesses to use for data loading. 0 means that \
                the data will be loaded in the main process.",
        )
        return parser
