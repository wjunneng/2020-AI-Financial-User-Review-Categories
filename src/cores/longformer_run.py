# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
import sys

sys.path.append('/content/2020-AI-Financial-User-Review-Categories')
os.chdir(sys.path[0])

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from test_tube import HyperOptArgumentParser
from torchnlp.random import set_seed

from src.libs.utils import setup_testube_logger
from src.cores.longformer_classifier import LONGFORMERClassifier


def main(hparams) -> None:
    """
    Main training routine specific for this project
    :param hparams:
    """
    set_seed(hparams.seed)
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = LONGFORMERClassifier(hparams)

    # ------------------------
    # 2 INIT EARLY STOPPING
    # ------------------------
    early_stop_callback = EarlyStopping(
        monitor=hparams.monitor,
        min_delta=0.0,
        patience=hparams.patience,
        verbose=True,
        mode=hparams.metric_mode,
    )
    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        logger=setup_testube_logger(),
        checkpoint_callback=True,
        early_stop_callback=early_stop_callback,
        default_save_path="../../data/experiments/",
        gpus=hparams.gpus,
        distributed_backend="dp",
        use_amp=False,
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        val_percent_check=hparams.val_percent_check,
    )

    # --------------------------------
    # 4 INIT MODEL CHECKPOINT CALLBACK
    # -------------------------------
    ckpt_path = os.path.join(
        trainer.default_save_path,
        trainer.logger.name,
        f"version_{trainer.logger.version}",
        "checkpoints",
    )
    # initialize Model Checkpoint Saver
    checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_path,
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.monitor,
        period=1,
        mode=hparams.metric_mode,
    )
    trainer.checkpoint_callback = checkpoint_callback

    # ------------------------
    # 5 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    parser = HyperOptArgumentParser(
        strategy="random_search",
        description="Minimalist BERT Classifier",
        add_help=True,
    )
    parser.add_argument("--seed", type=int, default=3, help="Training seed.")
    parser.add_argument(
        "--save_top_k",
        default=1,
        type=int,
        help="The best k models according to the quantity monitored will be saved.",
    )
    # Early Stopping
    parser.add_argument(
        "--monitor", default="val_acc", type=str, help="Quantity to monitor."
    )
    parser.add_argument(
        "--metric_mode",
        default="max",
        type=str,
        help="If we want to min/max the monitored quantity.",
        choices=["auto", "min", "max"],
    )
    parser.add_argument(
        "--patience",
        default=3,
        type=int,
        help=(
            "Number of epochs with no improvement "
            "after which training will be stopped."
        ),
    )
    parser.add_argument(
        "--min_epochs",
        default=4,
        type=int,
        help="Limits training to a minimum number of epochs",
    )
    parser.add_argument(
        "--max_epochs",
        default=5,
        type=int,
        help="Limits training to a max number number of epochs",
    )

    # Batching
    parser.add_argument(
        "--batch_size", default=6, type=int, help="Batch size to be used."
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        default=2,
        type=int,
        help=(
            "Accumulated gradients runs K small batches of size N before "
            "doing a backwards pass."
        ),
    )

    # gpu args
    parser.add_argument("--gpus", type=int, default=1, help="How many gpus")
    parser.add_argument(
        "--val_percent_check",
        default=1.0,
        type=float,
        help=(
            "If you don't want to use the entire dev set (for debugging or "
            "if it's huge), set how much of the dev set you want to use with this flag."
        ),
    )

    # each LightningModule defines arguments relevant to it
    parser = LONGFORMERClassifier.add_model_specific_args(parser)
    hparams = parser.parse_args()

    adversarial_validation = True
    if adversarial_validation:
        hparams.gpus = 1
        hparams.batch_size = 4
        hparams.accumulate_grad_batches = 1
        hparams.loader_workers = 0
        hparams.nr_frozen_epochs = 1
        hparams.save_top_k = 3
        hparams.patience = 5
        hparams.min_epochs = 8
        hparams.max_eppochs = 10
        hparams.encoder_model = '../../data/longformer-base-4096'

        hparams.train_csv = '../../data/adversarial_validation/train.csv'
        hparams.dev_csv = '../../data/adversarial_validation/test.csv'
        hparams.test_csv = '../../data/adversarial_validation/test.csv'
        hparams.label_set = '0,1'
        hparams.num_labels = 2
    else:
        # parameters
        hparams.gpus = 1
        hparams.batch_size = 4
        hparams.accumulate_grad_batches = 1
        hparams.loader_workers = 0
        hparams.nr_frozen_epochs = 1
        hparams.save_top_k = 3
        hparams.patience = 5
        hparams.min_epochs = 8
        hparams.max_eppochs = 10
        hparams.encoder_model = '../../data/longformer-base-4096'

        hparams.train_csv = '../../data/output/train.csv'
        hparams.dev_csv = '../../data/output/dev.csv'
        hparams.test_csv = '../../data/output/dev.csv'
        hparams.label_set = '0,1,2,3,4,5,6,7,8,9,10'
        hparams.num_labels = 11

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hparams)
