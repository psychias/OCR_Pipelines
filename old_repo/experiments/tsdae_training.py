
# first TSDAE on new LB and then fine tune one noise
# compore the results
# Neccessary imports

from dataclasses import dataclass
import json
import pandas as pd
import math
import os
import gzip
import csv
import random
import time
import torch
from pylatexenc.latex2text import LatexNodes2Text
import nltk
nltk.download('punkt')

from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from sentence_transformers import models, util, evaluation, losses
from sentence_transformers import datasets

import datasets as dts
from datasets import load_dataset

from torch.utils.data import DataLoader


import nltk
for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg)

os.environ["WANDB_DISABLED"] = "true"


@dataclass
class Config:
    model_name: str = "Alibaba-NLP/gte-multilingual-base"
    tokenizer_name: str = "Alibaba-NLP/gte-multilingual-base"
    decoder_name: str = "bert-base-multilingual-cased"
    num_epochs: int = 1
    sample_size: int = 1000
    batch_size: int = 8



device = "cuda" if torch.cuda.is_available() else "cpu"
config = Config()


def load_sentences(file):
    with open(file, 'r',encoding="utf-8") as f:
        lines = f.readlines()

    return lines

fr_lines = load_sentences('fr.txt')
fr_sents = [line.strip() for line in fr_lines]

de_lines = load_sentences('de.txt')
de_sents = [line.strip()  for line in de_lines]

def split_sentences(sents):

    train = sents[:int(len(sents) * 0.80)]
    val = sents[int(len(sents) * 0.80):int(len(sents) * 0.90)]
    test = sents[int(len(sents) * 0.90):]

    return train, val, test


def prepare_training_data():
    train_de, val_de, test_de = split_sentences(de_sents)
    train_fr, val_fr, test_fr = split_sentences(fr_sents)
    train_sentences=  train_fr + train_de
    random.shuffle(train_sentences)
    return train_sentences, val_de, test_de, val_fr, test_fr


train_sentences, val_de, test_de, val_fr, test_fr = prepare_training_data()



train_dataset = DataLoader(train_sentences, shuffle = True, batch_size = 8)
train_dataset_noised = datasets.DenoisingAutoEncoderDataset(train_sentences)
train_dataloader = DataLoader(train_dataset_noised, batch_size = 8, shuffle = True)




embeddings_model = models.Transformer(
    config.model_name,
    config_args={"trust_remote_code": True},
    model_args={"trust_remote_code": True},
    tokenizer_args={"trust_remote_code": True},
).to(device)

pooling_model = models.Pooling(embeddings_model.get_word_embedding_dimension(), 'cls')
model = SentenceTransformer(modules=[embeddings_model, pooling_model])
model = model.to(device)

train_loss = losses.DenoisingAutoEncoderLoss(model,
                                             decoder_name_or_path=config.decoder_name,
                                             tie_encoder_decoder=False )




# train the model
start_time = time.time()


num_epochs = 1
steps_per_epoch = len(train_dataloader)
warmup_ratio = 0.10                                # 10% of total steps
warmup_steps = math.ceil(steps_per_epoch * num_epochs * warmup_ratio)

print('before the fit')
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=config.num_epochs,
    warmup_steps=warmup_steps,
    scheduler='warmuplinear',
    optimizer_params={'lr': 3e-5},
    weight_decay=0.01,
    show_progress_bar=True,
    use_amp=True,
    # optional:
    # evaluator=val_evaluator, evaluation_steps=250, output_path="checkpoints", save_best_model=True
)

end_time = time.time()

elapsed_time = end_time - start_time

print(f"Model trained in {elapsed_time:.2f} seconds or {elapsed_time//60} minutes, on a Google Colab Pro with A100 GPU & High-RAM.")

pretrained_model_save_path = 'tsdae-gte_ge_fr'
model.save(pretrained_model_save_path)


