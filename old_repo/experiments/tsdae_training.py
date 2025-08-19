
# first TSDAE on new LB and then fine tune one noise
# compore the results
# Neccessary imports

import json
import pandas as pd
import math
import os
import gzip
import csv
import random
import time

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





def load_sentences(file):
    with open(file, 'r') as f:
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

embeddings_model = models.Transformer(model_name)
pooling_model = models.Pooling(embeddings_model.get_word_embedding_dimension(), 'cls')


model = SentenceTransformer(modules=[embeddings_model, pooling_model])


sample_size = 1000
batch_size = 8
epochs = 1



os.environ["WANDB_DISABLED"] = "true"

# train the model
start_time = time.time()
# model.fit(
#     train_objectives=[(train_dataloader,train_loss )],
#     epochs = 1,
#     weight_decay=0,
#     scheduler='constantlr',
#     optimizer_params={'lr': 3e-5},
#     show_progress_bar=True,
#     use_amp=True # set to False if GPU does not support FP16 cores
# )

import math

num_epochs = 1
steps_per_epoch = len(train_dataloader)
warmup_ratio = 0.10                                # 10% of total steps
warmup_steps = math.ceil(steps_per_epoch * num_epochs * warmup_ratio)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
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

# Calculate elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"Model trained in {elapsed_time:.2f} seconds or {elapsed_time//60} minutes, on a Google Colab Pro with A100 GPU & High-RAM.")

# Save path of the model
pretrained_model_save_path = 'output/tsdae-gte_ge_fr'
# Save the model locally
model.save(pretrained_model_save_path)



