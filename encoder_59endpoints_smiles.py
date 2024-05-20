#
# This script is used to derive the NYAN latent representation for the 59-endpoint acute toxicity data
# The generated latent is saved to /datasets/MTL/XX
# It should be noted that we can use this script to generate many different random NYAN latent
#

import os
import sys
import warnings
import json
import csv
import random
import loading
import numpy as np
import csv
import pandas as pd

from data import BondType

import logging

from tqdm import tqdm

dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"


input_smiles = dir_path + "datasets/MTL/dataset.txt"
output_file = dir_path + "datasets/MTL/NYAN_latent0.txt"

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from spektral.data import BatchLoader, Dataset, Graph
from spektral import transforms

from model import VAE, NyanEncoder, NyanDecoder, EpochCounter

import scipy as sp
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    for device in gpu_devices:
        print(device)
        tf.config.experimental.set_memory_growth(device, True)
else:
    print('No GPU available')


logging.getLogger("pysmiles").setLevel(logging.CRITICAL)

strategy = tf.distribute.MirroredStrategy()

save = dir_path + "saves/ZINC-extmodel5hk-3M-325K"
batch_size = 1

with strategy.scope():

    encoder = NyanEncoder(latent_dim=64, batched=True)
    decoder = NyanDecoder(fingerprint_bits=679, regression=1613)

    model = VAE(encoder, decoder)
    model.load_weights(save).expect_partial()

print("Generating latents using the save {}".format(save))

df = pd.read_csv(input_smiles, dtype={'RTECS_ID': str})  # read the SMILE inf

all_smiles = df[['SMILES']].values.squeeze().tolist()
all_ID =  df[['RTECS_ID']].values.squeeze().tolist()


# Initialise dataset
graph_data = list()
passed = list()

print("Loading {} molecules".format(len(all_smiles)))

for i in tqdm(range(len(all_smiles))):

    if all_smiles[i][0] == "":
        continue

    try:

        graph = loading.get_data(all_smiles[i], only_biggest=True, unknown_atom_is_dummy=True)

        x, a, e = loading.convert(*graph, bonds=[BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC, BondType.NOT_CONNECTED])

        graph = Graph(x=np.array(x), a=np.array(a), e=np.array(e), y=np.array(0))

        graph_data.append([graph, None])

        passed.append(all_ID[i])

    except Exception as error:

        print("Errored loading SMILES", smile)

class EvalDataset (Dataset):

    def __init__ (self, **kwargs):
        super().__init__(**kwargs)

    def read (self):
        return [x[0] for x in graph_data]

dataset = EvalDataset()
loader = BatchLoader(dataset, batch_size=batch_size, epochs=1, mask=True, shuffle=False, node_level=False)

predictions = encoder.predict(loader.load())
predictions = [[float(y) for y in x] for x in predictions]


with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    header = ['RTECS_ID'] + ['NYAN_Latent ' + str(i) for i in range(1, 65)]
    writer.writerow(header)
    for i in tqdm(range(len(passed))):
        writer.writerow([passed[i]] + predictions[i])


