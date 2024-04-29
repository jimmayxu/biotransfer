# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).

from copy import copy, deepcopy
import math
from pathlib import Path
import random
import os
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection

import pandas as pd
import numpy as np
import scipy
from tape.datasets import dataset_factory
from tape.datasets import pad_sequences
from tape.tokenizers import TAPETokenizer
from tape.registry import registry
from tape import ProteinBertForValuePrediction
import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
import itertools

from sklearn.preprocessing import MinMaxScaler, StandardScaler


class BCRDataset(Dataset):


    def __init__(self,
                 chain: str,
                 split: str,
                 data_path: Union[str, Path] = "/home/gridsan/groups/ai4bio_shared/datasets/AAbio_data/covid/",
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 average_replicates: bool = False,
                 add_static_ends: bool = False,
                 remove_static: bool = False,
                 variable_regions: bool = False,
                 filter_nan: str = "drop",
                 correction: str = "assay",
                 collate_type: str = "batch_padded",
                 token_encode_type: str = "embed",
                 ):
        """Inits dataset

        Args:
            chain [14H, 14L, 91H, 95L]: The backbone peptide chain to load in. Can be 14H, 14L, 91H, or 95L. Each has around the same amount of data.
            split [train, valid, test]: Train/valid/test split. Percentages are .8/.1/.1, with equal proportions of kmutations and affinity scores.
            data_path: Path to the covid data directory. Default one leads to shared llgrid location.
            tokenizer: Encodes amino acid characters into ids and appends start/end tokens. Default uses tape's iupac tokenizer.
            average_replicates: Every independent peptide has 3 trials of measurements. If False, all measurements are kept as independent
                                 samples. If True, all peptide trials are averaged into one sample.
            add_static_ends: The peptides within the data are missing static sequences on the beginning and end from the original sequence. If True,
                             this adds those static beginnings/ends back to the data. If False, leaves data as is.
            remove_static: The peptides in the data have various dynamic regions which were mutated for experiments, and regions which were kept 
                           completely static during experiments. If True, this removes the static regions, leaving only the dynamic ones. If False,
                           data is kept as is.
            filter_nan [drop, max]: How to handle pred_aff nans in data. Can be dropped or filled with the max value.

        """

        # Record initialization args
        self._init_kwargs = locals()
        self._init_kwargs.pop("self")

        # Assertions
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        #assert chain in ["IGH"]
        self.chain = chain
        self.variable_regions = variable_regions

        if data_path == None:
            return

        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. "
                             f"Must be one of ['train', 'valid', 'test']")

        if add_static_ends and remove_static:
            raise Exception("Cannot add static ends and remove static at the same time. Only one of these options should be True.")


        self.split = split
        self.average_replicates = average_replicates
        self.add_static_ends = add_static_ends
        self.remove_static = remove_static
        self.filter_nan = filter_nan

        # Load dataframe
        self.data_path = data_path
        if self.data_path[-1] != "/":
            self.data_path += "/"

        if self.filter_nan in ['median']:
            data_file = self.data_path + '{}/{}_{}.csv'.format(chain, chain, 'train')
            data_ = [pd.read_csv(data_file)]
            data_file = self.data_path + '{}/{}_{}.csv'.format(chain, chain, 'valid')
            data_.append(pd.read_csv(data_file))
            data_file = self.data_path + '{}/{}_{}.csv'.format(chain, chain, 'test')
            data_.append(pd.read_csv(data_file))
            data_all = pd.concat(data_)

            #target_scaler = StandardScaler()
            #target_scaler.fit(np.array([list(data_all["pred_aff"])]).T)  

        data_file = self.data_path + '{}/{}_{}.csv'.format(chain, chain, split)
        data = pd.read_csv(data_file)
        """
        if split == 'test':
          data = data.groupby(['mata_description']).filter(lambda x: len(x.dropna())>0)
        """




        # If "max", replace nans with max affinity values
        # If "drop", drop nans
        if self.filter_nan == "drop":
            data = data.dropna()
        elif self.filter_nan == 'max':
            max_val = data["pred_aff"].max()
            data = data.fillna(max_val)
        elif self.filter_nan == 'median':
            data_all = data_all.dropna()
            grouped_data = data_all.groupby(['mata_description'])
            grouped_data = grouped_data.filter(lambda x: len(x) < 3)
            median_val = grouped_data["pred_aff"].median()
            data = data.fillna(median_val)
        else:
            print('Error')
            import sys
            sys.exit()
        """
        # If true, average replicates over the same sequence and drop extra replicates
        if self.average_replicates:
            data['pred_aff'] = data.groupby(['mata_description']).pred_aff.transform('mean')
            data = data.drop_duplicates(subset=['mata_description'])
        
        # If true, adds static regions at the beginning and end from the original backbones to the sequences
        if self.add_static_ends:
            data_orig = data.copy(deep=True)
            data["aa_seq"] = data.aa_seq.transform(self.add_static_ends_transform)
        
        # If true, removes any static regions from the sequences. Bad for language modeling, potentially good for VAE.
        if self.remove_static:
            data["aa_seq"] = data.aa_seq.transform(self.remove_static_transform)
        """


        # Only use sequence and label
        #data['pred_aff'] = target_scaler.transform(np.array([list(data['pred_aff'])]).T).T[0]
        self.region = data.loc[:, ['fwr1', 'cdr1', 'fwr2', 'cdr2', 'fwr3', 'cdr3', 'fwr4']]


        # If batch_padded, pads each sample to max length in current batch
        # If full_padded, pads each sample to the max length in dataset
        assert collate_type in ["batch_padded", "full_padded"]
        self.collate_type = collate_type

        self.token_encode_type = token_encode_type
        self.pca_model = None

        self.data = data.to_dict('records')

        if self.variable_regions:
            self.regions = self.get_variable_regions()


    def __len__(self) -> int:
        """Returns length of dataset"""
        return len(self.data)

    def __getitem__(self, index: int):
        """Returns sample from dataset at specific index

        Args:
            index (int): Index of dataset
        """
        #item = self.data.iloc[index][["aa_seq", "pred_aff"]]
        item = self.data[index]
        tokens = self.tokenizer.tokenize(item["aa_seq"])
        tokens = self.tokenizer.add_special_tokens(tokens)
        token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(tokens), np.int64)
        input_mask = np.ones_like(token_ids)
        if "pred_aff" in item:
            input_label = float(item["pred_aff"]) #np.sign(float(item["pred_aff"]))*np.power(abs(float(item["pred_aff"])),0.5)
        else:
            input_label = None

        if self.variable_regions:
            input_region = self.regions[index]
        else:
            input_region = None
        return token_ids, input_mask, input_label, input_region




    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        """Turns list of samples into batch that can be fed to a model

        Args:
            batch (List[Any]): A list of samples to be turned into a batch.

        Returns:
            A batch corresponding to the given list of samples
        """
        input_ids, input_masks, stability_true_value, input_region = tuple(zip(*batch))
        if self.collate_type == "batch_padded":
            input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
            input_masks = torch.from_numpy(pad_sequences(input_masks, 0))
        else:
            print()
            input_ids = [np.pad(input_id, (0, 0), "constant", constant_values=(0,0)) for input_id in input_ids]
            input_ids = np.stack(input_ids)
            input_ids = torch.from_numpy(input_ids)


            input_masks = [np.pad(input_mask, (0, 0), "constant", constant_values=(0,0)) for input_mask in input_masks]
            input_masks = np.stack(input_masks)
            input_masks = torch.from_numpy(input_masks)

        if self.token_encode_type == "one_hot":
            input_ids = one_hot(input_ids, len(set(self.tokenizer.vocab.values())))
            input_ids = input_ids[:,self.variable_regions,:]
            input_masks = input_masks[:,self.variable_regions]
            input_ids = input_ids.permute(0, 2, 1)
            input_ids = input_ids.type(torch.float32)

        stability_true_value = torch.FloatTensor(stability_true_value)
        stability_true_value = stability_true_value.unsqueeze(1)



        return {'input_ids': input_ids,
                'input_masks': input_masks,
                'targets': stability_true_value,
                "input_region": list(input_region)}

    def get_variable_regions(self):
        """
        NOT FINISHED

        Returns list of indices representing the variable regions of the
           dataset chain type.
        """
        Variable_regions = list()
        for _, r in self.region.iterrows():
            var = [(r['fwr1'], r['cdr1']), (r['fwr2'], r['cdr2']), (r['fwr3'], r['cdr3'])]
            variable_regions = list(itertools.chain(*[list(range(i,j)) for i,j in var]))
            Variable_regions.append(variable_regions)


        return pd.DataFrame(Variable_regions).values.tolist()

