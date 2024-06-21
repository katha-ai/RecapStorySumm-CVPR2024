#!/usr/bin/env python

"""
dia_dataset.py: Dialogue Dataset for training and inference.
"""

import torch
import numpy as np

from pathlib import Path
from typing import Dict, List, Union
from utils.logger import return_logger
from utils.general_utils import load_pickle, get_file_name
from torch.utils.data import Dataset

logger = return_logger(__name__)


class DialogueDataset(Dataset):
    """
    Given list of Episodes, this module will try to
    extract encodings for each dialogue as well as
    their respective labels.
    ------------------------------------------------------
    Note: The episodes naming format would be EXX which
    should be in increasing order only and by default the Ground
    Truth labels would be imported according to the episode one
    ahead of the latest episode in the given list. 
    """

    def __init__(self,
                 ep_names: List[Path],
                 get_word_level: bool,
                 model_name: str,
                 label_type: str,
                 **kwargs
                 ) -> None:
        """
        Args:
            - ep_names: List of names of EPISODES that are to be used
              for model training or inference.
            - get_word_level: Whether to extract word level encodings or
              dialogue level encodings.
            - model_name: Name of the model that was used to generate
              encodings.
            - label_type: Type of labels to be used. It can be either
                `GT` or `SLV` or `H` or `F`.
        --------------------------------------------------------------------------------
        """
        super(DialogueDataset, self).__init__(**kwargs)
        self.ep_names = ep_names
        self.get_word_level = get_word_level
        self.model_name = model_name
        self.label_type = label_type
    
    def parse_word_encodings(self, eps_file: Union[str, Path]) -> np.ndarray:
        """
        For the given episode, this function will extract the
        encodings for each word of dialogue and will return an
        array of shape (`no. of sentences`, `max_seq_len`, `enc_dim`).
        Where:
            - `no. of sentences` is number of sentences,
            - `max_seq_len` is max no. of words can exist in a sentence, whereas
            - `enc_dim` is the embedding dimension of each word.

        Return:
            - `encodings`: Array of shape (`no. of sentences`, `max_seq_len`, `enc_dim`).
            - `mask`: Array of shape (`no. of sentences`, `max_seq_len`).
               This array will be used to mask the padded values.
        """
        enc_dict = load_pickle(eps_file)
        max_sent_len = max([v.shape[0] for v in enc_dict.values()])
        mask = np.zeros((len(enc_dict), max_sent_len))
        # by default, dict is ordered by keys in ascending order
        # so, we can use the same order to pad the encodings
        for i, (k, v) in enumerate(enc_dict.items()):
            if v.shape[0] < max_sent_len:
                enc_dict[k] = np.pad(v, ((0, max_sent_len - v.shape[0]), (0, 0)))
                mask[i, :v.shape[0]] = 1
            else:
                mask[i, :] = 1
        return np.stack(list(enc_dict.values())), mask

    def __len__(self):
        return len(self.ep_names)

    def __getitem__(self, idx:int)->Dict[str, np.ndarray]:
        if self.get_word_level:
            eps_file_name = get_file_name(self.ep_names[idx]/fr"encodings/dia_encodings/{self.model_name}_episode_*.pkl")
            eps_ENC, eps_mask = self.parse_word_encodings(eps_file_name)
        else:
            eps_file_name = get_file_name(self.ep_names[idx]/fr"encodings/dia_encodings/{self.model_name}_episode_*.npy")
            eps_ENC = np.load(eps_file_name).squeeze()
        eps_start = int(eps_file_name.stem.split("_")[-1])
        next_ep = "E{0:02n}".format(int(self.ep_names[idx].stem[-2:]) + 1)
        parent_dir = self.ep_names[idx].parent
        gt_file_name = self.label_type.upper() + ".pkl"
        labels = load_pickle(parent_dir/f"{parent_dir.stem}{next_ep}/scores/dia_scores"/\
                            f"recapVepisode/{gt_file_name}")[self.ep_names[idx].stem[-3:]]
        if gt_file_name == 'GT.pkl':
            labels = labels[eps_start:]
        assert eps_ENC.shape[0] == labels.shape[0],\
            (f"Labels and Encoding mismatch in shape. Enc shape = {eps_ENC.shape}; Labels Shape = {labels.shape}")
        if self.get_word_level:
            return {'dia_enc': eps_ENC, 'word_mask': eps_mask, 'labels': labels}
        return {'dia_enc': eps_ENC, 'labels': labels}

    def collate(self, data_batch: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        `collate` function for textual (dialogue) Dataset -> says -> how to
        (and in what shape) we need to provide data to model.
        """
        # lengthy episode in terms of no. of dialogues
        longest_ep_len = max([ep['dia_enc'].shape[0] for ep in data_batch])
        if self.get_word_level:
            # also the lengthiest sentence in terms of no. of words
            longest_sent_len = max([ep['dia_enc'].shape[1] for ep in data_batch])

        for episode in data_batch:
            padLen = longest_ep_len - episode['labels'].shape[0]
            if self.get_word_level:
                word_padLen = longest_sent_len - episode['dia_enc'].shape[1]
                episode['dia_enc'] = np.pad(episode['dia_enc'], ((0, padLen), (0, word_padLen), (0, 0)))
                episode['word_mask'] = np.pad(episode['word_mask'], ((0, padLen), (0, word_padLen)))
            else:
                # mask should be padded first as `dia_enc` shape is used to pad mask
                episode['dia_mask'] = np.hstack((np.ones(episode['dia_enc'].shape[0]), np.zeros(padLen)))
                episode['dia_enc'] = np.pad(episode['dia_enc'], ((0, padLen), (0, 0)))
            episode['labels'] = np.pad(episode['labels'], ((0, padLen)))
        keys = list(data_batch[0].keys())
        batch_dict = {k: torch.from_numpy(np.stack([ep[k] for ep in data_batch])) for k in keys}
        return batch_dict

if __name__ == "__main__":
    # test code
    dataset = DialogueDataset(ep_names=[Path("../../data/24/S02/S02E09"),
                                        Path("../../data/24/S03/S03E03")],
                              get_word_level=True,
                              model_name="pegasus-large",
                              label_type="SLV",
                              condition_on_current=False)
    t1 = dataset[0]
    t2 = dataset[1]
    batch = dataset.collate([t1, t2])
    # import ipdb; ipdb.set_trace(context=15)
