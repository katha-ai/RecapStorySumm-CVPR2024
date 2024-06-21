#!/usr/bin/env python
# coding: utf-8

"""
simple_inference.py: Inference for a single video...
------------------------------------------
Usage:
    python -m inference model_name='TaleSumm-ICVT|S1' gpus=[0]
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"
import torch
import hydra

import numpy as np
from pathlib import Path
from typing import Union
from trainer import Trainer
from omegaconf import DictConfig, OmegaConf
from utils.metrics import getScores
from utils.logger import return_logger
from utils.general_utils import ParseEPS
logger = return_logger(__name__)

class Inference(Trainer):
    r"""Class for inference."""
    def __init__(self, config: DictConfig)->None:
        super(Inference, self).__init__(config)
        self.model = self.model.to(self.device)
        self.config = config

    def inference(self, state_dict_path: Union[str, Path])->None:
        r"""
        Does inference on the test set.
        -----------------------------------------------------------------------------------
        Args:
            - state_dict_path (str or Path): Path to the state dict of the model.
        """
        # Update the config
        self.config['batch_size'] = 1

        # Load the model
        self.model.load_state_dict(torch.load(state_dict_path, map_location=self.device))
        self.model.eval()
        # prepare data
        test_dl = self.prepare_data(mode="test")
        if self.modality == 'both':
            vid_AP_lst, dia_AP_lst = [], []
            vid_F1_lst, dia_F1_lst = [], []
        else:
            AP_lst, F1_lst = [], []
        # inference starts
        with torch.no_grad():
            for idx, data_batch in enumerate(test_dl):
                if self.modality == 'both':
                    loss, vid_yhat, vid_targets, dia_yhat, dia_targets = \
                        self.transformANDforward(data_batch)
                    vid_score = getScores(vid_targets, vid_yhat, f1_threshold=0.5)
                    dia_score = getScores(dia_targets, dia_yhat, f1_threshold=0.5)
                    logger.info(f"EPISODE: {idx+1}/{len(test_dl)} = {self.config['test'][idx].stem}")
                    logger.info(f"Loss: {loss.item():.5f}")
                    logger.info(f"Vid Scores: AP: {vid_score[0]:.5f} | F1: {vid_score[1]:.5f}")
                    logger.info(f"Dia Scores: AP: {dia_score[0]:.5f} | F1: {dia_score[1]:.5f}\n")
                    vid_AP_lst.append(vid_score[0]); vid_F1_lst.append(vid_score[1])
                    dia_AP_lst.append(dia_score[0]); dia_F1_lst.append(dia_score[1])
                else:
                    loss, yhat, targets = self.transformANDforward(data_batch)
                    score = getScores(targets, yhat, f1_threshold=0.5)
                    logger.info(f"Loss: {loss.item():.5f}")
                    logger.info(f"Scores: AP: {score[0]:.5f} | F1: {score[1]:.5f}\n")
                    AP_lst.append(score[0]); F1_lst.append(score[1])
        if self.modality == 'both':
            logger.info(f"VID: AP: {np.mean(vid_AP_lst):.5f} | F1: {np.mean(vid_F1_lst):.5f}")
            logger.info(f"DIA: AP: {np.mean(dia_AP_lst):.5f} | F1: {np.mean(dia_F1_lst):.5f}")
        else:
            logger.info(f"AP: {np.mean(AP_lst):.5f} | F1: {np.mean(F1_lst):.5f}")

@hydra.main(config_path="configs/", config_name="inference_config", version_base='1.3')
def main(cfg: DictConfig):
    # NOTE: Do note that, you need to give a split_type_file which is a file (not a directory like intra-loocv)
    # NOTE: Also ensure atleast one episode to be in val set (unless u will hit zero-division error)
    if isinstance(cfg['hidden_sizes'], str) and cfg['hidden_sizes'] == 'd_model':
        cfg['hidden_sizes'] = [cfg['d_model']]
    model_config = OmegaConf.load(cfg.model_config_path)
    if cfg.split_type_file is not None:
        if os.path.isfile(cfg.split_type_file):
            split_info = OmegaConf.load(cfg.split_type_file)
            series_lst = ['24', 'prison-break'] if cfg['series'] == 'all' else cfg['series']
            split_info = ParseEPS(split_info, series=series_lst).dct
            model_config = OmegaConf.merge(model_config, split_info)
        else:
            raise FileNotFoundError(f"Split file not found at {cfg.split_type_file}")
    cfg = OmegaConf.merge(model_config, cfg)
    cfg = ParseEPS.convert2Path(cfg)
    infer = Inference(cfg)
    infer.inference(cfg['state_dict_path'])
    del infer

if __name__ == "__main__":
    main()