#!/usr/bin/env python
# coding: utf-8

"""
trainer.py: script to train the function.
------------------------------------------
Usage:
    python -m trainer wandb.logging=True wandb.model_name="TaleSumm-ICVT" split_id=[0,1,2,3,4]
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
import wandb
import yaml
import copy
import torch
import hydra
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch.nn as nn
import numpy as np

from torch import optim
from tqdm import tqdm
from utils.logger import return_logger
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict, Union
from omegaconf import DictConfig, OmegaConf, open_dict

from utils.metrics import getScores
from utils.model_config import get_model
from dataloader.multimodal_dataset import MultiModalDataset
from utils.general_utils import (ParseEPS, seed_everything, load_yaml, save_model)

__author__ = "rodosingh"
__copyright__ = "Copyright 2023, The Story-Summarization Project"
__credits__ = ["Aditya Singh", "Rodo Singh"]
__license__ = "GPL"
__version__ = "0.1"
__email__ = "aditya.si@research.iiit.ac.in"
__status__ = "Development"

logger = return_logger(__name__)

class Trainer(object):
    """
    The trainer class to train the model and prepare data.
    """
    def __init__(self, cfg: DictConfig) -> None:
        r"""
        Train the model with the given specifications and methods and evaluate at the same
        time.
        -----------------------------------------------------------------------------------
        Args:
            - cfg: A dictionary that have extra parameters or args to pass on.
        """
        # Declare device
        if torch.cuda.is_available() and len(cfg['gpus'])>=1:
            self.device = torch.device(f"cuda:{cfg['gpus'][0]}" if torch.cuda.is_available() else 'cpu')
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')

        # Import model
        model = get_model(cfg)

        # Set the weights for different series as well as their modality
        modality = cfg['modality']
        print(f"{cfg['series']}'s Modality = {modality} is selected!\n")

        # Initialize BCE loss function with positive weights
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg[f'{cfg.series}_{modality}']]).to(self.device))

        # Scheduler and Optimizer
        if cfg['mode'] == 'training':
            # https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html
            self.optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], amsgrad=cfg["amsgrad"])
            total_steps = int(np.ceil(len(cfg['train'])/cfg['batch_size'])*cfg['epochs'])
            if cfg['lr_scheduler'] == 'onecycle':
                self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=10*cfg['lr'], total_steps=total_steps)
            elif cfg['lr_scheduler'] == 'cyclic':
                self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=cfg['lr'], max_lr=10*cfg['lr'], step_size_up=total_steps//8, cycle_momentum=False, mode='triangular2')
            else:
                raise ValueError(f"Invalid lr_scheduler (={cfg['lr_scheduler']}).")
        else:
            self.optimizer = None
            self.scheduler = None

        # wandb section
        self.wandb_logging = cfg["wandb"]["logging"] and (cfg['mode'] == 'training')
        if self.wandb_logging and (not cfg["wandb"]["sweeps"]):
            wandb.init(project=cfg["wandb"]["project"], entity=cfg["wandb"]["entity"], config=OmegaConf.to_container(cfg, resolve=True), name=cfg["wandb"]["model_name"])
        if cfg['mode'] == 'training':
            # Whether to evaluate on Test set or not
            self.eval_test = cfg["eval_test"]
            self.mode = ["train", "val", "test"] if self.eval_test else ["train", "val"]
            # All the metrics to be logged
            self.metrics_name = ["AP", "F1"]

            # wandb run name
            self.name = wandb.run.name if cfg["wandb"]["sweeps"] else cfg["wandb"]["model_name"]
            # Save model and Early stopping
            self.model_save_path = cfg["ckpt_path"]
            self.save_best_model = cfg["ES"]["save_best_model"]
            self.early_stopping = cfg["ES"]["early_stopping"]
            self.best_val_AP = float('-inf')
            if self.early_stopping or self.save_best_model:
                self.best_val_loss = float('inf')
                if modality == "both":
                    self.best_vid_val_AP = float('-inf')
                    self.best_dia_val_AP = float('-inf')
                self.ctr, self.es = 0, 0
            if self.save_best_model:
                self.model_save_path = os.path.join(self.model_save_path, self.name)
                os.makedirs(self.model_save_path, exist_ok=True)
                with open(f"{self.model_save_path}/{self.name}_config.yaml", "w") as f:
                    f.write(OmegaConf.to_yaml(ParseEPS.convert2Yamlable(copy.deepcopy(cfg)), resolve=True))
                # save_yaml(f"{self.model_save_path}/{self.name}_config.yaml", ParseEPS.convert2Yamlable(cfg.copy()))
                logger.info(f"Saved config at {self.model_save_path}{self.name}_config.yaml")

        # model section
        self.model = model.to(self.device)
        if (len(cfg["gpus"])>1 and cfg['mode'] == 'training') or \
            (cfg['mode'] == 'inference' and len(cfg["gpus"])>=1):
            self.model = nn.DataParallel(self.model, device_ids=cfg["gpus"])

        # other section
        self.cfg = cfg
        self.modality = modality
        self.epochs = cfg["epochs"]
        

    def prepare_data(self, mode:str) -> DataLoader:
        """
        Prepare train and validation (and test too) data loader.
        ------------------------------------------
        Args:
            - mode (str): Whether train, validation, or test data loader. Options: ["train", "val", "test"]

        Returns:
            - dl (Dataloader): A pytorch dataloader object.
        """
        sampling_type = self.cfg['sampling_type']
        if sampling_type == "random" and mode in ["val", "test"]:
            sampling_type = "uniform"

        common_params = {'vary_window_size': self.cfg['vary_window_size'],
                         'scene_boundary_threshold': self.cfg['scene_boundary_threshold'],
                         'window_size': self.cfg['window_size'],
                         'bin_size': self.cfg['bin_size'],
                         'withGROUP': self.cfg['withGROUP'],
                         'normalize_group_labels': self.cfg['normalize_group_labels'],
                         'which_features': self.cfg['which_features'],
                         'modality': self.modality,
                         'vid_label_type': self.cfg['vid_label_type'],
                         'dia_label_type': self.cfg['dia_label_type'],
                         'which_dia_model': self.cfg['which_dia_model'],
                         'get_word_level': self.cfg['enable_dia_encoder'],
                         'max_cap': self.cfg['max_cap'],
                         'concatenation': self.cfg['concatenation']}
        dataset = MultiModalDataset(ep_names=self.cfg[mode],
                                    sampling_type=sampling_type,
                                    **common_params)
        dl = DataLoader(dataset, batch_size=self.cfg['batch_size'], shuffle=False,
                        collate_fn=dataset.collate_fn, num_workers=self.cfg['num-workers'])
        logger.info(f"{mode.upper()} data loader prepared with {len(dataset)} samples.")
        return dl

    def scoreDict(self, scores: List[np.ndarray],
                  mode: str,
                  combine: bool,
                  prefixes:List[str],
                  suffixes: List[str])->Dict:
        r"""
        Return a dictionary of scores.
        ----------------------------------
        Args:
            - scores (List[np.ndarray]): List of scores.
            - mode (str): Whether 'train', 'val', or 'test'.
            - combine (bool): Whether to combine scores or not.
            - prefixes (List[str]): List of prefixes. Usually, ['vid_', 'dia_']
            - suffixes (List[str]): List of suffixes. Usually, ['AP', 'F1', 'F1_T']
        """
        if combine:
            assert len(prefixes) == 2, "Only two prefixes are allowed for combining scores."
            return {f"{mode}_{suffix}": np.sqrt(scores[prefixes[0][:-1]][i] * scores[prefixes[1][:-1]][i]) for i, suffix in enumerate(suffixes)}
        else:
            if any([len(prefix) == 0 for prefix in prefixes]):
                return {f"{mode}_{suffix}": scores[i] for i, suffix in enumerate(suffixes)}
            else:
                return {f"{prefix}{mode}_{suffix}": scores[prefix[:-1]][i] for prefix in prefixes for i, suffix in enumerate(suffixes)}

    def calc_loss(self, yhat: torch.Tensor, yhat_mask: torch.Tensor,
                  targets: torch.Tensor, target_mask: torch.Tensor
                 )->Tuple[torch.Tensor, List[np.ndarray], List[np.ndarray]]:
        r"""
        Calculate loss for the given yhat and targets.
        ------------------------------------------------
        Args:
            - yhat (torch.Tensor): Predictions from the model.
            - yhat_mask (torch.Tensor): Mask invalid tokens in predictions.
            - targets (torch.Tensor): Ground truth.
            - target_mask (torch.Tensor): Mask invalid tokens in ground truth.
        
        Returns:
            - loss (torch.Tensor): Loss for the given yhat and targets.
            - yhat_lst (List[np.ndarray]): List of predictions.
            - target_lst (List[np.ndarray]): List of ground truth.
        """
        B, _ = yhat.shape
        loss = 0
        yhat_lst, target_lst = [], []
        for i in range(B):
            loss += self.criterion(yhat[i][yhat_mask[i]], targets[i][target_mask[i]])
            yhat_lst.append(torch.sigmoid(yhat[i][yhat_mask[i]]).detach().cpu().numpy())
            target_lst.append(targets[i][target_mask[i]].detach().cpu().numpy())
        return loss/B, yhat_lst, target_lst

    def transformANDforward(self, data_batch: Dict)->Tuple:
        # transform data batch for video and dialogue modality to device
        if self.cfg['withGROUP']:
            feat_dict, bin_indices, token_type, mask, group_idx, subgroup_len, labels = data_batch
            # convert labels to 1D tensor
            labels = labels.to(self.device)
        else:
            feat_dict, bin_indices, token_type, mask, group_idx, subgroup_len = data_batch

        # Convert everything to device
        bin_indices = bin_indices.to(self.device)
        token_type = token_type.to(self.device)
        mask = mask.to(self.device)
        group_idx = group_idx.to(self.device)
        subgroup_len = subgroup_len.to(self.device)
        if self.modality == "both":
            vid_feat_dict, dia_feat_dict = feat_dict
        elif self.modality == "vid":
            vid_feat_dict = feat_dict
            dia_feat_dict = None
        elif self.modality == "dia":
            dia_feat_dict = feat_dict
            vid_feat_dict = None
        else:
            raise ValueError(f"Invalid modality (={self.modality}).")
        
        if self.modality != "dia":
            vid_feat_dict = {k: v.to(torch.float32).to(self.device) for k, v in vid_feat_dict.items()}
            # extract video ground truth
            if self.cfg['concatenation']:
                vid_boolean_mask = (vid_feat_dict['vid_mask'].sum(dim = -1)>0)
            else:
                if len(self.cfg['which_features']) == 1 and \
                    'mvit' in self.cfg['which_features']:
                    IC_feat = 'mvit'
                else:
                    IC_feat = 'imagenet' if 'imagenet' in self.cfg['which_features'] else 'clip'
                vid_boolean_mask = (vid_feat_dict[f'{IC_feat}_mask'].sum(dim = -1)>0)
            vid_targets = vid_feat_dict['labels']
        if self.modality != "vid":
            dia_feat_dict = {k: v.to(torch.float32).to(self.device) for k, v in dia_feat_dict.items()}
            dia_targets = dia_feat_dict['labels']
            # extract dialogue ground truth
            if self.cfg['enable_dia_encoder']:
                dia_boolean_mask = (dia_feat_dict['word_mask'].sum(dim=-1)>0)
            else:
                dia_boolean_mask = (dia_feat_dict['dia_mask']>0)
        
        # forward pass
        if self.cfg['ours_model']:
            yhat = self.model(vid_feat_dict, dia_feat_dict, bin_indices,
                              token_type, group_idx, mask, subgroup_len)
        else:
            yhat = self.model(vid_feat_dict['vid_enc'], vid_feat_dict['vid_mask'])

        # extract video and dialogue predictions
        if self.modality != "dia":
            # when decoder is not in use, then yhat structure doesn't follow
            # the same structure as the one in the case of decoder
            if self.cfg['enable_decoder']:
                vid_loss, vid_yhat_lst, vid_target_lst = \
                    self.calc_loss(yhat, (token_type == 0), vid_targets, vid_boolean_mask)
            else:
                vid_loss, vid_yhat_lst, vid_target_lst = \
                    self.calc_loss(yhat, vid_boolean_mask, vid_targets, vid_boolean_mask)

        # for dialogs
        if self.modality != "vid":
            if self.cfg['enable_decoder']:
                dia_loss, dia_yhat_lst, dia_target_lst = \
                    self.calc_loss(yhat, (token_type == 1), dia_targets, dia_boolean_mask)
            else:
                dia_loss, dia_yhat_lst, dia_target_lst = \
                    self.calc_loss(yhat, dia_boolean_mask, dia_targets, dia_boolean_mask)

        # compute loss for both modalities
        if self.modality == "both":
            loss = vid_loss + dia_loss

        # compute loss for group tokens and total loss
        if self.cfg['withGROUP'] and self.cfg['computeGROUPloss']:
            if self.cfg['enable_decoder']:
                group_loss = self.calc_loss(yhat, (token_type == 2), labels, (labels > -1))[0]
            else:
                raise NotImplementedError("Decoder not enabled for group tokens. "+
                        "MLP or some simple model is used. Change enable_decoder to True.")
            if self.modality == "both":
                loss = loss + group_loss
            elif self.modality == "vid":
                vid_loss = vid_loss + group_loss
            elif self.modality == "dia":
                dia_loss = dia_loss + group_loss

        # return loss
        if self.modality == "both":
            return loss, vid_yhat_lst, vid_target_lst, dia_yhat_lst, dia_target_lst

        elif self.modality == "vid":
            return vid_loss, vid_yhat_lst, vid_target_lst
        
        elif self.modality == "dia":
            return dia_loss, dia_yhat_lst, dia_target_lst

    def evaluate(self, val_dl: DataLoader) -> Tuple[float, Union[List[float], Dict[str, List[float]]]]:
        """
        Same as train function, but only difference is that model is freezed
        and no parameters update happen and hence no gradient updates.
        ----------------------------------------------------------------------
        Args:
            - val_dl (DataLoader): Validation data loader.
        """
        self.model.eval()
        eval_loss = 0
        if self.modality == 'both':
            vid_y_true, dia_y_true, vid_y_pred, dia_y_pred = [], [], [], []
        else:
            y_true_epoch, y_pred_epoch = [], []
        with torch.no_grad():
            for _, data_batch in enumerate(tqdm(val_dl, disable=self.cfg["wandb"]["logging"])):
                if self.modality == 'both':
                    loss, vid_yhat, vid_targets, dia_yhat, dia_targets = \
                        self.transformANDforward(data_batch)
                    vid_y_pred.extend(vid_yhat)
                    vid_y_true.extend(vid_targets)
                    dia_y_pred.extend(dia_yhat)
                    dia_y_true.extend(dia_targets)
                else:
                    loss, yhat, targets = self.transformANDforward(data_batch)
                    y_pred_epoch.extend(yhat)
                    y_true_epoch.extend(targets)
                eval_loss += loss.item()
        if self.modality == 'both':
            scores = {'vid': [*getScores(vid_y_true, vid_y_pred)],
                      'dia': [*getScores(dia_y_true, dia_y_pred)]}
        else:
            scores = [*getScores(y_true_epoch, y_pred_epoch)]
        return eval_loss/len(val_dl), scores

    def train(self)->None:
        r"""
        Train the model here.
        """
        # create data
        train_dl = self.prepare_data(mode="train")
        val_dl = self.prepare_data(mode="val")
        # training starts
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            logger.info(f"EPOCH: {epoch+1}/{self.epochs}")
            if self.modality == 'both':
                vid_y_true, vid_y_pred, dia_y_true, dia_y_pred = [], [], [], []
            else:
                y_true_epoch, y_pred_epoch = [], []
            for _, data_batch in enumerate(tqdm(train_dl, disable=self.wandb_logging)):
                self.optimizer.zero_grad()
                if self.modality == 'both':
                    loss, vid_yhat, vid_targets, dia_yhat, dia_targets = \
                        self.transformANDforward(data_batch)
                    vid_y_pred.extend(vid_yhat)
                    vid_y_true.extend(vid_targets)
                    dia_y_pred.extend(dia_yhat)
                    dia_y_true.extend(dia_targets)
                else:
                    loss, yhat, targets = self.transformANDforward(data_batch)
                    y_pred_epoch.extend(yhat)
                    y_true_epoch.extend(targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                self.scheduler.step() # CyclicLR: called on every batch
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
            if self.modality == 'both':
                train_scores = {'vid': [*getScores(vid_y_true, vid_y_pred)],
                                'dia': [*getScores(dia_y_true, dia_y_pred)]}
            else:
                train_scores = [*getScores(y_true_epoch, y_pred_epoch)]
            val_loss, val_scores = self.evaluate(val_dl)
            # self.scheduler.step(val_loss) # called on every epoch (ReduceLROnPlateau)
            epoch_loss = epoch_loss/len(train_dl)
            logger.info(f"TRAIN: loss = {epoch_loss} | VAL: loss = {val_loss}\n")
            if self.eval_test:
                test_dl = self.prepare_data(mode="test")
                test_loss, test_scores = self.evaluate(test_dl)
                logger.info(f"TEST: loss = {test_loss}\n")

            # ======================= LOG Best Metrics and REST =======================
            if self.modality == 'both':
                val_AP = np.sqrt(val_scores['vid'][0]*val_scores['dia'][0])
            else:
                val_AP = val_scores[0]
            if self.wandb_logging:
                best_happened = True if val_AP > self.best_val_AP else False
                if self.modality == 'both':
                    train_scores_dict = self.scoreDict(train_scores, "train", True, ["vid_", "dia_"], self.metrics_name)
                    val_scores_dict = self.scoreDict(val_scores, "val", True, ["vid_", "dia_"], self.metrics_name)
                    vid_dia_train_scores_dict = self.scoreDict(train_scores, "train", False, ["vid_", "dia_"], self.metrics_name)
                    vid_dia_val_scores_dict = self.scoreDict(val_scores, "val", False, ["vid_", "dia_"], self.metrics_name)
                    tmp_summary = {"train_loss": epoch_loss, **train_scores_dict, **vid_dia_train_scores_dict,
                                   "val_loss": val_loss, **val_scores_dict, **vid_dia_val_scores_dict}
                    if self.eval_test:
                        test_scores_dict = self.scoreDict(test_scores, "test", True, ["vid_", "dia_"], self.metrics_name)
                        vid_dia_test_scores_dict = self.scoreDict(test_scores, "test", False, ["vid_", "dia_"], self.metrics_name)
                        tmp_summary = {**tmp_summary, "test_loss": test_loss, **test_scores_dict, **vid_dia_test_scores_dict}
                    if best_happened:
                        best_tmp_summary = {f"best_{mod}_{m}_{metric}": tmp_summary[f"{mod}_{m}_{metric}"] for mod in ['vid', 'dia']
                                            for m in self.mode[1:] for metric in self.metrics_name[:2]}
                        best_tmp_summary.update({f"best_{m}_{metric}": tmp_summary[f"{m}_{metric}"]
                                            for m in self.mode[1:] for metric in self.metrics_name[:2]})
                        last_best = best_tmp_summary.copy()
                        best_happened = False
                    else:
                        best_tmp_summary = last_best.copy()

                else:
                    train_scores_dict = self.scoreDict(train_scores, "train", False, [''], self.metrics_name)
                    val_scores_dict = self.scoreDict(val_scores, "val", False, [''], self.metrics_name)
                    tmp_summary = {"train_loss": epoch_loss, **train_scores_dict, "val_loss": val_loss, **val_scores_dict}
                    if self.eval_test:
                        test_scores_dict = self.scoreDict(test_scores, "test", False, [''], self.metrics_name)
                        tmp_summary = {**tmp_summary, "test_loss": test_loss, **test_scores_dict}
                    if best_happened:
                        best_tmp_summary = {f"best_{m}_{metric}": tmp_summary[f"{m}_{metric}"] for m in self.mode[1:]
                                            for metric in self.metrics_name[:2]}
                        last_best = best_tmp_summary.copy()
                        best_happened = False
                    else:
                        best_tmp_summary = last_best.copy()
                tmp_summary['lr'] = self.scheduler.get_last_lr()[0]
                tmp_summary.update(best_tmp_summary)
                wandb.log(tmp_summary)
            else:
                if self.modality == "both":
                    logger.info(f"TRAIN: Vid_AP = {train_scores['vid'][0]:.3f} | Vid_F1 = {train_scores['vid'][1]:.3f} | Dia_AP = {train_scores['dia'][0]:.3f} | Dia_F1 = {train_scores['dia'][1]:.3f}")
                    logger.info(f"VAL:   Vid_AP = {val_scores['vid'][0]:.3f} | Vid_F1 = {val_scores['vid'][1]:.3f} | Dia_AP = {val_scores['dia'][0]:.3f} | Dia_F1 = {val_scores['dia'][1]:.3f}")
                    if self.eval_test:
                        logger.info(f"TEST:  Vid_AP = {test_scores['vid'][0]:.3f} | Vid_F1 = {test_scores['vid'][1]:.3f} | Dia_AP = {test_scores['dia'][0]:.3f} | Dia_F1 = {test_scores['dia'][1]:.3f}")
                else:
                    logger.info(f"TRAIN: AP = {train_scores[0]:.3f} | F1 = {train_scores[1]:.3f}")
                    logger.info(f"VAL:   AP = {val_scores[0]:.3f} | F1 = {val_scores[1]:.3f}")
                    if self.eval_test:
                        logger.info(f"TEST:  AP = {test_scores[0]:.3f} | F1 = {test_scores[1]:.3f}")

            # ======================= Do early stopping and Save Models, Logging =======================
            # TODO: Save not only the model but also the optimizer, scheduler, and other stuffs (if possible)
            if self.early_stopping or self.save_best_model:
                self.es += 1
                self.improvement_flag = False
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.improvement_flag = True
                    if self.save_best_model:
                        save_model(self.model, self.model_save_path, self.name+"_loss.pt", epoch+1, val_loss)
                if val_AP > self.best_val_AP:
                    self.best_val_AP = val_AP
                    self.improvement_flag = True
                    if self.save_best_model:
                        save_model(self.model, self.model_save_path, self.name+"_AP.pt", epoch+1, val_AP)
                if self.modality == 'both':
                    dia_val_AP = val_scores['dia'][0]
                    vid_val_AP = val_scores['vid'][0]
                    if dia_val_AP > self.best_dia_val_AP:
                        self.best_dia_val_AP = dia_val_AP
                        self.improvement_flag = True
                        if self.save_best_model:
                            save_model(self.model, self.model_save_path, self.name+"_diaAP.pt", epoch+1, dia_val_AP)
                    if vid_val_AP > self.best_vid_val_AP:
                        self.best_vid_val_AP = vid_val_AP
                        self.improvement_flag = True
                        if self.save_best_model:
                            save_model(self.model, self.model_save_path, self.name+"_vidAP.pt", epoch+1, vid_val_AP)
                self.ctr = 0 if self.improvement_flag else (self.ctr + 1)
                if self.early_stopping and self.ctr > self.cfg["ES"]["patience"]:
                    break
            else:
                if val_AP > self.best_val_AP:
                    self.best_val_AP = val_AP

        if self.wandb_logging:
            if self.early_stopping:
                wandb.config.update({"early-stopped-at": self.es, "patience": self.cfg["ES"]["patience"]})
            wandb.run.summary = tmp_summary
            wandb.finish()
        logger.info("TRAINING ENDS !!!\n\n")


def main(config: DictConfig):

    # seed everything
    seed_everything(config['seed'], harsh=True)
    # =================================== TRAINER CONFIG ===================================
    if isinstance(config['hidden_sizes'], str) and config['hidden_sizes'] == 'd_model':
        config['hidden_sizes'] = [config['d_model']]
    # ======================== parse the episode config ========================
    split_type_path = os.path.join(config["split_dir"], config["split_type"])
    logger.info(f"Split type: {config['split_type']} at {split_type_path}")
    if os.path.isfile(split_type_path):    
        episode_config = load_yaml(split_type_path)
        series_lst = ['24', 'prison-break'] if config['series'] == 'all' else config['series']
        split_dict = ParseEPS(episode_config, series=series_lst).dct
        with open_dict(config):
            config.update(split_dict)

    # See the config set...
    print(OmegaConf.to_yaml(config, resolve=True))

    if not os.path.isfile(split_type_path):
        if config['wandb']['sweeps']:
            orig_name = wandb.run.name
        else:
            orig_name = config['wandb']['model_name']
        for idx in config['split_id']:
            logger.info(f"Split {idx+1} out of {len(config['split_id'])}")
            eps_config = load_yaml(os.path.join(split_type_path, f"split{idx+1}.yaml"))
            split_dict = ParseEPS(eps_config, series=config['series']).dct
            with open_dict(config):
                config.update(split_dict)
            logger.info(f"Train Samples: {len(config['train'])} | Val Samples: {len(config['val'])}")
            if config['eval_test']:
                logger.info(f"Test Samples: {len(config['test'])}")
            if config['wandb']['sweeps']:
                wandb.run.name = orig_name + f"|S{idx+1}"
            else:
                config['wandb']['model_name'] = orig_name + f"|S{idx+1}"
            trainer = Trainer(config)
            trainer.train()
            del trainer
    else:
        trainer = Trainer(config)
        trainer.train()
        del trainer


def sweep_agent_manager():
    r"""
    Sweep agent manager to run the sweep.
    """
    wandb.init()
    config = wandb.config
    wd = config['weight_decay']
    ams = config['amsgrad']
    lr = config['lr']
    lrs = config['lr_scheduler']
    feat_fusion_style = config['feat_fusion_style']
    epochs = config['epochs']
    wandb.run.name = (f"SWP26|E{epochs}|WD{wd}|AMS{ams}|LR{lr}|LRS{lrs}|{feat_fusion_style}")
    main(dict(config))

@hydra.main(config_path="./configs", config_name="trainer_config", version_base='1.3')
def driver(cfg: DictConfig):
    if cfg.wandb.sweeps:
        wandb.agent(sweep_id=cfg["wandb"]["sweep_id"],
                    function=sweep_agent_manager,
                    count=cfg["wandb"]["sweep_agent_run_count"])
    else:
        main(cfg)

if __name__ == '__main__':
    driver()
