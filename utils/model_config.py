#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
This file load model as per user config.
"""

from typing import Callable

from utils.logger import return_logger
from models.talesumm.story_sum import StorySum
from models.published_baselines.msva_model import MSVA_adapted
from models.published_baselines.pgl_sum_model import PGL_Sum_Adapted

logger = return_logger(__name__)

def get_model(config: dict)->Callable:
    """
    As per USER config return the required model.
    """
    
    if config["ours_model"]:
        kwargs = {"modality": config["modality"],
                "which_features": config["which_features"],
                "which_dia_model": config["which_dia_model"],
                "vid_concatenate": config["concatenation"],
                "withGROUP": config["withGROUP"],
                "enable_encoder": config["enable_encoder"],
                "encoder_type": config["encoder_type"],
                "enable_dia_encoder": config["enable_dia_encoder"],
                "dia_encoder_type": config["dia_encoder_type"],
                "pool_location": config["pool_location"],
                "enable_decoder": config["enable_decoder"],
                "attention_type": config["attention_type"],
                "differential_attention": config["differential_attention"],
                "differential_attention_type": config["differential_attention_type"],
                "max_groups": config["max_groups"],
                "d_model": config['d_model'],
                "ffn_ratio": config["ffn_ratio"],
                "enc_layers": config["enc_layers"],
                "dec_layers": config["dec_layers"],
                "enc_num_heads": config["enc_num_heads"],
                "dec_num_heads": config["dec_num_heads"],
                "drop_proj": config["drop_proj"],
                "drop_trm": config["drop_trm"],
                "drop_fc": config["drop_fc"],
                "feat_fusion_style": config["feat_fusion_style"],
                "activation_trm": config["activation_trm"],
                "activation_mlp": config["activation_mlp"],
                "activation_clf": config["activation_clf"],
                "mlp_hidden_sizes": config["hidden_sizes"],
                "init_weights": config["init_weights"],
                "max_pos_enc_len": config["max_pos_enc_len"]}
        
        # loading the model
        model = StorySum(**kwargs)
        if config['verbose']:
            if config["modality"] == "vid":
                if config["enable_decoder"]:
                    logger.info(f"Video: Shot-Level {config['encoder_type']} with concatenation={config['concatenation']} along " +
                                f"with Episode level TRM using {config['which_features']} is loaded.")
                else:
                    logger.info(f"Video: Shot-Level {config['encoder_type']} with concatenation={config['concatenation']} of {config['which_features']}" +
                                " along with MLP at End is loaded.")
            elif config["modality"] == "dia":
                if config["enable_dia_encoder"] and not config["enable_decoder"]:
                    logger.info(f"Dialog: Word-level {config['dia_encoder_type']} using {config['which_dia_model']}" +
                                " encodings is loaded with MLP Classifier head.")
                elif config["enable_dia_encoder"] and config["enable_decoder"]:
                    logger.info(f"Dialog: Word-level {config['dia_encoder_type']} along with Utterance-Level TRM " +
                                f"(the second-level) using {config['which_dia_model']} encodings is loaded.")
                elif not config["enable_dia_encoder"] and config["enable_decoder"]:
                    logger.info(f"Dialog: Utterance-Level TRM (the second-level) using {config['which_dia_model']}" +
                                " encodings is loaded.")
                else:
                    logger.info(
                        f"Dialog: SIMPLE BASELINE using {config['which_dia_model']} with MLP as classification head is loaded.")
            elif config["modality"] == "both":
                if config["enable_dia_encoder"]:
                    logger.info(f"Video+Dialog: Shot-Level {config['encoder_type']} & Word-level {config['dia_encoder_type']} " +
                                f" along with Episode level TRM using {config['which_features']} " +
                                f"video features (concatenation={config['concatenation']})" +
                                f" and {config['which_dia_model']} word-level features" +
                                f" along with Attention type={config['attention_type']}" +
                                f", differential attention as {config['differential_attention']}" + 
                                f" with type {config['differential_attention_type']}" +
                                f", withGROUP={config['withGROUP']} is loaded.")
                else:
                    logger.info(f"Video+Dialog: Shot-Level {config['encoder_type']} / Sentence-level pretrained TRM  " +
                                f" with {config['which_dia_model']} dialog features" +
                                f" along with Episode level TRM using {config['which_features']} " +
                                f"video features (concatenation={config['concatenation']})" +
                                f" along with Attention type={config['attention_type']}" +
                                f", differential attention as {config['differential_attention']}" + 
                                f" with type {config['differential_attention_type']}" +
                                f", withGROUP={config['withGROUP']} is loaded.")
        else:
            logger.info("Model is loaded.")
    else:
        if config['pb'] == "pgl":
            logger.info(f"Published Baseline: PGL-Sum with {config['which_features']} is loaded.")
            feat_dim_dict = {"imagenet": 1664, "mvit": 768, "clip": 512,
                             'googlenet': 1024, 'i3d_flow': 1024, 'i3d_rgb': 1024}
            dim = sum([feat_dim_dict[feat] for feat in config["which_features"]])
            kwargs = {"feat_inp_dim": dim, "drop_proj": config["drop_proj"]}
            model = PGL_Sum_Adapted(**kwargs)
        else:
            logger.info(f"Published Baseline: MSVA with IMC is loaded.")
            kwargs = {"d_model": config['d_model'], "first_feat_dim": config['first_feat_dim'],
                      "second_feat_dim": config['second_feat_dim'], "third_feat_dim": config['third_feat_dim'],
                      "drop_proj": config["drop_proj"], "drop_trm": config["drop_trm"]}
            model = MSVA_adapted(**kwargs)

    return model

if __name__ == '__main__':
    print("Hello World Check!")
