#!/usr/bin/env python
# coding: utf-8

"""
Compare labels from different sources.
------------------------------------------
Usage:
    python -m main.compareLabels
"""
import pandas as pd
import numpy as np

from typing import Tuple, List, Union
from pathlib import Path
from utils.general_utils import load_pickle, ParseEPS, get_file_name, load_yaml
from utils.logger import return_logger
from utils.metrics import getScores


logger = return_logger(__name__)

def path2names(path:Path)->Tuple[str, str, str]:
    r"""
    Convert path to series name, season, episode.
    ------------------------------------------
    Args:
        - path: path to episode
    Returns:
        - Dict[str, str]: series name, season, episode
    """
    ep = path.stem[-3:]
    season = path.parent.stem
    series = path.parent.parent.stem
    return series, season, ep

def randomScoreComputer(labels:np.ndarray, trials: int)->Tuple[float, float]:
    r"""
    Compute random scores.
    ------------------------------------------
    Args:
        - labels: labels
        - trials: number of trials to average over
    Returns:
        - Tuple[float, float]: mean AP, mean F1
    """
    score = []
    for _ in range(trials):
        preds = np.random.random_sample(size=len(labels))
        score.append(list(getScores([labels], [preds])))
    return np.mean(score, axis=0)

def trainRandomBaseline(modality: str,
                        trials: int,
                        split_file: Path = Path("configs/data_configs/splits/default_split.yaml"),
                        series: Union[str, List[str]] = '24',
                        seed: int = 0
                       ) -> pd.DataFrame:
    r"""
    Predict random scores within [0, 1] range.
    ------------------------------------------
    Args:
        - modality: 'vid' or 'dia'
        - trials: number of trials to average over
        - split_file: path to split file that contains train, val, test splits
        - series: '24' or 'prison-break'
        - seed: random seed
    """

    episode_config = load_yaml(split_file)
    split_dict = ParseEPS(episode_config, series=[series]).dct
    modes = ['val', 'test']
    mode_scores = {mode: {'AP': [], 'F1': []} for mode in modes}
    np.random.seed(seed)
    for mode in modes:
        for eps_path in split_dict[mode]:
            ser, seas, ep = path2names(eps_path)
            labels = prepareLabels('MSR', modality, ser, seas, ep)
            scores = randomScoreComputer(labels, trials)
            mode_scores[mode]['AP'].append(scores[0])
            mode_scores[mode]['F1'].append(scores[1])
    for mode in modes:
        mode_scores[mode]['AP'] = [np.mean(mode_scores[mode]['AP'])*100, np.std(mode_scores[mode]['AP'])*100]
        mode_scores[mode]['F1'] = [np.mean(mode_scores[mode]['F1'])*100, np.std(mode_scores[mode]['F1'])*100]
    df = pd.DataFrame.from_dict(mode_scores)
    # df.to_csv(f"scores/rand_base_PB_{modality}_{which_labels}_{trials}.csv")
    return df

def prepareLabels(label_type: str, modality: str, series: str,
                  season: str, ep: str, model_pred_path: Path = None)->np.ndarray:
    r"""
    Prepare labels for evaluation.
    ------------------------------------------
    Args:
        - label_type: 'MSR', 'H', 'F', 'Model', 'MR'
        - modality: 'vid' or 'dia'
        - series: '24' or 'prison-break'
        - season: 'S01', 'S02', ...
        - ep: 'E01', 'E02', ...
        - model_pred_path: path to model predictions if `label_type` is 'Model' else None
    Returns:
        - np.ndarray: labels
    """
    next_ep = "E%02d" % (int(ep[-2:]) + 1)
    if label_type.lower() in ['msr', 'mr']:
        file_name = 'GT' if label_type.lower() == 'mr' else 'SL' if modality == 'vid' else 'SLV'
        if modality == 'vid':
            labels = load_pickle(f"data/{series}/{season}/{season}{next_ep}/scores/{modality}_scores/recapVepisode/{file_name}.pkl")[ep]
        else:
            labels = load_pickle(f"data/{series}/{season}/{season}{next_ep}/scores/dia_scores/recapVepisode/{file_name}.pkl")[ep]
            if file_name == 'GT':
                file_path = get_file_name(f"data/{series}/{season}/{season}{ep}/encodings/dia_encodings/newSrtObj_*.pkl")
                start = int(file_path.stem.split("_")[-1]) + 1
                labels = labels[start:]
                
    elif label_type.lower() in ['h','f']:
        assert str(series) == '24', "Only 24 series is supported for now."
        labels = load_pickle(f"data/{series}/{season}/{season}{next_ep}/scores/{modality}_scores/recapVepisode/{label_type.upper()}.pkl")[ep]
    elif label_type.lower() == 'model':
        labels = np.load(model_pred_path)
    else:
        raise ValueError(f"Unknown label type: {label_type}")
    return labels

def compareLabels(modality: str = 'vid',
                  label_type1: str = 'MSR',
                  label_type2: str = 'F',
                  model_pred_path: Path = None,
                  series: str = '24', season: str = 'S01', episode: str = 'E01'
                 )->pd.DataFrame:
    r"""
    Compare `which_labels` against `which_pred_labels`.
    ------------------------------------------
    Args:
        - modality: 'vid' or 'dia'
        - label_type1: 'MSR', 'H', 'F', 'Model', 'MR'
        - label_type2: 'MSR', 'H', 'F', 'Model', 'MR'
        - model_pred_path: path to model predictions if `label_type1` or
          `label_type2` is 'Model' else `None`
        - series: '24' or 'prison-break'
    Returns:
        - pd.DataFrame: AP and F1 scores
    """

    label1 = prepareLabels(label_type1, modality, series, season, episode, model_pred_path)
    label2 = prepareLabels(label_type2, modality, series, season, episode, model_pred_path)
    scores = getScores([label1], [label2])

    df = pd.DataFrame(data=np.array(scores).reshape(1,2), columns=['AP', 'F1'])
    return df

if __name__ == '__main__':
    from IPython.display import display
    # ------------------ PARAMETERS TO CHANGE ------------------
    eval_random_baseline = False
    label_type1 = 'MR' # 'MSR', 'H', 'F', 'Model', 'MR'
    label_type2 = 'H' # 'MSR', 'H', 'F', 'Model', 'MR'
    modality = 'vid' # 'vid' or 'dia'
    # ----------------------------------------------------------
    if eval_random_baseline:
        print(f"Evaluating random baseline for {modality}...")
        seed = 0
        # If computing random-baseline scores for Intra-CVT
        split_file_paths = [Path("configs/data_configs/splits/intra-loocv/split1.yaml"),
                            Path("configs/data_configs/splits/intra-loocv/split2.yaml"),
                            Path("configs/data_configs/splits/intra-loocv/split3.yaml"),
                            Path("configs/data_configs/splits/intra-loocv/split4.yaml"),
                            Path("configs/data_configs/splits/intra-loocv/split5.yaml")]
        # Else for cross-series
        # split_file_paths = [Path("configs/data_configs/splits/cross-series.yaml")]
        trials = 1000
        for sp_pth in split_file_paths:
            df = trainRandomBaseline(modality=modality,
                                    trials=trials,
                                    split_file=sp_pth,
                                    series='24',
                                    seed=seed)
            # show df in terminal
            df.style.set_properties(**{'text-align': 'center'})
            df.style.format("{:.2f}")
            df.style.set_caption(f"Comparison of labels - {sp_pth.stem}")
            df.style.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
            print(f"Comparison of labels - {sp_pth.stem}")
            display(df)
    else:
        print(f"Comparing labels for modality: {modality} with Label type: {label_type1}, Label type2: {label_type2}")
        fandom_eps = load_yaml("configs/data_configs/fandom_time.yaml")
        for seas in fandom_eps['24']:
            for ep in fandom_eps['24'][seas]:
                df = compareLabels(modality=modality, label_type1=label_type1, label_type2=label_type2, season=seas, episode=ep)
                # concat df
                df_all = df if 'df_all' not in locals() else pd.concat([df_all, df], axis=0, ignore_index=True)
        # show df in terminal
        display(df_all)
        print(f"Avg AP: {df_all['AP'].mean():.4f} +/- {df_all['AP'].std():.4f}")
                
