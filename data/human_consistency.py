#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Avg. Cronbach Alpha and pairwise F1 score for each episode
using labels from different sources.
Note: Fleiss-Kappa code is taken from
https://www.statsmodels.org/stable/_modules/statsmodels/stats/inter_rater.html#fleiss_kappa
------------------------------------------
Usage:
    python -m main.human_consistency
"""

# Compute CronBach Alpha for each episode given the labels
import numpy as np
import pandas as pd
from typing import List, Tuple, Union
from sklearn.metrics import f1_score
from data.compareLabels import prepareLabels
from utils.general_utils import load_yaml, load_pickle

def aggregate_raters(data, n_cat=None):
    '''convert raw data with shape (subject, rater) to (subject, cat_counts)

    brings data into correct format for fleiss_kappa

    bincount will raise exception if data cannot be converted to integer.

    Parameters
    ----------
    data : array_like, 2-Dim
        data containing category assignment with subjects in rows and raters
        in columns.
    n_cat : None or int
        If None, then the data is converted to integer categories,
        0,1,2,...,n_cat-1. Because of the relabeling only category levels
        with non-zero counts are included.
        If this is an integer, then the category levels in the data are already
        assumed to be in integers, 0,1,2,...,n_cat-1. In this case, the
        returned array may contain columns with zero count, if no subject
        has been categorized with this level.

    Returns
    -------
    arr : nd_array, (n_rows, n_cat)
        Contains counts of raters that assigned a category level to individuals.
        Subjects are in rows, category levels in columns.
    categories : nd_array, (n_category_levels,)
        Contains the category levels.

    '''
    data = np.asarray(data)
    n_rows = data.shape[0]
    if n_cat is None:
        #I could add int conversion (reverse_index) to np.unique
        cat_uni, cat_int = np.unique(data.ravel(), return_inverse=True)
        n_cat = len(cat_uni)
        data_ = cat_int.reshape(data.shape)
    else:
        cat_uni = np.arange(n_cat)  #for return only, assumed cat levels
        data_ = data

    tt = np.zeros((n_rows, n_cat), int)
    for idx, row in enumerate(data_):
        ro = np.bincount(row)
        tt[idx, :len(ro)] = ro

    return tt, cat_uni

def fleiss_kappa(table, method='fleiss'):
    r"""Fleiss' and Randolph's kappa multi-rater agreement measure

    Parameters
    ----------
    table : array_like, 2-D
        assumes subjects in rows, and categories in columns. Convert raw data
        into this format by using
        :func:`statsmodels.stats.inter_rater.aggregate_raters`
    method : str
        Method 'fleiss' returns Fleiss' kappa which uses the sample margin
        to define the chance outcome.
        Method 'randolph' or 'uniform' (only first 4 letters are needed)
        returns Randolph's (2005) multirater kappa which assumes a uniform
        distribution of the categories to define the chance outcome.

    Returns
    -------
    kappa : float
        Fleiss's or Randolph's kappa statistic for inter rater agreement

    Note:
    -------
    - According to wikipedia, N = # of subjects, n = # of raters, k = # of categories
    - For our case, \# of subjects = \# of shots/dialogs = N
                    \# of raters = \# of label types = n
                    \# of categories = 2 (0/1) = k
    - So we need to convert the list of labels to shape (N, n) which is in our case (N, 3)
      and then convert it to (N, k) which is (N, 2) using aggregate_raters() function.
    - The first output of aggregate_raters() is the table which is the input of this function.
    """

    table = 1.0 * np.asarray(table)   #avoid integer division
    n_sub, n_cat =  table.shape
    n_total = table.sum()
    n_rater = table.sum(1)
    n_rat = n_rater.max()
    #assume fully ranked
    assert n_total == n_sub * n_rat

    #marginal frequency  of categories
    p_cat = table.sum(0) / n_total

    table2 = table * table
    p_rat = (table2.sum(1) - n_rat) / (n_rat * (n_rat - 1.))
    p_mean = p_rat.mean()

    if method == 'fleiss':
        p_mean_exp = (p_cat*p_cat).sum()
    elif method.startswith('rand') or method.startswith('unif'):
        p_mean_exp = 1 / n_cat

    kappa = (p_mean - p_mean_exp) / (1- p_mean_exp)
    return kappa

def compute_cronbach_alpha(data: np.ndarray, threshold: float = 0.5) -> float:
    r"""
    Compute Cronbach Alpha for a list of labels.

    alpha = (k/(k-1))*(1 - (sum(var_i)/var_total))
    ------------------------------------------
    Args:
        - data: numpy array of shape (RATER, N) where N is the number of shots/dialogs.
        - threshold: float, threshold for being positive.
    Returns:
        - float: Cronbach Alpha
    """
    # Each row should represent a respondent,
    # and each column should represent a different item.
    # Each label type is a respondent, each column (shot/dialog) score is an item.

    data = (data > threshold).astype(np.int32)
    item_var = np.var(data, axis=0, ddof=1)
    vat_tot_score = np.var(data.sum(axis=1), ddof=1)

    num_items = data.shape[1]
    alpha1 = (num_items / (num_items - 1)) * (1 - (item_var.sum() / vat_tot_score))
    cov = np.cov(data.T)
    c_bar = (cov.sum() + np.trace(cov)) / (num_items * (num_items + 1))
    alpha2 = (num_items * c_bar)/(1 + (num_items - 1) * c_bar)
    return alpha1, alpha2

def compute_pairwise_f1(data: np.ndarray, threshold: float = 0.5)->float:
    r"""
    Compute pairwise F1 score for a list of labels
    and average over all pairs.
    ------------------------------------------
    Args:
        - data: numpy array of shape (RATER, N) where N is the number of shots/dialogs.
        - threshold: float, threshold for being positive.
    Returns:
        - float: pairwise F1 score
    """
    n = data.shape[0]
    f1 = []
    for i in range(n):
        for j in range(i+1, n):
            dat_i = data[i]
            dat_j = data[j]
            if np.any((dat_i > 0) & (dat_i < 1)):
                dat_i = (dat_i > threshold).astype(np.int32)
            if np.any((dat_j > 0) & (dat_j < 1)):
                dat_j = (dat_j > threshold).astype(np.int32)
            f1.append(f1_score(dat_i, dat_j))
    return np.mean(f1)

def allScores(labels: Union[np.ndarray, List[np.ndarray]], threshold: float = 0.5)->Tuple[float, float, float]:
    r"""
    Take different types of labels and compute Cronbach Alpha, pairwise F1 score
    and Fleiss Kappa score.
    ------------------------------------------
    Args:
        - labels: list of labels where each label is a numpy array (coming from a RATER)
          or numpy array of shape (RATER, N) where N is the number of shots/dialogs.
    Returns:
        - Tuple: Cronbach Alpha, pairwise F1 score, Fleiss Kappa score
    """
    if isinstance(labels, list):
        labels = np.vstack(labels)
    alpha1, alpha2 = compute_cronbach_alpha(labels, threshold)
    f1 = compute_pairwise_f1(labels, threshold)
    a, _ = aggregate_raters((labels.T > threshold).astype(np.int32))
    kappa = fleiss_kappa(a)
    return alpha1, alpha2, f1, kappa

if __name__ == "__main__":
    from IPython.display import display
    ours_data = True
    if not ours_data:
        # Test for Benchmark DataSet
        data_name = "tvsum"
        vid_names = load_yaml("benchdata/video_info/data_names.yaml")[data_name]
        if data_name == "tvsum":
            tvsum_labels = load_pickle(f"benchdata/video_info/tvsum_anno_normalized.pkl")
        df  = pd.DataFrame(columns=['vid_name', 'C_alpha1', 'C_alpha2', 'p_F1', 'kappa'])
        for v in vid_names:
            if data_name == "summe":
                labels = load_pickle(f"data/benchmark-dataset/{data_name}/features/IMAGE/{v}.pkl")['user_summary']
            else:
                labels = tvsum_labels[v]
            alpha1, alpha2, f1, kappa = allScores(labels, threshold=0.4999)
            df = pd.concat([df, pd.DataFrame([[v, alpha1, alpha2, f1, kappa]], columns=df.columns)], ignore_index=True)
        # Mean over all videos
        mean_score = df[['C_alpha1', 'C_alpha2', 'p_F1', 'kappa']].mean(axis=0)
        df = pd.concat([df, pd.DataFrame([['mean', mean_score[0], mean_score[1], mean_score[2], mean_score[3]]],
                                        columns=df.columns)], ignore_index=True)
        # Display results
        display(df)
        # os.makedirs("scores", exist_ok=True)
        # df.to_csv(f"scores/{data_name}_alpha_F1_kappa.csv")
    else:
        # Test for PlotSnap DataSet
        df = pd.DataFrame(columns=['season', 'episode', 'v_C_alpha1', 'v_C_alpha2', 'v_p_F1', 'v_kappa',
                                    'd_C_alpha1', 'd_C_alpha2', 'd_p_F1', 'd_kappa'])
        fandom_eps = load_yaml("configs/data_configs/fandom_time.yaml")
        label_types = ['MSR', 'F', 'H']
        for seas in fandom_eps['24']:
            for ep in fandom_eps['24'][seas]:
                vid_labels = [prepareLabels(lt, 'vid', '24', seas, ep) for lt in label_types]
                # If you are using pingouin, you can use the following code
                # vid_df = pd.DataFrame([*vid_labels])
                # vid_alpha = pg.cronbach_alpha(data=vid_df)[0]
                vid_alpha1, vid_alpha2, vid_f1, vid_kappa = allScores(vid_labels)
                dia_labels = [prepareLabels(lt, 'dia', '24', seas, ep) for lt in label_types]
                dia_alpha1, dia_alpha2, dia_f1, dia_kappa = allScores(dia_labels)
                df = pd.concat([df, pd.DataFrame([[seas, ep, vid_alpha1, vid_alpha2, vid_f1, vid_kappa,
                                                dia_alpha1, dia_alpha2, dia_f1, dia_kappa]],
                                                columns=df.columns)], ignore_index=True)
        # Mean over all episodes
        mean_score = df[['v_C_alpha1', 'v_C_alpha2', 'v_p_F1', 'v_kappa',
                         'd_C_alpha1', 'd_C_alpha2', 'd_p_F1', 'd_kappa']].mean(axis=0)
        df = pd.concat([df, pd.DataFrame([['mean', 'mean', mean_score[0], mean_score[1], mean_score[2], mean_score[3],
                                            mean_score[4], mean_score[5], mean_score[6], mean_score[7]]],
                                            columns=df.columns)], ignore_index=True)
        # Display results
        display(df)
        # os.makedirs("scores", exist_ok=True)
        # df.to_csv("scores/alpha_F1_kappa.csv")
