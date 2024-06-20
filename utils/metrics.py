#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
metrics.py: metric for evaluation of video summarization.
Difference in calculations for metric and loss?
https://saturncloud.io/blog/keras-understanding-the-difference-between-loss-and-metric-calculation/
"""

import numpy as np
from typing import Tuple, Optional, List
from sklearn.metrics import (average_precision_score, f1_score)

# ************************************************************************************
#                   Defining Relevant Metrics: AP metric
# ************************************************************************************

def getScores(y_true:List[np.ndarray],
	      	  y_pred:List[np.ndarray],
			  f1_threshold:Optional[float]=0.5
			 )->Tuple[float, float]:
	r"""
	Given y_true and y_pred, return (AP, F1,
	AvgAP (averaged across different tIOUs), and IOU_AP
	(for different tIOUs) scores).
	-------------------------------------------------------------------
	Args:
		- y_true: (List[np.ndarray]) The ground truth labels.
		- y_pred: (List[np.ndarray]) Soft scores or hard scores(binary labels)
		  predicted from model.
		- f1_threshold: (Optional[float]) Threshold for F1 score.

	Returns:
		- AP, F1: (Tuple)
	"""
	AP, F1 = 0, 0
	n_videos = len(y_true)
	for i in range(n_videos):
		y_t = y_true[i]; y_p = y_pred[i]
		# check if ground truth contains soft labels
		if np.any((y_t > 0) & (y_t < 1)):
			y_t = (y_t > 0.5).astype(np.int32)
		# For AP, F1 
		AP_ = average_precision_score(y_t, y_p)
		y_p = (y_p > f1_threshold).astype(np.int32)
		F1_ = f1_score(y_t, y_p)
		AP += AP_; F1 += F1_
	# Average over all videos
	AP /= n_videos; F1 /= n_videos
	return AP, F1


if __name__ == "__main__":

	# Test
	y_true = np.array([0, 1, 0, 1, 1, 1, 0, 0, 0, 1])
	y_pred = np.array([0.1, 0.9, 0.4, 0.8, 0.3, 0.5, 0.2, 0.1, 0.1, 0.7])
	AP, P, R, F1, avg_ap, ap_thresh = getScores(y_true, y_pred)
	print("AP: ", AP)
	print("P: ", P)
	print("R: ", R)
	print("F1: ", F1)
	print("avg_ap: ", avg_ap)
	print("ap_thresh: ", ap_thresh)
