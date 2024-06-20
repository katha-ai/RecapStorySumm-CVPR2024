#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
`general_utils.py`: This file contains utility functions for general purposes.
"""

import os
import tqdm
import time
import yaml
import glob
import json
import torch
import codecs
import pickle
import random
import datetime
import numpy as np

from datetime import datetime as dt
from omegaconf import OmegaConf, DictConfig
from multiprocessing import cpu_count, Pool
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from typing import (Callable, Dict, Tuple, List, Any,
                    Union, BinaryIO, Optional)

from utils.logger import return_logger

logger = return_logger(__name__)

def pad(x: Union[Tuple, List[np.ndarray]],
        padding_dim: int = 0,
        pad_value: float = 0.0
        )-> torch.Tensor:
    r"""
    Pad the array/tensor sequence.
    -------------------------------
    Args:
        - `x`: Tuple/List of Tensor or array of shape `(b, m, d)` or `(b, m)` of same kind.
            where `b` is the batch size, `m` is the sequence length, and `d` is the feature dimension.
        - `padding_dim`: dimension to pad. Default is `0`.
        - `pad_value`: value to pad. Default is `0.0`.
    
    Returns:
        - padded_tensor: Tensor of shape `(b, m, d)` or `(b, m)`
    """
    assert padding_dim < len(x[0].shape)
    # pad to the max length in this batch
    max_len = max([ele.shape[padding_dim] for ele in x])
    # padding sequence generator
    pad_f = lambda y: tuple([(0, y) if i==padding_dim else (0, 0) for i in range(len(x[0].shape))])
    padded = []
    for ele in x:
        pad_len = max_len - ele.shape[padding_dim]
        padded.append(np.pad(ele, pad_f(pad_len), mode='constant', constant_values=pad_value))
    return np.stack(padded, axis=0)


def save_model(model: torch.nn.Module,
               model_path: Union[Path, str],
               model_name: str,
               epoch: Optional[int],
               score: Optional[float]
              )->None:
    """
    Save the model.
    ----------------
    Args:
        - model (torch.nn.Module): The model to be saved.
        - model_path (Union[Path, str]): The path to save the model.
        - model_name (str): The name of the model.
    """
    epoch = "NA" if epoch is None else epoch
    score = 0. if score is None else score
    metric = model_name.split(".")[0].split("_")[-1]
    logger.info(f"Saving best model with {metric} = {score:.3f} at Epoch {epoch+1}")
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_path, model_name))
    logger.info(f"Saved model at {model_path}/{model_name}")

def seed_everything(seed=0, harsh=False):
    """
    Seeds all important random functions
    -------------------------------------
    Args:
        seed (int, optional): seed value. Defaults to 0.
        harsh (bool, optional): torch backend deterministic. Defaults to False.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    if harsh:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

class ParseEPS(object):
    """
    Parse the `yaml` file and return `training`, `val`, `test`, `human`
    episodes across all seasons of a particular `series` or List of
    series.
    """
    def __init__(
        self, config: Union[Dict, DictConfig], series: List[str] = ["24", "prison-break"]
    ) -> None:
        """
        Parse the `yaml` file and return `train`, `val`, `test`, and `human`
        ----------------------------------------------------------------
        Args:
            config (Dict): The `yaml` file.
        Returns:
            Dict: A dictionary with keys: `train`, `val`, `test`, and `human` and
            the corresponding episodes.
        """
        pth = Path(config["data_path"])
        train_eps_names = []
        val_eps_names = []
        test_eps_names = []
        human_eps_names = []
        if isinstance(series, str):
            series = [series]
        for ser in series:
            for season in config[ser]:
                train_eps_names += [pth/f"{ser}/{season}/{season}{ep}" for ep in config[ser][season]["train"]]
                val_eps_names += [pth /f"{ser}/{season}/{season}{ep}" for ep in config[ser][season]["val"]]
                test_eps_names += [pth /f"{ser}/{season}/{season}{ep}" for ep in config[ser][season]["test"]]
                human_eps_names += [pth /f"{ser}/{season}/{season}{ep}" for ep in config[ser][season]["human"]]
        self.dct = {"train": train_eps_names, "val": val_eps_names,
                    "test": test_eps_names, "human": human_eps_names}
        
    @staticmethod
    def convert2Yamlable(dct: Dict) -> Dict:
        dct["train"] = [ep.as_posix() for ep in dct["train"]]
        dct["val"] = [ep.as_posix() for ep in dct["val"]]
        dct["test"] = [ep.as_posix() for ep in dct["test"]]
        dct["human"] = [ep.as_posix() for ep in dct["human"]]
        return dct

    @staticmethod
    def convert2Path(dct: Dict)->Dict:
        dct["train"] = [Path(ep) for ep in dct["train"]]
        dct["val"] = [Path(ep) for ep in dct["val"]]
        dct["test"] = [Path(ep) for ep in dct["test"]]
        dct["human"] = [Path(ep) for ep in dct["human"]]
        return dct

def get_config(conf_path: Union[str, Path]) -> Dict:
    base_conf = OmegaConf.load(conf_path)
    overrides = OmegaConf.from_cli()
    updated_conf = OmegaConf.merge(base_conf, overrides)
    # OmegaConf.update(updated_conf, "model_name", fill_model_name(updated_conf))
    return OmegaConf.to_container(updated_conf)

def load_yaml(path: Union[Path, str]) -> Dict:
    with open(path, "r") as f:
        conf = yaml.safe_load(f)
    return conf

def save_yaml(path: Union[Path, str], obj: Dict) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, indent=4, default_flow_style=False,
                       encoding='utf-8', allow_unicode=True)

def load_json(path: Union[Path, str]) -> Dict:
    with open(path, "r") as f:
        conf = json.load(f)
    return conf

def save_json(path: Union[Path, str], obj: Dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)

def load_txt_file(path: Path) -> List[str]:
    with open(path, "r") as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def load_pickle(path: Union[Path, str])->Any:
    """Load pickle file. `path` must contain .pkl file name"""
    if isinstance(path, Path):
        path = path.as_posix()
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj

def save_pickle(path: Union[Path, str], obj: Any)->None:
    """Save pickle file. `path` must contain .pkl file name"""
    if isinstance(path, Path):
        path = path.as_posix()
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def get_file_name(file_path: Union[Path, str]) -> Path:
    """
    Returns the required file in folder
    according to `file_path` string.
    """
    f_names = []
    if isinstance(file_path, Path):
        file_path = file_path.as_posix()
    for f in glob.glob(file_path):
        f_names.append(f)
    if len(f_names):
        return Path(f_names[0])
    return Path("")


def frame2time(frame_idx: int, fps: float):
    """Convert frame index to time"""
    return round(frame_idx/fps, 3)

def time2frame(frame_time: float, fps: float):
    """convert frame_time to frame index"""
    return round(frame_time*fps)

def time2secs(time_obj: datetime.time) -> float:
    """
    Convert time object to seconds.
    """
    return (time_obj.hour*3600 + time_obj.minute * 60 + time_obj.second) + time_obj.microsecond/1000000

def sec2datetime(ti: float) -> datetime.time:
    str_time = time.strftime("%H:%M:%S", time.gmtime(ti)) + ("%.3f" % (ti - int(ti)))[1:] + "000"
    return dt.strptime(str_time, '%H:%M:%S.%f').time()
    
def euclidean_loss(A: np.ndarray, B:np.ndarray) -> np.ndarray:
    """
    Given Two 2D-Matrix of size (m x n) and (p x n), trying to estimate point-wise
    euclidean distance (between two vectors).

    Args:
        - A: numpy array of dimension of (m x n)
        - B: numpy array of dimension of (p x n)

    Return:
        - A matrix of size (m x p), where `(i,j)` entry = 
        euclidean_distance(i-th row vec of A, j-th row vec of B)
    """
    arr = np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    for i, row in enumerate(A):
        arr[i] = np.linalg.norm(row - B, axis=1)
    return arr


def readVidEvents(path: Union[Path, str],
                  split_val:int=2,
                  remove_header:bool=True
                 )->List[list]:
    """
    Read VideoEvents (i.e., shot starting frame idx and time-stamp) from file .videvents.
    """
    videvents = []
    if isinstance(path, Path):
        path = path.as_posix()
    with codecs.open(path, "r", "utf-8") as f:
        if remove_header:
            _ = f.readline()  # removing header
        while True:
            line = f.readline()
            if not line:
                break
            frame = line.split()[:split_val]
            if split_val == 2:
                videvents.append([int(frame[0]), float(frame[1])])#frame_no, frame_time
            else:
                videvents.append([int(k) for k in frame[:-1]]+[float(frame[-1])])
    return videvents

def getShotObj(shot_arr:List, shot_no:Any)->Any:
    """
    Given an array of shot objects, try to find the given
    shot object with the given `shot_no` .
    -----------------------------------------------------
    Note:
    The given array of shots (`shot_arr`) is already sorted
    in ascending manner in terms of shot no.
    """
    if shot_no < shot_arr[0].shot_no or\
        shot_no > shot_arr[-1].shot_no:
        return None

    low = 0
    high = len(shot_arr) - 1
    mid = 0
    while low <= high:
        mid = (high + low) // 2
        if shot_arr[mid].shot_no < shot_no:
            low = mid + 1
        elif shot_arr[mid].shot_no > shot_no:
            high = mid - 1
        else:
            return shot_arr[mid]
    return None

def getLastLine(filename: Union[Path, str]):
    if isinstance(filename, Path):
        filename = filename.as_posix()
    with open(filename, 'rb') as f:
        try:  # catch OSError in case of a one line file
            # Go to the end of the file before the last break-line
            f.seek(-2, os.SEEK_END)
            # Keep reading backward until you find the next break-line
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        return f.readline().decode()

def cosine_similarity_chunk(t: Tuple) -> np.ndarray:
    return cosine_similarity(t[0][t[1][0]: t[1][1]], t[0]).astype(np.float16)


def get_cosine_similarity(
    X: np.ndarray, Y: np.ndarray, verbose: bool = True, chunk_size: int = 1000, threshold: int = 10000
) -> np.ndarray:
    n_rows = X.shape[0]

    if n_rows <= threshold:
        return cosine_similarity(X, Y)

    else:
        logger.info(
            'Large feature matrix thus calculating cosine similarities in chunks...'
        )
        start_idxs = list(range(0, n_rows, chunk_size))
        end_idxs = start_idxs[1:] + [n_rows]
        cos_sim = [cosine_similarity(X[s:e], Y) for s, e in zip(start_idxs, end_idxs)]

        return np.vstack(cos_sim)

def get_cosine_similarity1(
    X: np.ndarray, verbose: bool = True, chunk_size: int = 1000, threshold: int = 10000
) -> np.ndarray:
    n_rows = X.shape[0]

    if n_rows <= threshold:
        return cosine_similarity(X)

    else:
        logger.info(
            'Large feature matrix thus calculating cosine similarities in chunks...'
        )
        start_idxs = list(range(0, n_rows, chunk_size))
        end_idxs = start_idxs[1:] + [n_rows]
        cos_sim = parallelise(
            cosine_similarity_chunk,
            [(X, idxs) for i, idxs in enumerate(zip(start_idxs, end_idxs))],
            verbose,
        )

        return np.vstack(cos_sim)

def get_files_to_remove(duplicates: Dict[str, List]) -> List:
    """
    Get a list of files to remove.

    Args:
        duplicates: A dictionary with file name as key and a list of duplicate file names as value.

    Returns:
        A list of files that should be removed.
    """
    # iterate over dict_ret keys, get value for the key and delete the dict keys that are in the value list
    files_to_remove = set()

    for k, v in duplicates.items():
        tmp = [
            i[0] if isinstance(i, tuple) else i for i in v
        ]  # handle tuples (image_id, score)

        if k not in files_to_remove:
            files_to_remove.update(tmp)

    return list(files_to_remove)


def save_json(results: Dict, filename: str, float_scores: bool = False) -> None:
    """
    Save results with a filename.

    Args:
        results: Dictionary of results to be saved.
        filename: Name of the file to be saved.
        float_scores: boolean to indicate if scores are floats.
    """
    logger.info('Start: Saving duplicates as json!')

    if float_scores:
        for _file, dup_list in results.items():
            if dup_list:
                typecasted_dup_list = []
                for dup in dup_list:
                    typecasted_dup_list.append((dup[0], float(dup[1])))

                results[_file] = typecasted_dup_list

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)

    logger.info('End: Saving duplicates as json!')


def parallelise(function: Callable, data: List, verbose: bool) -> List:
    pool = Pool(processes=cpu_count())
    results = list(
        tqdm.tqdm(pool.imap(function, data, 100), total=len(data), disable=not verbose)
    )
    pool.close()
    pool.join()
    return results

def mergeArrays(arr1:List, arr2:List)->List:
    n1, n2 = len(arr1), len(arr2)
    arr3 = [None] * (n1 + n2)
    i,j,k = 0,0,0
    while i < n1 and j < n2:
        if arr1[i] < arr2[j]:
            arr3[k] = arr1[i]
            k = k + 1
            i = i + 1
        else:
            arr3[k] = arr2[j]
            k = k + 1
            j = j + 1
    while i < n1:
        arr3[k] = arr1[i]
        k = k + 1
        i = i + 1
    while j < n2:
        arr3[k] = arr2[j]
        k = k + 1
        j = j + 1
    return arr3

# Updating references for modules in pickled objects when directory changes #
class CustomUnpickler(pickle.Unpickler):
    def __init__(self,
                 file: BinaryIO,
                 module_mapping: Dict[str, str]
                )->None:
        super().__init__(file)
        self.module_mapping = module_mapping
        
    def find_class(self,
                   module: str,
                   name: str
                  )->Any:
        if module in self.module_mapping:
            module = self.module_mapping[module]
        return super().find_class(module, name)

def update_module_references(input_path: Union[Path, str],
                             output_path: Union[Path, str],
                             module_mapping: Dict[str, str]
                            )->None:
    r"""
    Args:
        - input_path: Path to the input file (from where the object will be loaded)
        - output_path: Path to the output file (where the updated object will be saved)
        - module_mapping: Mapping of old module names to new module names

    Returns:
        - None
    ---------------------------------------------------------------------------------------------
    USAGE:
    
    >>> # Define the mapping of old module names to new module names
    >>> module_mapping = {
    >>>     'old_module_name': 'new_module_name',
    >>>     'vid_utils': 'utils.vid_utils',
    >>>     # Add more mappings if needed
    >>> }
    >>> # Update the module references in the pickled object
    >>> update_module_references('data/24/S02/S02E19/encodings/vid_encodings/episode_OBJ.pkl',
    >>>                          'data/24/S02/S02E19/encodings/vid_encodings/episode_OBJ.pkl',
    >>>                          module_mapping)
    """
    if isinstance(input_path, Path):
        input_path = input_path.as_posix()
    if isinstance(output_path, Path):
        output_path = output_path.as_posix()
    cache_path = input_path + '.cache' if input_path == output_path else output_path
    with open(input_path, 'rb') as input_file:
        with open(cache_path, 'wb') as output_file:
            custom_unpickler = CustomUnpickler(input_file, module_mapping)
            unpickled_object = custom_unpickler.load()

            # Repickle the object with updated module references
            pickle.dump(unpickled_object, output_file)

    if input_path == output_path:
        os.remove(input_path)
        # Rename the cache file to the output file and delete the cache file
        os.rename(cache_path, output_path)
