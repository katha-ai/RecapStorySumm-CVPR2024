#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
`extract_IMAGENET.py` is a module to extract features from video frames using DenseNet169 model.
"""

import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from pathlib import Path, PurePath
from utils.logger import return_logger
from moviepy.editor import VideoFileClip
from utils.image_utils import load_image
from typing import Dict, List, Optional, Union, Callable, Tuple
from utils.general_utils import save_json, frame2time, get_cosine_similarity, get_cosine_similarity1

class DenseNetHead(nn.Module):
    def __init__(self, densnet_model: Callable) -> None:
        super(DenseNetHead, self).__init__()
        self.model = densnet_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out


class IMAGE(object):
    """
    Find duplicates using CNN and/or generate CNN encodings given a single image or a directory of images.

    The module can be used for 2 purposes: Encoding generation and duplicate detection.
    - Encodings generation:
    To propagate an image through a Convolutional Neural Network architecture and generate encodings. The generated
    encodings can be used at a later time for deduplication. Using the method 'encode_image', the CNN encodings for a
    single image can be obtained while the 'encode_images' method can be used to get encodings for all images in a
    directory.

    - Duplicate detection:
    Find duplicates either using the encoding mapping generated previously using 'encode_images' or using a Path to the
    directory that contains the images that need to be deduplicated. 'find_duplciates' and 'find_duplicates_to_remove'
    methods are provided to accomplish these tasks.
    """

    def __init__(self,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = True,
                 model_name: str = 'densenet',
                 model_init: bool = True,
                 skip: int = 11
                 ) -> None:
        """
        Initialize a keras ImageNet model that is sliced at the last convolutional layer.
        Set the batch size for keras generators to be 64 samples. Set the input image size to (224, 224) for providing
        as input to ImageNet model.

        Args:
            - device: torch.device object to load the model on.
            - verbose: Display progress bar if True else disable it. Default value is `True`.
            - model_name: Name of the model to use. Available `densenet` and `resnet`.
              Default value is `densenet`.
            - model_init: Do u want to initialize an instance of model?
            - skip: No. of frames to skip, to sample (uniform sampling) frames
        """
        self.target_size = 224  # as mentioned in paper - Table1
        self.batch_size = 64
        self.logger = return_logger(__name__)
        self.skip = skip
        self.model_name = model_name
        if model_init:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            self.device = device
            self._build_model()

        self.verbose = 1 if verbose else 0

    def _build_model(self):
        """
        Build ImageNet model sliced at the last convolutional layer with global average pooling added.
        """
        if self.model_name == 'densenet':
            model = torch.hub.load(
                'pytorch/vision:v0.10.0', 'densenet169', pretrained=True)
            model.eval()
            self.model = DenseNetHead(model)
        elif self.model_name == 'resnet':
            self.model = torch.hub.load(
                'pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
            self.model.eval()
            self.model.fc = nn.Identity()
        self.logger.info(f'Loaded: {self.model_name} model')
        self.model.to(self.device)
        self.logger.info(
            'Initialized: ImageNet pretrained on ImageNet dataset sliced at last conv layer and added '
            'GlobalAveragePooling'
        )

    @torch.no_grad()
    def _get_cnn_features_single(self, image_array: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Generate CNN encodings for a single image.

        Args:
            image_array: Image typecast to numpy array.

        Returns:
            Encodings for the image in the form of numpy array.
        """
        if isinstance(image_array, np.ndarray):
            image_pil = Image.fromarray(image_array)
        image_pp = self.transform(image_pil)
        image_pp = image_pp.unsqueeze(0).to(self.device)
        tensor = self.model(image_pp)
        return tensor.detach().cpu().numpy()

    def _get_cnn_features_batch(self, image_dir: PurePath, mat_format: bool = False) -> Dict[str, np.ndarray]:
        """
        Generate CNN encodings for all images in a given directory of images.
        Args:
            image_dir: Path to the image directory.

        Returns:
            A dictionary that contains a mapping of filenames and corresponding numpy array of CNN encodings.
        """
        self.logger.info('Start: Image encoding generation')
        features = list()
        filenames = sorted(os.listdir(image_dir))
        for img_file in filenames:
            img = Image.open(image_dir/img_file)
            if img.mode != 'RGB':
                # convert to RGBA first to avoid warning
                # we ignore alpha channel if available
                img = img.convert('RGBA').convert('RGB')
            features.append(self._get_cnn_features_single(img))
        self.logger.info('End: Image encoding generation')
        if mat_format:
            return np.vstack(features)
        self.encoding_map = {j: features[i] for i, j in enumerate(filenames)}
        return self.encoding_map

    def encode_image(
        self,
        image_file: Optional[Union[PurePath, str]] = None,
        image_array: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate CNN encoding for a single image.

        Args:
            image_file: Path to the image file.
            image_array: Optional, used instead of image_file. Image typecast to numpy array.

        Returns:
            encoding: Encodings for the image in the form of numpy array.

        Example:
        ```
        from imagededup.methods import CNN
        myencoder = CNN()
        encoding = myencoder.encode_image(image_file='path/to/image.jpg')
        OR
        encoding = myencoder.encode_image(image_array=<numpy array of image>)
        ```
        """
        if isinstance(image_file, str):
            image_file = Path(image_file)

        if isinstance(image_file, PurePath):
            if not image_file.is_file():
                raise ValueError(
                    'Please provide either image file path or image array!'
                )
            image_pp = load_image(
                image_file=image_file, target_size=None, grayscale=False
            )
        elif isinstance(image_array, np.ndarray):
            image_pp = image_array
        else:
            raise ValueError(
                'Please provide either image file path or image array!')

        return self._get_cnn_features_single(image_pp)

    def blackwhite_test(self,
                        image: np.ndarray,
                        black_thresh: float,
                        white_thresh: float
                        ) -> bool:
        """
        Checks white/black tone in image. If excess (`> thresh`)
        then `return True else False`.
        --------------------------------------------------------
        Args:
            - image: 3-D Numpy array of image (expects `dtype=uint8`).
            - black_thresh: Decision value for blackiness.
            - white_thresh: Decision value for whiteness.
        """
        avg_VAL = np.average(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        std_VAL = np.std(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        return True if (avg_VAL < black_thresh and std_VAL < black_thresh+4) or\
            (avg_VAL > white_thresh and std_VAL < 255-white_thresh-45) else False

    def get_DENSE_feats(self,
                        start: int,
                        end: int,
                        vid: Union[str, VideoFileClip],
                        shot: int = -1,
                        black_thresh: float = 8.0,
                        white_thresh: float = 180.0
                        ) -> Union[Tuple, dict]:
        """
        Extract features from video frame with `DenseNet169` for a given video clip.
        Also it returns 'selected' frames to be written in local storage.
        -----------------------------------------------------------------------------
        Args:
            - start: Starting Frame Index.
            - end: Ending Frame Index.
            - vid: The path to video from which we want to extract frames
            - shot: The shot number.
            - black_thresh: Threshold to reject uniformly colored blackish frames.
            - white_thresh: Threshold to reject uniformly colored whitish frames.

        Return:
            - Dictionary of Encoding of Frames with key as frame's string
        """

        if type(vid) == str:
            vid = VideoFileClip(vid)
        encode_arr = {}
        img_lst, img_fn = [], []
        bw_count = 0
        if start == end:
            image = vid.get_frame(frame2time(start, vid.fps))
            key = "shot_{0:04n}_fn_{1:06n}".format(shot, start) if\
                shot is not None else "shot_NONE_fn_{0:06n}".format(start)
            if self.blackwhite_test(image, black_thresh=black_thresh,
                                    white_thresh=white_thresh):
                key += "_bw"
                bw_count += 1
                img_fn.append(-1*start)
            else:
                img_fn.append(start)
            self.logger.info(f"Encoding Frame: {start} of shot {shot}.")
            encode_arr[key] = self.encode_image(
                image_array=image).squeeze(axis=0)
            img_lst.append(image)
        else:
            # POSSIBLE UPGRADATION: Use `subclip` to extract frames.
            # set_fps() can be used and further checks happening below can be removed.
            # iter_frames() can be used to extract frames.
            clip = vid.subclip(frame2time(start, vid.fps),
                               frame2time(end, vid.fps))
            old_k = -(self.skip + 1)
            for k, image in enumerate(clip.iter_frames()):
                if k - old_k - 1 == self.skip:
                    self.logger.info(
                        f"Encoding Frame: {start + k} of shot {shot}.")
                    key = "shot_{0:04n}_fn_{1:06n}".format(shot, start + k) if\
                        shot is not None else "shot_NONE_fn_{0:06n}".format(start + k)
                    if self.blackwhite_test(image, black_thresh=black_thresh, white_thresh=white_thresh):
                        key += "_bw"
                        bw_count += 1
                        img_fn.append(-(start + k))
                    else:
                        img_fn.append(start + k)
                    encode_arr[key] = self.encode_image(
                        image_array=image).squeeze(axis=0)
                    img_lst.append(image)
                    old_k = k
        self.logger.info(
            f"Black/White Frames: {bw_count} of {len(encode_arr)}")
        return encode_arr, img_lst, img_fn

    def encode_images(self, image_dir: Union[PurePath, str], mat_format: bool = False) -> Dict:
        """Generate CNN encodings for all images in a given directory of images.

        Args:
            image_dir: Path to the image directory.
        Returns:
            dictionary: Contains a mapping of filenames and corresponding numpy array of CNN encodings.
        Example:
            ```
            from imagededup.methods import CNN
            myencoder = CNN()
            encoding_map = myencoder.encode_images(image_dir='path/to/image/directory')
            ```
        """
        if isinstance(image_dir, str):
            image_dir = Path(image_dir)

        if not image_dir.is_dir():
            raise ValueError('Please provide a valid directory path!')

        return self._get_cnn_features_batch(image_dir, mat_format=mat_format)

    @staticmethod
    def _check_threshold_bounds(thresh: float) -> None:
        """
        Check if provided threshold is valid. Raises TypeError if wrong threshold variable type is passed or a
        ValueError if an out of range value is supplied.

        Args:
            thresh: Threshold value (must be float between -1.0 and 1.0)

        Raises:
            TypeError: If wrong variable type is provided.
            ValueError: If wrong value is provided.
        """
        if not isinstance(thresh, float):
            raise TypeError('Threshold must be a float between -1.0 and 1.0')
        if thresh < -1.0 or thresh > 1.0:
            raise ValueError('Threshold must be a float between -1.0 and 1.0')

    def _find_duplicates_dict(
        self,
        encoding_map: Dict[str, list],
        min_similarity_threshold: float,
        scores: bool,
        outfile: Optional[str] = None,
    ) -> Dict:
        """
        Take in dictionary {filename: encoded image}, detects duplicates above the given cosine similarity threshold
        and returns a dictionary containing key as filename and value as a list of duplicate filenames. Optionally,
        the cosine distances could be returned instead of just duplicate filenames for each query file.

        Args:
            encoding_map: Dictionary with keys as file names and values as encoded images.
            min_similarity_threshold: Cosine similarity above which retrieved duplicates are valid.
            scores: Boolean indicating whether similarity scores are to be returned along with retrieved duplicates.

        Returns:
            if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
            score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}
            if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
            'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}
        """

        # get all image ids
        # we rely on dictionaries preserving insertion order in Python >=3.6
        image_ids = np.array([*encoding_map.keys()])

        # put image encodings into feature matrix
        features = np.array([*encoding_map.values()])

        self.logger.info('Start: Calculating cosine similarities...')

        self.cosine_scores = get_cosine_similarity1(features, self.verbose)

        np.fill_diagonal(
            self.cosine_scores, 2.0
        )  # allows to filter diagonal in results, 2 is a placeholder value

        self.logger.info('End: Calculating cosine similarities.')

        self.results = {}
        for i, j in enumerate(self.cosine_scores):
            duplicates_bool = (j >= min_similarity_threshold) & (j < 2)

            if scores:
                tmp = np.array([*zip(image_ids, j)], dtype=object)
                duplicates = list(map(tuple, tmp[duplicates_bool]))

            else:
                duplicates = list(image_ids[duplicates_bool])

            self.results[image_ids[i]] = duplicates

        if outfile and scores:
            save_json(results=self.results,
                      filename=outfile, float_scores=True)
        elif outfile:
            save_json(results=self.results, filename=outfile)
        return self.results

    def _find_duplicates_dict_mod(
        self,
        vid1_encoding_map: Dict[str, list],
        vid2_encoding_map: Dict[str, list],
        min_similarity_threshold: float,
        scores: bool,
        outfile: Optional[str] = None,
    ) -> Dict:
        """
        Take in dictionary {filename: encoded image}, detects duplicates above the given cosine similarity threshold
        and returns a dictionary containing key as filename and value as a list of duplicate filenames. Optionally,
        the cosine distances could be returned instead of just duplicate filenames for each query file. For black
        and white frames of a video, return empty List


        Args:
            encoding_map: Dictionary with keys as file names and values as encoded images.
            min_similarity_threshold: Cosine similarity above which retrieved duplicates are valid.
            scores: Boolean indicating whether similarity scores are to be returned along with retrieved duplicates.

        Returns:
            if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
            score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}
            if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
            'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}
        """

        # get all image ids
        # we rely on dictionaries preserving insertion order in Python >=3.6
        vid1_bw_keys = [key for key in vid1_encoding_map.keys()
                        if key.endswith('_bw')]
        vid1_encoding_map = {
            k: v for k, v in vid1_encoding_map.items() if not k.endswith('bw')}
        vid2_encoding_map = {
            k: v for k, v in vid2_encoding_map.items() if not k.endswith('bw')}
        vid1_image_ids = np.array([*vid1_encoding_map.keys()])
        vid2_image_ids = np.array([*vid2_encoding_map.keys()])

        # put image encodings into feature matrix
        vid1_features = np.array([*vid1_encoding_map.values()])
        vid2_features = np.array([*vid2_encoding_map.values()])

        self.logger.info('Start: Calculating cosine similarities...')

        self.cosine_scores = get_cosine_similarity(
            vid1_features, vid2_features, self.verbose)

        self.logger.info('End: Calculating cosine similarities.')

        self.results = {}
        for i, j in enumerate(self.cosine_scores):
            duplicates_bool = (j >= min_similarity_threshold) & (j < 2)

            if scores:
                tmp = np.array([*zip(vid2_image_ids, j)], dtype=object)
                duplicates = list(map(tuple, tmp[duplicates_bool]))

            else:
                duplicates = list(vid2_image_ids[duplicates_bool])

            self.results[vid1_image_ids[i]] = duplicates

        for key in vid1_bw_keys:
            self.results[key] = []

        if outfile and scores:
            save_json(results=self.results,
                      filename=outfile, float_scores=True)
        elif outfile:
            save_json(results=self.results, filename=outfile)
        return self.results

    def _find_duplicates_dir(
        self,
        image_dir: Union[PurePath, str],
        min_similarity_threshold: float,
        scores: bool,
        outfile: Optional[str] = None,
    ) -> Dict:
        """
        Take in path of the directory in which duplicates are to be detected above the given threshold.
        Returns dictionary containing key as filename and value as a list of duplicate file names.  Optionally,
        the cosine distances could be returned instead of just duplicate filenames for each query file.

        Args:
            image_dir: Path to the directory containing all the images.
            min_similarity_threshold: Optional, hamming distance above which retrieved duplicates are valid. Default 0.9
            scores: Optional, boolean indicating whether Hamming distances are to be returned along with retrieved
                    duplicates.
            outfile: Optional, name of the file the results should be written to.

        Returns:
            if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
            score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}
            if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
            'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}
        """
        self.encode_images(image_dir=image_dir)

        return self._find_duplicates_dict(
            encoding_map=self.encoding_map,
            min_similarity_threshold=min_similarity_threshold,
            scores=scores,
            outfile=outfile,
        )

    def find_duplicates(
        self,
        image_dir: Union[PurePath, str] = None,
        vid1_encoding_map: Dict[str, list] = None,
        vid2_encoding_map: Dict[str, list] = None,
        min_similarity_threshold: float = 0.9,
        scores: bool = False,
        outfile: Optional[str] = None,
    ) -> Dict:
        """
        Find duplicates for each file. Take in path of the directory or encoding dictionary in which duplicates are to
        be detected above the given threshold. Return dictionary containing key as filename and value as a list of
        duplicate file names. Optionally, the cosine distances could be returned instead of just duplicate filenames for
        each query file.

        Args:
            image_dir: Path to the directory containing all the images or dictionary with keys as file names
            and values as numpy arrays which represent the CNN encoding for the key image file.
            encoding_map: Optional, used instead of image_dir, a dictionary containing mapping of filenames and
                          corresponding CNN encodings.
            min_similarity_threshold: Optional, threshold value (must be float between -1.0 and 1.0). Default is 0.9
            scores: Optional, boolean indicating whether similarity scores are to be returned along with retrieved
                    duplicates.
            outfile: Optional, name of the file to save the results, must be a json. Default is None.

        Returns:
            dictionary: if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
                        score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}. if scores is False, then a
                        dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg', 'image1_duplicate2.jpg'],
                        'image2.jpg':['image1_duplicate1.jpg',..], ..}

        Example:
        ```
        from imagededup.methods import CNN
        myencoder = CNN()
        duplicates = myencoder.find_duplicates(image_dir='path/to/directory', min_similarity_threshold=0.85, scores=True,
        outfile='results.json')

        OR

        from imagededup.methods import CNN
        myencoder = CNN()
        duplicates = myencoder.find_duplicates(encoding_map=<mapping filename to cnn encodings>,
        min_similarity_threshold=0.85, scores=True, outfile='results.json')
        ```
        """
        self._check_threshold_bounds(min_similarity_threshold)

        if image_dir:
            result = self._find_duplicates_dir(
                image_dir=image_dir,
                min_similarity_threshold=min_similarity_threshold,
                scores=scores,
                outfile=outfile,
            )
        elif vid1_encoding_map:
            result = self._find_duplicates_dict_mod(
                vid1_encoding_map=vid1_encoding_map,
                vid2_encoding_map=vid2_encoding_map,
                min_similarity_threshold=min_similarity_threshold,
                scores=scores,
                outfile=outfile,
            )

        else:
            raise ValueError('Provide either an image directory or encodings!')

        return result
