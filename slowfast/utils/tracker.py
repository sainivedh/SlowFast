import os.path
import urllib.request

import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import multivariate_normal

import torch
import torchvision

from slowfast.utils import feature_extractor

from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import nn_matching


def to_tlwh(bb):
    """Takes a box of tlbr and returns a tlwh"""
    w = bb[2] - bb[0]
    h = bb[3] - bb[1]
    return (bb[0], bb[1], w, h)


def metrics(bb1, bb2):
    """Computes metrics between two boxes"""
    i1, j1, w1, h1 = to_tlwh(bb1)
    i2, j2, w2, h2 = to_tlwh(bb2)

    top = max(i1, i2)
    bottom = min(i1 + h1, i2 + h2)
    left = max(j1, j2)
    right = min(j1 + w1, j2 + w2)

    overlap_height = bottom - top
    overlap_width = right - left

    if overlap_height <= 0 or overlap_width <= 0:
        overlap_area = 0
    else:
        overlap_area = overlap_height * overlap_width

    area1 = h1 * w1
    area2 = h2 * w2

    union_area = area1 + area2 - overlap_area
    iou = overlap_area / union_area

    # height difference
    hdiff = abs(h1 - h2)
    # width difference
    wdiff = abs(w1 - w2)

    # center coordinates for box1
    cx1, cy1 = j1 + w1 / 2, i1 + h1 / 2
    # center coordinates for box2
    cx2, cy2 = j2 + w2 / 2, i2 + h2 / 2

    # euclidean distance between the 2 box centers
    cd = distance.euclidean((cx1, cy1), (cx2, cy2))

    return iou, hdiff, wdiff, cd


def match_boxes(mid_boxes, bboxes):
    """
    Matches mid_boxes and bboxes using the euclidean distance
    Parameters
    ==========
        mid_boxes: the output of the DeepSortTracker
        bboxes: the input to the DeepSortTracker
    Returns
    =======
        ordered_index: array an index array to go from mid_boxes to bboxes
    """
    score_matrix = np.zeros((len(mid_boxes), len(bboxes)))
    for i, box in enumerate(bboxes):
        for j, mid_box in enumerate(mid_boxes):
            _, _, _, cd = metrics(mid_box, box)
            score_matrix[j, i] = cd
    col_idx, row_idx = linear_sum_assignment(score_matrix)
    ordered_index = np.zeros((len(bboxes),), dtype=int)
    for row, col in zip(row_idx, col_idx):
        ordered_index[row] = col
    return ordered_index


def download_file(url, file_path):
    urllib.request.urlretrieve(url, file_path)


def load_encoder(model_path="mars-small128.pb"):
    if not os.path.isfile(model_path):
        download_file(
            "https://github.com/theAIGuysCode/yolov4-deepsort/blob/9e745bfb3ea5e7c7505cb11a8e8654f5b1319ad9/model_data/mars-small128.pb?raw=true",
            model_path,
        )
    return feature_extractor.create_box_encoder(model_path, batch_size=1)


class DeepSortTracker:
    def __init__(self, max_age=120, max_cosine_distance=0.4):
        nn_budget = None
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget
        )
        self.tracker = Tracker(metric, max_age=max_age)
        self.encoder = load_encoder()

    def advance(self, new_bounding_boxes, new_frame):
        features = self.encoder(new_frame, new_bounding_boxes)

        detections = [
            Detection(bbox, 1.0, feature)  # TODO: extract confidence from detectron2
            for bbox, feature in zip(new_bounding_boxes, features)
        ]

        self.tracker.predict()
        self.tracker.update(detections)

        box_ids = [
            track.track_id
            for track in self.tracker.tracks
            if track.is_confirmed() and track.time_since_update < 1
        ]
        bboxes = [
            track.to_tlwh()
            for track in self.tracker.tracks
            if track.is_confirmed() and track.time_since_update < 1
        ]
        return box_ids, bboxes


class Track:
    def __init__(self, track_id, task_id, feature, bbox, max_history, max_age):
        self.features = np.array([feature])
        self.max_history = max_history
        self.max_age = max_age
        self.track_id = track_id
        self.bbox = bbox
        self.last_updated = task_id

    def is_active(self, current_task_id):
        """Whether or not the track is part of the current boxes"""
        return current_task_id == self.last_updated

    def is_outdated(self, current_task_id):
        """Whether or not the track should be removed"""
        return current_task_id - self.last_updated > self.max_age

    def update(self, new_feature, new_bbox, current_task_id):
        """Updates the track with a new detection"""
        self.bbox = new_bbox
        self.last_updated = current_task_id
        self.features = np.concatenate((self.features, [new_feature]))[-self.max_history :]


def track_builder(max_history, max_age):
    def new_track(*args, **kwargs):
        return Track(
            *args,
            **kwargs,
            max_age=max_age,
            max_history=max_history,
        )

    return new_track


class HungarianTracker:
    def __init__(self, alpha=0.5, max_history=120, max_age=50):
        self.current_task_id = 0
        self.track_builder = track_builder(max_history, max_age)
        self.tracks = []
        self.id_counter = 0
        self.alpha = alpha
        self.encoder = load_encoder()

    def new_id(self):
        new_id = self.id_counter
        self.id_counter += 1
        return new_id

    def compare_embeddings(self, feature, track):
        """
        Returns the maximum cosine similarity between feature
        and track.features
        Parameters
        ==========
            feature: 128 * float the feature embedding
            track: Track a track to compare it to
        Returns
        =======
            score: a similarity metric
        """

        def cosine_similarity(ft1, ft2):
            dot_product = ft1.dot(ft2)
            norm1 = np.linalg.norm(ft1)
            norm2 = np.linalg.norm(ft2)
            return dot_product / (norm1 * norm2)

        return max(
            [
                cosine_similarity(feature, track_feature.reshape(-1))
                for track_feature in track.features
            ]
        )

    def get_ids(self):
        return [
            track.track_id
            for track in self.tracks
            if track.is_active(self.current_task_id)
        ]

    def advance(self, new_bounding_boxes, new_frame):
        """
        Takes the bounding boxes and returns an id for each one
        If the number of boxes doesn't match the previous task, then new ids
        are assigned for non matched bounding boxes or ids are dropped
        if the number is lower.

        Parameters
        ----------
            new_bounding_boxes: [Tuple(int, 4)]
            new_frame: [int] The new image frame
        Returns
        -------
            ids: [int]
        """
        new_embeddings = self.encoder(new_frame, new_bounding_boxes)
        self.current_task_id += 1

        if self.current_task_id == 0:  # first task
            self.tracks = [
                self.track_builder(self.new_id(), self.current_task_id, feature, bbox)
                for feature, bbox in zip(new_embeddings, new_bounding_boxes)
            ]
            return self.get_ids()

        new_matrix = np.zeros(
            (len(self.tracks), len(new_bounding_boxes)), dtype=np.float32
        )
        for i, bounding_box in enumerate(new_bounding_boxes):
            for j, track in enumerate(self.tracks):
                iou, _, _, _ = metrics(bounding_box, track.bbox)
                score = self.compare_embeddings(
                    new_embeddings[
                        i,
                    ],
                    track,
                )
                new_matrix[j, i] = self.alpha * iou + (1 - self.alpha) * score
        row_idx, col_idx = linear_sum_assignment(new_matrix, maximize=True)

        ids = []
        for i, box in enumerate(new_bounding_boxes):
            if i not in col_idx:
                new_id = self.new_id()
                self.tracks.append(
                    self.track_builder(
                        new_id,
                        self.current_task_id,
                        new_embeddings[i],
                        box,
                    )
                )
                ids.append(new_id)
            else:
                col = np.where(col_idx == i)
                row = row_idx[col][0]
                self.tracks[row].update(
                    new_embeddings[i], new_bounding_boxes[i], self.current_task_id
                )
                ids.append(self.tracks[row].track_id)

        for j in range(len(self.tracks)):
            if self.tracks[i].is_outdated(self.current_task_id):
                self.tracks.pop(j)

        return ids, new_bounding_boxes
