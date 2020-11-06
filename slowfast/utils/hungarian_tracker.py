import os.path
import urllib.request

import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import multivariate_normal

import torch
import torchvision


def metrics(bb1, bb2):
    def to_tuple(bb):
        w = bb[2] - bb[0]
        h = bb[3] - bb[1]
        return (bb[1], bb[2], h, w)
    i1, j1, h1, w1 = to_tuple(bb1)
    i2, j2, h2, w2 = to_tuple(bb2)

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

    #height difference
    hdiff = abs(h1-h2)
    #width difference
    wdiff = abs(w1-w2)

    #center coordinates for box1
    cx1, cy1 = j1 + w1/2, i1 + h1/2
    #center coordinates for box2
    cx2, cy2 = j2 + w2/2, i2 + h2/2

    #euclidean distance between the 2 box centers
    cd = distance.euclidean((cx1, cy1), (cx2, cy2))

    return iou, hdiff, wdiff, cd


def get_gaussian_mask():
    #128 is image size
    x, y = np.mgrid[0:1.0:128j, 0:1.0:128j]
    xy = np.column_stack([x.flat, y.flat])
    mu = np.array([0.5,0.5])
    sigma = np.array([0.22,0.22])
    covariance = np.diag(sigma**2)
    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
    z = z.reshape(x.shape)

    z = z / z.max()
    z  = z.astype(np.float32)

    mask = torch.from_numpy(z)

    return mask


class FeatureEncoder():
    def __init__(self, weight_path="ckpts/model640.pt"):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = torch.load(weight_path).to(self.device)
        self.gaussian_mask = get_gaussian_mask().to(self.device)
        self.pre_process = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor(),
        ])

    def process_crops(self, crops):
        """Return embeddings for the crop"""
        input_tensors = torch.stack([self.pre_process(crop) for crop in crops]).to(self.device)
        input_batch = input_tensors * self.gaussian_mask
        with torch.no_grad():
            features = self.model.forward_once(input_batch)
        features = features.detach().cpu().numpy()
        return features


def download_file(url, file_path):
    urllib.request.urlretrieve(url, file_path)


def load_encoder(weight_path="model640.pt"):
    if not os.path.isfile(weight_path):
        download_file(
            'https://github.com/abhyantrika/nanonets_object_tracking/raw/master/ckpts/model640.pt',
            weight_path)
    return FeatureEncoder(weight_path)


class HungarianTracker():
    def __init__(self):
        self.current_task_id = 0
        self.tracking = []
        self.embeddings = [] # crop embeddings TODO: store for more than 1 prediction
        self.ids = []
        self.id_counter = 0
        self.encoder = load_encoder()

    def new_id(self):
        new_id = self.id_counter
        self.id_counter += 1
        return new_id

    def metrics(self, bb1, bb2):
        return metrics(bb1, bb2)

    def compare_embeddings(self, ft1, ft2):
        """Returns the cosine distance between ft1 and ft2"""
        dot_product = ft1.dot(ft2)
        norm1 = np.linalg.norm(ft1)
        norm2 = np.linalg.norm(ft2)
        return dot_product / (norm1 * norm2)

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
        def get_crop(bounding_box):
            x1, y1, x2, y2 = bounding_box.int()
            crop = new_frame[y1:y2, x1:x2] # extract crop
            return crop
        # Batch all new boxes
        new_embeddings = self.encoder.process_crops([get_crop(bb) for bb in new_bounding_boxes])

        if self.current_task_id == 0:  # first task
            self.tracking = new_bounding_boxes
            self.ids = list(range(len(new_bounding_boxes)))
            self.id_counter = len(new_bounding_boxes)
            self.current_task_id += 1
            self.embeddings = new_embeddings
            return self.ids

        new_matrix = np.zeros(
            (len(self.tracking), len(new_bounding_boxes)),
            dtype=np.float32)

        for i, bounding_box in enumerate(new_bounding_boxes):
            for j, tracked_bounding_box in enumerate(self.tracking):
                score = self.compare_embeddings(self.embeddings[j], new_embeddings[i,])
                new_matrix[j, i] = score

        row_idx, col_idx = linear_sum_assignment(new_matrix, maximize=False)
        self.current_task_id += 1
        new_ids = [
            self.new_id() if i not in col_idx else 0
            for i in range(len(new_bounding_boxes))]  # init new ids
        for row, col in zip(row_idx, col_idx):
            new_ids[col] = self.ids[row]  # reuse id if match
        self.tracking = new_bounding_boxes
        self.ids = new_ids

        return self.ids
