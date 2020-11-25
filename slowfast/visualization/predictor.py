#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import queue
import numpy as np
import cv2
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import slowfast.utils.checkpoint as cu
from slowfast.datasets import cv2_transform
from slowfast.models import build_model
from slowfast.utils import logging
from slowfast.visualization.utils import process_cv2_inputs, draw_predictions
from slowfast.utils.tracker import DeepSortTracker

logger = logging.get_logger(__name__)


class Predictor:
    """
    Action Predictor for action recognition.
    """

    def __init__(self, cfg, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            gpu_id (Optional[int]): GPU id.
        """
        if cfg.NUM_GPUS:
            self.gpu_id = (
                torch.cuda.current_device() if gpu_id is None else gpu_id
            )

        # Build the video model and print model statistics.
        self.model = build_model(cfg, gpu_id=gpu_id)
        self.model.eval()
        self.cfg = cfg

        if cfg.DETECTION.ENABLE:
            self.object_detector = Detectron2Predictor(cfg, gpu_id=self.gpu_id)

        logger.info("Start loading model weights.")
        cu.load_test_checkpoint(cfg, self.model)
        logger.info("Finish loading model weights")

    def predict_frame(self, frame):
        """
        Uses the underlying object detector to return bounding boxes
        """
        if not self.cfg.DETECTION.ENABLE:
            raise Exception("object detector is not enabled")
        return self.object_detector.predict_frame(frame)

    def __call__(self, task):
        """
        Returns the prediction results for the current task.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames, boxes)
        Returns:
            task (TaskInfo object): the same task info object but filled with
                prediction values (a tensor) and the corresponding boxes for
                action detection task.
        """
        if self.cfg.DETECTION.ENABLE:
            task = self.object_detector(task)

        frames, bboxes, nbboxes = task.frames, task.bboxes, task.nbboxes
        if bboxes is not None:
            bboxes = cv2_transform.scale_boxes(
                self.cfg.DATA.TEST_CROP_SIZE,
                bboxes,
                task.img_height,
                task.img_width,
            )
        if nbboxes is not None:
            for i, frame_boxes in enumerate(nbboxes):
                nbboxes[i] = cv2_transform.scale_boxes(
                    self.cfg.DATA.TEST_CROP_SIZE,
                    frame_boxes,
                    task.img_height,
                    task.img_width,
                )
        if self.cfg.DEMO.INPUT_FORMAT == "BGR":
            frames = [
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames
            ]

        frames = [
            cv2_transform.scale(self.cfg.DATA.TEST_CROP_SIZE, frame)
            for frame in frames
        ]
        inputs = process_cv2_inputs(frames, self.cfg)
        if bboxes is not None:
            index_pad = torch.full(
                size=(bboxes.shape[0], 1),
                fill_value=float(0),
                device=bboxes.device,
            )

            # Pad frame index for each box.
            bboxes = torch.cat([index_pad, bboxes], axis=1)
        if self.cfg.NUM_GPUS > 0:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(
                        device=torch.device(self.gpu_id), non_blocking=True
                    )
            else:
                inputs = inputs.cuda(
                    device=torch.device(self.gpu_id), non_blocking=True
                )
        if self.cfg.DETECTION.ENABLE and not bboxes.shape[0]:
            preds = torch.tensor([])
        else:
            preds = self.model(inputs, bboxes)

        if self.cfg.NUM_GPUS:
            preds = preds.cpu()
            if bboxes is not None:
                bboxes = bboxes.detach().cpu()

        preds = preds.detach()
        task.add_action_preds(preds)
        if bboxes is not None:
            task.add_bboxes(bboxes[:, 1:])

        return task


class ActionPredictor:
    """
    Synchronous Action Prediction and Visualization pipeline with AsyncVis or SyncVis.
    """

    def __init__(self, cfg, async_vis=None, vis=None, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            async_vis (AsyncVis object): asynchronous visualizer.
            gpu_id (Optional[int]): GPU id.
        """
        self.predictor = Predictor(cfg=cfg, gpu_id=gpu_id)
        self.is_async = async_vis is not None
        if async_vis is not None:
            self.async_vis = async_vis
        else:
            self.vis = vis
            self.vis_queue = queue.Queue()

    def put(self, task):
        """
        Make prediction and put the results in `async_vis` task queue.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames, boxes)
        """
        task = self.predictor(task)
        if not self.is_async:
            frames = draw_predictions(task, self.vis)
            task.frames = np.array(frames)
            self.vis_queue.put(task)
            return
        self.async_vis.get_indices_ls.append(task.id)
        self.async_vis.put(task)

    def get(self):
        """
        Get the visualized clips if any.
        """
        if not self.is_async:
            try:
                task = self.vis_queue.get()
            except (queue.Empty, IndexError):
                raise IndexError("No results to get")
            return task
        try:
            task = self.async_vis.get()
        except (queue.Empty, IndexError):
            raise IndexError("Results are not available yet.")

        return task


class Detectron2Predictor:
    """
    Wrapper around Detectron2 to return the required predicted bounding boxes
    as a ndarray.
    """

    def __init__(self, cfg, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            gpu_id (Optional[int]): GPU id.
        """

        self.cfg = get_cfg()
        self.cfg.merge_from_file(
            model_zoo.get_config_file(cfg.DEMO.DETECTRON2_CFG)
        )
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.DEMO.DETECTRON2_THRESH
        self.cfg.MODEL.WEIGHTS = cfg.DEMO.DETECTRON2_WEIGHTS
        self.cfg.INPUT.FORMAT = cfg.DEMO.INPUT_FORMAT
        if cfg.NUM_GPUS and gpu_id is None:
            gpu_id = torch.cuda.current_device()
        self.cfg.MODEL.DEVICE = (
            "cuda:{}".format(gpu_id) if cfg.NUM_GPUS > 0 else "cpu"
        )

        logger.info("Initialized Detectron2 Object Detection Model.")

        self.predictor = DefaultPredictor(self.cfg)
        self.tracker = DeepSortTracker()

    def predict_frame(self, frame):
        """
        Returns bounding boxes for human on a single frame
        Args:
            frame (ndarray): the frame
        Returns:
            bounding_boxes (ndarray): the bounding boxes for the subjects
        """
        outputs = self.predictor(frame)
        mask = outputs["instances"].pred_classes == 0
        pred_boxes = outputs["instances"].pred_boxes.tensor[mask]
        return pred_boxes

    def __call__(self, task):
        """
        Return bounding boxes predictions as a tensor.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames)
        Returns:
            task (TaskInfo object): the same task info object but filled with
                prediction values (a tensor) and the corresponding boxes for
                action detection task.
        """
        # TODO: batch frames
        bboxes = [self.predict_frame(frame) for frame in task.frames]
        ids_boxes = [
            self.tracker.advance(bbox, frame)
            for bbox, frame in zip(bboxes, task.frames)
        ]
        bboxes = [bbox for _, bbox in ids_boxes]
        box_ids = [box_ids for box_ids, _ in ids_boxes]

        pred_boxes = bboxes[len(task.frames) // 2]

        task.add_bboxes(pred_boxes)
        task.add_nbboxes(bboxes)
        task.add_box_ids(box_ids)

        return task
