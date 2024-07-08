"""
File: main.py
Project: obs-stream-overlay
Created Date: 2024-06-26
Author: martinnr1
-----
Last Modified: Sun Jul 07 2024
Modified By: martinnr1
-----
Copyright (c) 2024
"""

import atexit
import copyreg
import importlib
import json
import logging
import math
import os
import time
from multiprocessing import Lock, Process, cpu_count
from multiprocessing.managers import BaseManager
from os.path import dirname, isfile, join, realpath
from typing import Sequence

import click
import cv2 as cv
import imutils
import numpy as np
from cv2 import Feature2D, KeyPoint
from cv2.typing import MatLike, Point2f
from mss import mss

import detection.detector as detector
import obs.obssocket as obssocket


import platform


def system():
    try:
        return platform.system()
    except:
        return "Windows"


def is_windows() -> bool:
    return system() == "Windows"


def is_linux() -> bool:
    return system() != "Windows"


if is_linux():
    import line_profiler


### Logging setup
class CustomFormatter(logging.Formatter):

    color = {}
    color[logging.CRITICAL] = "1;31;1"
    color[logging.ERROR] = "1;31"
    color[logging.WARNING] = "0;33"
    color[logging.INFO] = "0;37"
    color[logging.DEBUG] = "1;34"

    def format(self, record):
        log_fmt = (
            "%(asctime)s - %(filename)12s:%(lineno)-4d - %(levelname)-8s -\33["
            + self.color[record.levelno]
            + "m %(message)s \33[m"
        )
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(CustomFormatter())

logging.basicConfig(level=logging.INFO, handlers=[handler])
log = logging.getLogger(__name__)
###


MASK_DIR = join(os.getcwd(), "mask")

DEBUGGING = False
OBJECT_TRACING = True
GPU_CUDA = False
FRAME_PREPROCESSING = False

old_frame: MatLike = None
obs: obssocket.OBSConnection = None


### multiprocessing
def _pickle_keypoints(point):
    return cv.KeyPoint, (
        *point.pt,
        point.size,
        point.angle,
        point.response,
        point.octave,
        point.class_id,
    )


copyreg.pickle(cv.KeyPoint().__class__, _pickle_keypoints)


class KeypointsDescriptors:
    keypoints: Sequence[KeyPoint] = []
    descriptors: MatLike = None

    def addKeypoints(self, keypoints: Sequence[KeyPoint]):
        # log.info(len(keypoints))
        self.keypoints = np.append(self.keypoints, keypoints)

    def addDescriptors(self, descriptors: MatLike):
        if self.descriptors is None:
            self.descriptors = descriptors
            return
        if descriptors is None or descriptors.size == 0:
            return
        self.descriptors = np.concatenate((self.descriptors, descriptors), axis=0)

    def getKeypoints(self):
        return self.keypoints

    def getDescriptors(self):
        return self.descriptors


# custom manager to support custom classes
class CustomManager(BaseManager):
    # nothing
    pass


CustomManager.register("KeypointsDescriptors", KeypointsDescriptors)


def detect_compute(
    detector: Feature2D, img: MatLike, out: KeypointsDescriptors, lck
) -> None:

    ### The compute intense part
    kps, des = detector.detectAndCompute(img, None)

    with lck:
        out.addKeypoints(kps)
        # log.info(f"Thread: {current_process().pid}: {len(kps)}")
        out.addDescriptors(des)


# split frame into subframes to distribute over processors for speedup the image detection (not detecting everything; not usable yet)
def split_and_detect_compute(
    img_detector: detector.ImageDetector, frame: MatLike
) -> tuple[Sequence[KeyPoint], MatLike]:
    with CustomManager() as manager:

        shared_obj = manager.KeypointsDescriptors()

        threads: list[Process] = []

        lck = Lock()
        frame_parts = []
        SIZE = int(min(math.floor(math.sqrt(cpu_count())), 4))
        SPLIT_W_CNT = SIZE
        SPLIT_H_CNT = SIZE
        h, w = frame.shape
        for i in range(SPLIT_W_CNT):
            for j in range(SPLIT_H_CNT):
                sub_frame = frame[
                    i * int(h / SPLIT_W_CNT) : (i + 1) * int(h / SPLIT_W_CNT),
                    j * int(w / SPLIT_H_CNT) : (j + 1) * int(w / SPLIT_H_CNT),
                ]
                frame_parts.append(sub_frame)

        for sub_frame in frame_parts:
            # cv.imshow("sub_frame", sub_frame)
            # cv.waitKey()
            p = Process(
                target=detect_compute,
                args=(img_detector.detector, sub_frame, shared_obj, lck),
            )
            threads.append(p)
            p.start()

        for p in threads:
            p.join()

        keypoints_scene = shared_obj.getKeypoints()
        descriptors_scene = shared_obj.getDescriptors()

    return keypoints_scene, descriptors_scene


###


# @line_profiler.profile
def frame_action(
    img_detector: detector.ImageDetector,
    frame: cv.typing.MatLike,
    frame_rgb: cv.typing.MatLike,
    image_files: list[str],
    image_objects,
    object_features: tuple[Sequence[KeyPoint], MatLike],
    default_source_name,
    scene_height: int,
    scene_width: int,
    stream_delay,
) -> int:
    global DEBUGGING, OBJECT_TRACING, FRAME_PREPROCESSING, old_frame

    cmodule = importlib.__import__("detector_cython")
    # log.info(frame)
    # keypoints_scene, descriptors_scene = cmodule.detectAndCompute(frame_rgb)

    start_time = time.time()

    ### image preprocessing to reduce costs of feature matching
    if FRAME_PREPROCESSING == True:
        mask_objects = obs.calcActiveFilterMask(frame, OBJECT_TRACING)
        if old_frame is not None:
            frame_diff = detector.preprocessImage(old_frame, frame, mask_objects)
        old_frame = frame.copy()
        frame = frame_diff
    ###

    keypoints_scene: Sequence[KeyPoint] = []
    descriptors_scene: MatLike = []

    keypoints_scene, descriptors_scene = img_detector.calc_descriptors(frame)

    # log.info(
    #     f"{keydes2.getDescriptors().shape} > {descriptors_scene.shape} and {len(keydes2.keypoints)} > {len(keypoints_scene)}"
    # )
    # assert keydes2.getDescriptors().shape[0] == len(
    #     keydes2.keypoints
    # ) and descriptors_scene.shape[0] == len(keypoints_scene)

    i = -1
    for kps_object, des_object in object_features:
        i = i + 1
        img_path = image_files[i]
        img = image_objects[i]

        if DEBUGGING == True:
            log.info(
                f"Searching for object {img_path} with {np.size(kps_object)} kps, {np.size(des_object)} descriptors",
            )

        src, dst = img_detector.calc_matches(
            kps_object, des_object, keypoints_scene, descriptors_scene
        )

        found = np.size(src) > 0 and np.size(dst) > 0

        if found == True:
            corners = img_detector.calc_matches_corners(
                img,
                src,
                dst,
            )

            if np.size(corners) == 0:
                continue

            # log.info(corners)
            corner_points = corners.squeeze()
            img_in_frame = cv.polylines(
                frame_rgb, [corners], True, (0, 255, 0), 2, cv.LINE_AA
            )
            cv.putText(
                img_in_frame,
                os.path.basename(img_path),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                thickness=2,
                color=(0, 255, 0),
                org=corner_points[0] - np.array([0, 6]),
                fontScale=0.8,
            )
            new_width = int(0.5 * scene_width)
            img_in_frame = imutils.resize(img_in_frame, width=new_width)

            mask = cv.fillPoly(
                np.zeros(frame.shape[:2], dtype=np.uint8), [corners], 255
            )

            if DEBUGGING == True:

                cv.imshow("img_in_frame", img_in_frame)
                cv.waitKey()
                # res = cv.bitwise_and(frame_rgb, frame_rgb, mask=mask)
                # cv.imshow("res", res)
                # cv.waitKey()
                cv.destroyAllWindows()

            has_changed_or_is_new = obs.adaptFilterCorners(img_path, corners)
            if has_changed_or_is_new == False:
                continue

            mask_path = join(
                MASK_DIR,
                os.path.basename(img_path).split(".png")[0] + "_mask.png",
            )
            log.info(f"Saving new filter mask to {mask_path} ...")
            duration = time.time() - start_time
            log.info(f"-> took {duration}s to adapt filter")
            cv.imwrite(mask_path, mask)

            # create/update obs filter
            # (if the filter is already active and
            # only the position of img changed
            # then updating the mask image file will be sufficient)
            new_filter = obssocket.OBSFilter(default_source_name, mask_path, corners)
            obs.updateFilter(new_filter, img_path)
        else:
            obs.deleteFilter(
                img_path, delay=max(0, stream_delay - (time.time() - start_time))
            )

    active_count = obs.activeFilterCnt()

    # if active_count == 0:
    #     cv.imshow("show_img", show_img)
    #     cv.waitKey()
    #     cv.destroyAllWindows()

    return active_count


@click.command()
@click.option(
    "--preprocessing", is_flag=True
)  # preprocess frame image by comparing to last frame and only search in changed areas (still buggy - not recommended yet)
@click.option("--debug", is_flag=True)
@click.option(
    "--gpu", is_flag=True
)  # use gpu acceleration with cuda (high gpu load - not recommended while gaming)
@click.argument(
    "resource-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option("--password")
# @click.option(
#     "--tracing", is_flag=True
# )  # support masking objects that only slightly move over the screen instead of immediately appearing objects when they
def main(resource_dir, password, debug, gpu, preprocessing):
    global DEBUGGING, OBJECT_TRACING, GPU_CUDA, FRAME_PREPROCESSING, obs

    DEBUGGING = debug
    OBJECT_TRACING = True
    GPU_CUDA = gpu
    FRAME_PREPROCESSING = preprocessing

    if DEBUGGING == False:
        log.setLevel(level=logging.ERROR)

    with open(os.getcwd() + "/config.json") as settings_file:
        application_settings = json.load(settings_file)

    log.info(f"Application settings: {application_settings}")
    default_scene_name = application_settings["default_scene_name"]
    host = application_settings["obs_server_address"]
    port = application_settings["obs_server_port"]
    log.info(application_settings)

    if os.path.exists(MASK_DIR) == False:
        os.mkdir(MASK_DIR)

    # log.info(json.dumps(ret.datain, indent=4))

    obs = obssocket.OBSConnection(host, port, password)
    atexit.register(obssocket.OBSConnection.handle_cleanup, obs)

    default_source_name, scene_h, scene_w = obs.getSourceInfo(default_scene_name)

    image_files = [
        join(resource_dir, f)
        for f in os.listdir(resource_dir)
        if isfile(join(resource_dir, f))
    ]

    image_objects = [
        cv.cvtColor(cv.imread(path), cv.COLOR_BGR2GRAY) for path in image_files
    ]

    # detector_cpu = detection.ImageDetector(type=detection.Configuration.BRISK_BF)

    if GPU_CUDA == True:
        img_detector = detector.ImageDetector(type=detector.Configuration.SURF_GPU)
    else:
        img_detector = detector.ImageDetector(type=detector.Configuration.SURF_CPU)

    object_features = [
        img_detector.calc_descriptors(img_object) for img_object in image_objects
    ]

    screencap = mss()
    stream_delay = 0.5

    while True:
        start_time = time.time()
        frame_rgb = np.array(screencap.grab(screencap.monitors[1]))

        frame = cv.cvtColor(frame_rgb, cv.COLOR_BGR2GRAY)
        active_count = frame_action(
            img_detector,
            frame,
            frame_rgb,
            image_files,
            image_objects,
            object_features,
            default_source_name,
            scene_h,
            scene_w,
            stream_delay,
        )
        duration = time.time() - start_time
        timediff = max(0, stream_delay - (time.time() - start_time))

        if active_count > 0 and DEBUGGING:
            log.info(f"active filters: {active_count}")
            log.info(f"{duration}s")

        # as the stream is delayed by up to 500 ms anyway we can save come computation time and sleep
        time.sleep(timediff)
