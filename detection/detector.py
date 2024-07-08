"""
File: detection.py
Project: obs-stream-overlay
Created Date: 2024-07-02
Author: martinnr1
-----
Last Modified: Sun Jul 07 2024
Modified By: martinnr1
-----
Copyright (c) 2024
"""

import logging as log
import traceback
import typing
from typing import Sequence

import cv2 as cv
import imutils
import memory_profiler
import numpy as np
from cv2 import Feature2D, KeyPoint
from cv2.typing import MatLike, Point2f


class Configuration:
    ORB_BF = 0  # quick but doesnt find everything
    SIFT_FLANN = 1  # good but very slow
    BRISK_BF = 2  # good and twice that fast as sift (still slow)
    AKAZE_BF = 3  # not good
    KAZE_FLANN = 4  # bad and slow
    SURF_GPU = (5,)
    SURF_CPU = 6


class ArrayAsKey:
    def __init__(self, matlike: cv.typing.MatLike):
        self.array = matlike

    def __hash__(self):
        return hash(str(self.array.tolist))

    def __eq__(self, other) -> bool:
        return np.all(self.array == other)


class ImageDetector:

    # detector_gpu: cv.cuda.SURF_CUDA = None
    # matcher_gpu: cv.cuda.DescriptorMatcher = None
    detector_gpu = None
    matcher_gpu = None

    def __init__(self, type: int = Configuration.SIFT_FLANN, debugging=False):
        self.type = type
        self.DEBUGGING = debugging
        self.object_cache: dict[
            str,
            tuple[
                int, cv.typing.MatLike, typing.Sequence[cv.KeyPoint], cv.typing.MatLike
            ],
        ] = {}
        self.detector: cv.Feature2D = None

        if type == Configuration.ORB_BF:
            self.detector: cv.ORB = cv.ORB_create(
                nfeatures=600,
                scoreType=cv.ORB_FAST_SCORE,  # Scoring Method Variation. Pros: May provide better keypoints for certain images. Cons: May not be significant for all tasks.
                nlevels=1,  # Varying Octave Layers. Pros: Captures multi-scale features. Cons: Increases computational overhead.
                fastThreshold=10,
            )

            self.matcher: cv.BFMatcher = cv.BFMatcher(cv.NORM_HAMMING)

        elif type == Configuration.SIFT_FLANN:
            self.detector = cv.SIFT_create()  #   # cv.xfeatures2d.SURF_create(400)

            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict()

            self.matcher: cv.FlannBasedMatcher = cv.FlannBasedMatcher(
                index_params, search_params
            )
        elif type == Configuration.BRISK_BF:

            self.detector = cv.BRISK_create()  #   # cv.xfeatures2d.SURF_create(400)

            self.matcher: cv.BFMatcher = cv.BFMatcher(cv.NORM_HAMMING)
        elif type == Configuration.AKAZE_BF:
            self.detector = cv.AKAZE_create()  #   # cv.xfeatures2d.SURF_create(400)

            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict()

            self.matcher: cv.FlannBasedMatcher = cv.FlannBasedMatcher(
                index_params, search_params
            )
        elif type == Configuration.KAZE_FLANN:
            self.detector = cv.KAZE_create()  #   # cv.xfeatures2d.SURF_create(400)

            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict()

            self.matcher: cv.FlannBasedMatcher = cv.FlannBasedMatcher(
                index_params, search_params
            )

        elif type == Configuration.SURF_CPU:
            self.detector = (
                cv.xfeatures2d.SURF_create()
            )  #   # cv.xfeatures2d.SURF_create(400)
            self.detector.setHessianThreshold(400)

            self.matcher: cv.BFMatcher = cv.BFMatcher(cv.NORM_L2)
        elif type == Configuration.SURF_GPU:
            self.detector_gpu = cv.cuda.SURF_CUDA.create(400)

            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict()

            self.matcher_gpu = cv.cuda.DescriptorMatcher.createBFMatcher(cv.NORM_L2)

    # with resizing, without rotation etc.
    def match_template(
        self,
        screenshot_path,
        template_path,
        scaling: tuple[float, float, int] = (1.0, 1.0, 1),
    ):
        img = cv.imread(screenshot_path, cv.IMREAD_GRAYSCALE)
        assert img is not None, "file could not be read, check with os.path.exists()"
        img2 = img.copy()
        template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)
        assert (
            template is not None
        ), "file could not be read, check with os.path.exists()"
        w, h = template.shape[::-1]

        method = cv.TM_CCOEFF_NORMED

        for scale in np.linspace(scaling[0], scaling[1], num=scaling[2]):

            new_width = int(template.shape[1] * scale)
            new_height = int(template.shape[0] * scale)

            template_resized = imutils.resize(template, width=new_width)

            # Apply template Matching
            res = cv.matchTemplate(img, template_resized, method)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            cv.rectangle(img, top_left, bottom_right, 255, 2)
            # plt.clf()  # clear plots
            # plt.subplot(121), plt.imshow(res, cmap="gray")
            # plt.title("Matching Result"), plt.xticks([]), plt.yticks([])
            # plt.subplot(122), plt.imshow(img, cmap="gray")
            # plt.title("Detected Point"), plt.xticks([]), plt.yticks([])
            # plt.suptitle(f"scale: {scale}")
            # plt.show()
            # plt.clf()  # clear plots

    def calc_descriptors(self, img: MatLike) -> tuple[Sequence[KeyPoint], MatLike]:

        if self.type == Configuration.SURF_GPU:
            img_gpu = cv.cuda.GpuMat()
            img_gpu.upload(img)

            kps_gpu, des_gpu = self.detector_gpu.detectWithDescriptors(img_gpu, None)
            kps = self.detector_gpu.downloadKeypoints(kps_gpu)
            des = des_gpu.download()
        else:

            kps, des = self.detector.detectAndCompute(img, None)

        return kps, des

    def match_features(self, descriptors_object, descriptors_scene):

        if self.type == Configuration.SURF_GPU:

            des_obj_gpu = cv.cuda.GpuMat()
            des_obj_gpu.upload(descriptors_object)

            des_scene_gpu = cv.cuda.GpuMat()
            des_scene_gpu.upload(descriptors_scene)

            matches = self.matcher_gpu.knnMatch(des_obj_gpu, des_scene_gpu, k=2)
        else:
            matches = self.matcher.knnMatch(descriptors_object, descriptors_scene, k=2)

        return matches

    # with rotation, resizing etc. (homography)
    # @memory_profiler.profile
    def calc_matches(
        self,
        keypoints_object: typing.Sequence[cv.KeyPoint],
        descriptors_object: cv.typing.MatLike,
        keypoints_scene: typing.Sequence[cv.KeyPoint],
        descriptors_scene: cv.typing.MatLike,
    ) -> tuple[np.float32, np.float32]:

        src_pts = []
        dst_pts = []

        if self.DEBUGGING == True:
            for attribute in dir(self.detector):
                if not attribute.startswith("get"):
                    continue
                param = attribute.replace("get", "")
                get_param = getattr(self.detector, attribute)
                val = get_param()
                log.info(f"{param}, '=', {val}")

        if descriptors_scene is None:
            # log.warning("descriptors_scene is None")
            return (src_pts, dst_pts)
        matches = self.match_features(descriptors_object, descriptors_scene)
        if len(matches) < 2:
            log.warning("Not enough matches for iteration")
            return (src_pts, dst_pts)
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)
        except ValueError as e:
            log.error(f"VALUE ERROR: {e} \nmatch count: {len(matches)}")
            log.error(f"good matches: {np.size(good)}")

        MIN_MATCH_CNT = 4
        if self.type == Configuration.SURF_GPU or self.type == Configuration.SURF_CPU:
            MIN_MATCH_CNT = 200
        if len(good) < MIN_MATCH_CNT:
            return (src_pts, dst_pts)

        # extract the matched keypoints
        src_pts = np.float32([keypoints_object[m.queryIdx].pt for m in good]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([keypoints_scene[m.trainIdx].pt for m in good]).reshape(
            -1, 1, 2
        )

        return (src_pts, dst_pts)

    def calc_matches_corners(
        self, img: cv.typing.MatLike, src_pts: np.float32, dst_pts: np.float32
    ) -> np.int32:
        ## find homography matrix and do perspective transform
        corners = []
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        if M is None:
            return corners
        h, w = img.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
            -1, 1, 2
        )
        dst = cv.perspectiveTransform(pts, M)
        corners = np.int32(dst)

        # if self.DEBUGGING == True:
        #     # draw found regions
        #     img2 = cv.polylines(
        #         img_scene, [np.int32(dst)], True, (0, 0, 255), 1, cv.LINE_AA
        #     )
        #     # cv.imshow("found", img2)

        #     ## draw match lines
        #     res: cv.typing.MatLike = cv.drawMatches(
        #         img_object,
        #         keypoints_object,
        #         img2,
        #         keypoints_scene,
        #         good,
        #         None,
        #         flags=2,
        #     )

        #     new_width = int(0.5 * gui.size()[0])
        #     res = imutils.resize(res, width=new_width)
        #     cv.imshow("match", res)
        #     cv.waitKey()
        #     cv.destroyAllWindows()
        return corners


def preprocessImage(
    last_frame: MatLike, new_frame: MatLike, mask_last_detected_objects
) -> MatLike:

    frame_diff = new_frame

    # substract the parts of the image that did not change
    # keep the parts that were filtered by active_filters and look if the object is still there
    current_frame = new_frame.copy()
    frame_masked_parts = cv.bitwise_and(
        current_frame, current_frame, mask=mask_last_detected_objects
    )
    frame_diff_only = cv.subtract(
        current_frame, cv.bitwise_and(last_frame, current_frame)
    )
    # for compute intense detection process only look on places where:
    # either objects were before (for whether or not they are still there)
    # or the frame really differs from last frame (where new objects could have been popped up or known objects could have moved to)
    frame_diff = cv.bitwise_or(frame_diff_only, frame_masked_parts)

    # if DEBUGGING == True:
    #     rect = np.ones(shape=(100, frame.shape[1]), dtype=np.uint8)
    #     text = lambda x: cv.putText(
    #         rect,
    #         x,
    #         fontFace=cv.FONT_HERSHEY_SIMPLEX,
    #         org=(50, 50),
    #         thickness=2,
    #         color=(0, 255, 0),
    #         fontScale=0.8,
    #     )

    #     view_list = [last_frame, current_frame, frame_diff]

    #     captions = []
    #     for v in view_list:
    #         captions.append(text(str(v.dtype)))
    #     caption_rects = cv.hconcat(captions)

    #     views = cv.hconcat(view_list)
    #     show_img = cv.vconcat([caption_rects, views])
    #     show_img = imutils.resize(show_img, width=int(0.8 * gui.size()[0]))
    #     cv.imshow("show_img", show_img)
    #     cv.waitKey()
    #     cv.destroyAllWindows()

    return frame_diff
