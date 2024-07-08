"""
File: obssocket.py
Project: obs-stream-overlay
Created Date: 2024-07-06
Author: martinnr1
-----
Last Modified: Sun Jul 07 2024
Modified By: martinnr1
-----
Copyright (c) 2024
"""

import logging
import logging as log
import os
import time
from multiprocessing import Lock, Process
from os.path import dirname, isfile, join, realpath

import cv2 as cv
import numpy as np
import obswebsocket
from obswebsocket import events, obsws, requests


class OBSFilter:
    def __init__(self, source: str, mask: str, corners: np.int32):
        self.source = source
        self.mask = mask
        self.corners = corners


class OBSConnection:
    lck = Lock()
    obs: obsws = None
    active_filters: dict[str, OBSFilter] = {}

    def __init__(self, host, port, password) -> None:
        try:
            self.obs = obsws(host, port, password)
            self.obs.register(self.on_event)
            self.obs.register(self.on_switch, events.SwitchScenes)
            self.obs.register(self.on_switch, events.CurrentProgramSceneChanged)

            # Toggle the mute state of your Mic input
            self.obs.connect()
        except obswebsocket.exceptions.ConnectionFailure as e:
            log.error(
                "Could not connect obs websocket. Please check OBS is running and properly configured in file 'config.json'"
            )
            exit(-1)

    def __del__(self):
        self.handle_cleanup()
        self.obs.disconnect()

    def handle_cleanup(self):
        for mask in self.active_filters:
            self.delete_filter(self.active_filters[mask])
        log.info(
            f"Successfully removed {len(self.active_filters)} temporary filters from obs"
        )

    def on_event(self, message):
        log.info("Got message: {}".format(message))

    def on_switch(self, message):
        log.info("You changed the scene to {}".format(message.getSceneName()))

    def delete_filter(self, filter: OBSFilter):

        with self.lck:
            ret = self.obs.call(
                requests.RemoveSourceFilter(
                    sourceName=filter.source,
                    filterName=os.path.basename(filter.mask),
                )
            )
            log.debug(ret)
        return ret

    def delayed_remove(self, delay: float, filter: OBSFilter):
        time.sleep(delay)
        self.delete_filter(filter)

    def add_filter(self, filter: OBSFilter):

        filterSettings = {
            "color": 4294967295,
            "image_path": os.path.abspath(filter.mask),
            "opacity": 1.0,
            "type": "blend_sub_filter.effect",
        }

        with self.lck:

            ret = self.obs.call(
                requests.CreateSourceFilter(
                    sourceName=filter.source,
                    filterName=os.path.basename(filter.mask),
                    filterKind="mask_filter_v2",
                    filterSettings=filterSettings,
                )
            )
        return ret

    def getSourceInfo(self, scene_name) -> tuple[str, int, int]:
        with self.lck:
            ret = self.obs.call(requests.GetSceneItemList(sceneName=scene_name))
        items = ret.datain["sceneItems"]
        for item in items:
            if item["sourceName"] is not None:
                source_name = item["sourceName"]
                scene_h = int(item["sceneItemTransform"]["sourceHeight"])
                scene_w = int(item["sceneItemTransform"]["sourceWidth"])
        return source_name, scene_h, scene_w

    def activeFilterCnt(self) -> int:
        return len(
            [m for m in self.active_filters if self.active_filters[m] is not None]
        )

    def calcActiveFilterMask(self, frame, expand_corners=True):
        mask_objects = np.zeros(frame.shape[:2], dtype=np.uint8)
        for m in self.active_filters:
            corners = self.active_filters[m].corners
            corner_points = corners.squeeze()
            res_h, res_w = (
                corner_points[1][1] - corner_points[0][1],
                corner_points[3][0] - corner_points[0][0],
            )

            if expand_corners == True:
                # let rectangle defined by corners grow by 1/2 of its original size into each direction
                # to allow tracking the image object in the new frame if its position has only slightly changed
                # and difference-frame is corrupted by this
                corners_expanded = corner_points
                expansion_factor = 0.25
                corners_expanded[0] = [
                    max(0, corners_expanded[0, 0] - int(res_w * expansion_factor)),
                    max(0, corners_expanded[0, 1] - int(res_h * expansion_factor)),
                ]
                corners_expanded[1] = [
                    max(0, corners_expanded[1, 0] - int(res_w * expansion_factor)),
                    max(0, corners_expanded[1, 1] + int(res_h * expansion_factor)),
                ]
                corners_expanded[2] = [
                    max(0, corners_expanded[2, 0] + int(res_w * expansion_factor)),
                    max(0, corners_expanded[2, 1] + int(res_h * expansion_factor)),
                ]
                corners_expanded[3] = [
                    max(0, corners_expanded[3, 0] + int(res_w * expansion_factor)),
                    max(0, corners_expanded[3, 1] - int(res_h * expansion_factor)),
                ]
                corners = corners_expanded

            # if object is still there it will be detected again
            # otherwise the filter for it gonna be removed (see image_files loop)
            mask_objects = cv.fillPoly(mask_objects, [corners], 255)
        return mask_objects

    def deleteFilter(self, img_path: str, delay: float = 0.0):
        if img_path in self.active_filters:
            # delete obs filter
            old_filter = self.active_filters[img_path]

            del self.active_filters[img_path]
            log.info(
                f"[-] Removing filter at position {old_filter.corners.tolist()} scheduled"
            )

            # don't erase the filter in obs too early

            # sleep inside separate process so if in the meanwhile other objects appeared on screen we will detect them
            Process(
                target=self.delayed_remove,
                args=(
                    delay,
                    old_filter,
                ),
            ).start()

    def updateFilter(self, new_filter: OBSFilter, img_path: str):
        if img_path not in self.active_filters:
            self.add_filter(new_filter)
            log.info(
                f"[+] Added new filter for object {img_path} at position {new_filter.corners.tolist()}"
            )

        self.active_filters[img_path] = new_filter

    def adaptFilterCorners(self, img_path: str, new_corners: np.float32) -> bool:
        if img_path in self.active_filters:

            if (
                np.array_equal(self.active_filters[img_path].corners, new_corners)
                == True
            ):
                # log.info(
                #     f"[i] The image masked by {active_filters[img_path].mask} is still at the same position"
                # )
                return False

            log.info(
                f"[*] Moved filter by {(self.active_filters[img_path].corners-new_corners).tolist()}"
            )
            self.active_filters[img_path].corners = new_corners
            return True

        return True
