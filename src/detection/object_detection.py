"""
File: object_detection.py
Project: obs-stream-overlay
Created Date: 2024-07-02
Author: martinnr1
-----
Last Modified: Sun Jul 07 2024
Modified By: martinnr1
-----
Copyright (c) 2024
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from mss import mss


def detect(img: cv.typing.MatLike, model_file: str) -> bool:

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    model = cv.CascadeClassifier(model_file)

    found = model.detectMultiScale(
        img_gray, minSize=(200, 200), minNeighbors=5, flags=cv.CASCADE_SCALE_IMAGE
    )

    amount_found = len(found)

    if amount_found != 0:

        for x, y, width, height in found:

            cv.rectangle(img_rgb, (x, y), (x + height, y + width), (0, 255, 0), 5)
            cv.putText(
                img_rgb,
                model_file,
                (x, y),
                cv.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(0, 255, 0),
                thickness=5,
            )

    plt.subplot(1, 1, 1)
    plt.imshow(img_rgb)
    plt.show()
    return True


if __name__ == "__main__":
    model = "haar/stop_data.xml"

    screencap = mss()

    screen = np.array(screencap.grab(screencap.monitors[1]))

    detect(screen, model)
