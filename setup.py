"""
File: build.py
Project: obs-stream-overlay
Created Date: 2024-07-07
Author: martinnr1
-----
Last Modified: Sun Jul 07 2024
Modified By: martinnr1
-----
Copyright (c) 2024
"""

import subprocess
import platform
import os
import venv
import sys


def system():
    try:
        return platform.system()
    except:
        return "Windows"


def is_windows() -> bool:
    return system() == "Windows"


def is_linux() -> bool:
    return system() != "Windows"


if __name__ == "__main__":
    try:
        print("Setting up python environment...")
        venv.create(".venv", with_pip=True)
        # run("python3 -m venv .venv")

        print("Downloading python modules... (this could take a moment)")
        if is_windows() == True:
            subprocess.check_call(
                [
                    ".venv\\Scripts\\python",
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    "requirements.txt",
                ]
            )
        else:
            exec = "/bin/bash"
            p = subprocess.run(
                f".venv/bin/python -m pip install -r requirements.txt",
                shell=True,
                executable=exec,
            )
            p.check_returncode()

        # print("Setting up opencv-contrib module...")
        # run(
        #     "cp -r venv/lib/python3.11/site-packages/cv2 .venv/lib/python3.11/site-packages/"
        # )
    except subprocess.CalledProcessError as e:
        print(f"Failed to execute command: {e.cmd}, exited with code: {e.returncode}")
