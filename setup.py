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
            exec = os.path.abspath(".venv/Scripts/python")

        else:
            exec = ".venv/bin/python"
        subprocess.check_call(
            [
                exec,
                "-m",
                "pip",
                "install",
                "-r",
                "requirements.txt",
            ]
        )

        # print("Setting up opencv-contrib module...")
        # subprocess.check_call(
        #         [
        #             "cp -r build/target-install/windows/usr/local/lib/* .venv/Lib/",
        #         ]
        #     )
    except subprocess.CalledProcessError as e:
        print(f"Failed to execute command: {e.cmd}, exited with code: {e.returncode}")
