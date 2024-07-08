"""
File: run.py
Project: obs-stream-overlay
Created Date: 2024-07-08
Author: martinnr1
-----
Last Modified: Mon Jul 08 2024
Modified By: martinnr1
-----
Copyright (c) 2024
"""

import subprocess
import traceback
import logging as log

import src.core.main as main

if __name__ == "__main__":
    log.info(
        subprocess.Popen(
            ".venv/bin/python src/setup_cython.py build_ext --inplace",
            shell=True,
            stdout=subprocess.PIPE,
        ).stdout.read()
    )

    try:
        main.main()
    except Exception as e:
        log.error(traceback.print_exc())
