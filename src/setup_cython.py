"""
File: setup_cython.py
Project: obs-stream-overlay
Created Date: 2024-07-04
Author: martinnr1
-----
Last Modified: Sun Jul 07 2024
Modified By: martinnr1
-----
Copyright (c) 2024
"""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "detector_cython",
        ["src/detection/detector_cython.pyx"],
        extra_compile_args=["-O3", "-Wno-cpp", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            # ("OMP_NUM_THREADS", 8),
        ],
    )
]
compiler_directives = {"language_level": 3, "embedsignature": True}
extensions = cythonize(extensions, compiler_directives=compiler_directives)

setup(
    name="detector",
    ext_modules=extensions,
    include_dirs=[
        np.get_include(),
    ],
)
