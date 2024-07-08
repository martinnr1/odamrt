"""
 * File: detector_cython.pyx
 * Project: obs-stream-overlay
 * Created Date: 2024-07-04
 * Author: martinnr1
 * -----
 * Last Modified: Sun Jul 07 2024
 * Modified By: martinnr1
 * -----
 * Copyright (c) 2024
"""



#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
from cython.parallel import parallel, prange
cimport numpy as np
cimport cython
cimport openmp
import cv2 as cv
cimport cv2 as cv
from libc.stdio cimport printf


time_ke = 0
ctypedef np.uint8_t TYPE


@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cpdef detectAndCompute(unsigned char[:,:,:] img):
    cdef int N
    cdef Py_ssize_t i,j
   
    cdef np.ndarray[TYPE, ndim=3] arr = np.array(img, dtype=np.uint8)
    cdef np.ndarray[TYPE, ndim=2] img_copy = cv.cvtColor(arr, cv.COLOR_BGR2GRAY)

    return cv.SIFT_create().detectAndCompute(img_copy, None)