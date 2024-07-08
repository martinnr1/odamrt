# Requirements

## Python >= 3.11 

* for Windows: https://www.python.org/downloads/windows/
* for Linux (debian): `sudo apt install python3.11`

## (Work in progress: OpenCV-contrib with CUDA-Support (Pre-builds))

* non-free opencv algorithms (contrib)
* CUDA support
  <!-- `sudo apt install libgtk2.0-dev pkg-config` `CMAKE_ARGS="-DOPENCV_ENABLE_NONFREE=ON" .venv/bin/python -m pip install --no-binary=opencv-contrib-python opencv-contrib-python`-->



# Usage

## 1. Installation

* Run the build script by typing the following command in a CLI:

    `python setup.py`

## 2. Configuration

* Setup OBS default `scene`, add a `delay filter` of 500 ms to it and assign a `source`
* Enable OBS WebSocket via `Tools -> WebSocket Server Settings`
* Set parameters on `config.json` accordingly
* Insert `.png`-images of the object(s) you do not want to show in your live-stream (they will get masked by a black rectangle) to the folder `img/`

    To guarantee an object getting detected properly make sure your image contains significant features identifying the object itself. If necessary pre-process the image with e.g. GIMP and cut out irrelevant parts before using for detection. 
    
    There are already two examples in `img/` for a map-filter of an online game, delete those images if you do not need them.



## 3. Run 
* Launch the main script by typing

    * on Windows: `.venv\Scripts\python run.py img`

    * on Linux: `.venv/bin/python run.py img/`

    or 

    * `.venv/bin/python run.py img/ --password=<your password>` 

    if you have set a password for the OBS WebSocket Server.