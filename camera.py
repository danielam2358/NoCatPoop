import logging
import threading
import time

import cv2 as cv
import numpy as np

logger = logging.getLogger("NoCatPoop")


class CameraReadException(Exception):
    pass


class Camera:

    def __init__(self, url: str):
        self._url = url
        self._lock = threading.Lock()
        self._vid = None
        self._last_ret = None
        self._latest_frame = None
        self._init_video()

        # Start the thread to read frames from the video stream
        self.thread = threading.Thread(target=self.reader, args=())
        self.thread.daemon = True
        self.thread.start()

    def _init_video(self):
        self._vid = cv.VideoCapture(self._url, cv.CAP_FFMPEG)
        assert self._vid.isOpened()
        # self._vid.set(cv.CV_CAP_PROP_BUFFERSIZE, 3)

    def reader(self):
        while True:
            time.sleep(0.001)
            if self._vid.isOpened():
                with self._lock:
                    self._last_ret, self._latest_frame = self._vid.read()

    def read(self) -> np.ndarray:
        with self._lock:
            ret, frame = self._last_ret, self._latest_frame
            if not ret:
                self._init_video()
                raise CameraReadException()
        return frame.copy()

    def stream(self):
        while True:
            cv.imshow('frame', self.read())
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
