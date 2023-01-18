import cv2 as cv
import numpy as np


class Camera:

    def __init__(self, url: str):
        self._vid = cv.VideoCapture(url)
        assert self._vid.isOpened()

    def read(self) -> np.ndarray:
        ret, frame = self._vid.read()
        assert ret
        return frame

    def stream(self):
        while True:
            cv.imshow('frame', self.read())
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
