from collections import namedtuple
from datetime import datetime
import cv2 as cv
import numpy as np

CLASSES = open('yolo/coco.names').read().strip().split('\n')
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype='uint8')

Detection = namedtuple("Detection", ["box", "confidence", "class_id", "class_name"])


class Model:

    def __init__(self):
        self._net = cv.dnn.readNetFromDarknet('yolo/yolov3.cfg', 'yolo/yolov3.weights')
        self._net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

        ln = self._net.getLayerNames()
        self._ln = [ln[i - 1] for i in self._net.getUnconnectedOutLayers()]

    def detect(self, image: np.ndarray) -> np.ndarray:
        blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        self._net.setInput(blob)
        outputs = self._net.forward(self._ln)

        # combine the 3 output groups into 1 (10647, 85)
        # large objects (507, 85)
        # medium objects (2028, 85)
        # small objects (8112, 85)
        outputs = np.vstack(outputs)
        return outputs

    @staticmethod
    def process_outputs(image: np.ndarray, outputs: np.ndarray, conf: float) -> list[Detection]:
        image_height, image_weight = image.shape[:2]

        detections: list[Detection] = []

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            scores = output[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf:
                x, y, w, h = output[:4] * np.array([image_weight, image_height, image_weight, image_height])
                p0 = int(x - w // 2), int(y - h // 2)
                # p1 = int(x + w // 2), int(y + h // 2)
                boxes.append([*p0, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        indices = cv.dnn.NMSBoxes(boxes, confidences, conf, conf - 0.1)
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
                detection = Detection(
                    box=boxes[i],
                    confidence=confidences[i],
                    class_id=class_ids[i],
                    class_name=CLASSES[class_ids[i]]
                )
                detections.append(detection)

        return detections

    @staticmethod
    def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
        image = image.copy()

        for detection in detections:
            (x, y) = (detection.box[0], detection.box[1])
            (w, h) = (detection.box[2], detection.box[3])
            color = [int(c) for c in COLORS[detection.class_id]]
            cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(detection.class_name, detection.confidences)
            cv.putText(image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return image

    @staticmethod
    def draw_time(image: np.ndarray) -> np.ndarray:
        image = image.copy()

        datetime_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        cv.putText(image, datetime_str, (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return image
