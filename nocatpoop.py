import sys
import time
import datetime
import traceback
import logging
import cv2 as cv

from utils import get_config_yaml
from inits import init_telegram_notifier, init_camera, init_model
from telegram import TelegramChatSender
from camera import Camera, CameraReadException
from model import Model

logger = logging.getLogger()


def main_loop(model: Model, camera: Camera, telegram_notifier: TelegramChatSender):
    classes_to_detect = ["person", "cat"]
    detection_image_index = 0
    show_stream = get_config_yaml()["show_stream"]
    while True:
        time.sleep(2)

        try:
            image = camera.read()
        except CameraReadException:
            logger.warning("CameraReadException")
            continue

        outputs = model.detect(image=image)
        detections = model.process_outputs(image=image, outputs=outputs, conf=0.55)
        image = model.draw_detections(image=image, detections=detections)

        if show_stream:
            image = model.draw_time(image=image)
            cv.imshow('frame', image)

        if detections:
            logger.debug("camera detect objects")
            for i, detection in enumerate(detections):
                logger.debug(f"detection #{i}: {str(detection)}")

        filtered_classes = [detection for detection in detections if detection.class_name in classes_to_detect]

        for detection in filtered_classes:
            logger.info(f"{detection.class_name} detected")

            save_path = f"/home/danielam/NoCatPoop/outputs/detection/{detection_image_index}.jpg"
            detection_image_index += 1
            print(f"{datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}: {detection.class_name} detected, save image to {save_path}")
            cv.imwrite(save_path, image)

            telegram_notifier.send_image(image_path=save_path, image_caption=f"{detection.class_name} detected")

        if cv.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    camera = init_camera()
    model = init_model()
    telegram_notifier = init_telegram_notifier()

    try:
        main_loop(model=model, camera=camera, telegram_notifier=telegram_notifier)
    except Exception as e:
        logger.error(f"Exit exception {e}")
        logger.error(f"{traceback.format_exc()}")


if __name__ == "__main__":
    main()
