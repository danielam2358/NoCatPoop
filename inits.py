import logging

from camera import Camera
from model import Model
from telegram import TelegramChatSender, TelegramLoggingHandler
from utils import get_config_yaml

TELEGRAM_LOGGER_CONFIG_YAML_FIELD = "telegram_logger"
TELEGRAM_NOTIFIER_CONFIG_YAML_FIELD = "telegram_notifier"
CAMERA_CONFIG_YAML_FIELD = "camera"
MODEL_CONFIG_YAML_FIELD = "model"


def init_telegram_log() -> TelegramChatSender:
    api_token = get_config_yaml()[TELEGRAM_LOGGER_CONFIG_YAML_FIELD]["api_token"]
    chat_id = get_config_yaml()[TELEGRAM_LOGGER_CONFIG_YAML_FIELD]["chat_id"]
    return TelegramChatSender(api_token=api_token, chat_id=chat_id)


def init_logger() -> logging.Logger:
    telegram_log = init_telegram_log()
    telegram_handler = TelegramLoggingHandler(telegram_chat_sender=telegram_log)

    logger = logging.getLogger("NoCatPoop")
    logger.setLevel(level=logging.DEBUG)
    logger.addHandler(hdlr=telegram_handler)
    return logger


def init_telegram_notifier() -> TelegramChatSender:
    api_token = get_config_yaml()[TELEGRAM_NOTIFIER_CONFIG_YAML_FIELD]["api_token"]
    chat_id = get_config_yaml()[TELEGRAM_NOTIFIER_CONFIG_YAML_FIELD]["chat_id"]
    return TelegramChatSender(api_token=api_token, chat_id=chat_id)


def init_camera() -> Camera:
    use_pc_camera = bool(get_config_yaml()[CAMERA_CONFIG_YAML_FIELD]["use_pc_camera"])

    if use_pc_camera:
        pc_camera_channel = get_config_yaml()[CAMERA_CONFIG_YAML_FIELD]["pc_camera_channel"]
        url = pc_camera_channel
    else:
        ip = get_config_yaml()[CAMERA_CONFIG_YAML_FIELD]["ip"]
        user_name = get_config_yaml()[CAMERA_CONFIG_YAML_FIELD]["user_name"]
        password = get_config_yaml()[CAMERA_CONFIG_YAML_FIELD]["password"]
        url_template = get_config_yaml()[CAMERA_CONFIG_YAML_FIELD]["url_template"]
        url = url_template.format(ip=ip, user_name=user_name, password=password)

    return Camera(url=url)


def init_model() -> Model:
    config_file_path = get_config_yaml()[MODEL_CONFIG_YAML_FIELD]["config_file_path"]
    weights_file_path = get_config_yaml()[MODEL_CONFIG_YAML_FIELD]["weights_file_path"]
    classes_file_path = get_config_yaml()[MODEL_CONFIG_YAML_FIELD]["classes_file_path"]
    return Model(config_file_path=config_file_path,
                 weights_file_path=weights_file_path,
                 classes_file_path=classes_file_path)
