import requests
import logging

from utils import run_async_in_queue


class TelegramChatSender:

    def __init__(self, api_token: str, chat_id: str):
        self._api_token = api_token
        self._chat_id = chat_id

    @run_async_in_queue
    def send_message(self, message: str):
        api_url = f'https://api.telegram.org/bot{self._api_token}/sendMessage'
        response = requests.post(api_url, json={'chat_id': self._chat_id, 'text': message})
        return response

    @run_async_in_queue
    def send_image(self, image_path: str, image_caption: str = ""):
        data = {"chat_id": self._chat_id, "caption": image_caption}
        api_url = f'https://api.telegram.org/bot{self._api_token}/sendPhoto'
        with open(image_path, "rb") as image_file:
            response = requests.post(api_url, data=data, files={"photo": image_file})
        return response


class TelegramLoggingHandler(logging.Handler):
    
    def __init__(self, telegram_chat_sender: TelegramChatSender, level=logging.NOTSET):
        super().__init__(level=level)
        self._telegram_chat_sender: TelegramChatSender = telegram_chat_sender

    def emit(self, record: logging.LogRecord):
        text = f"{record.name}::{record.levelname} {record.msg}"
        self._telegram_chat_sender.send_message(message=text)
