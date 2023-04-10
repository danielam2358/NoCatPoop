import io
import sys
import threading
import functools
import logging
import typing
import queue
import yaml
from pathlib import Path


class Worker(threading.Thread):

    def __init__(self, que: queue.Queue):
        self.que = que

        threading.Thread.__init__(self, daemon=True)
        self.start()

    def run(self):
        while True:
            job = self.que.get()
            job()
            self.que.task_done()


def run_async_in_queue(func):
    queue_dic: dict[str, queue.Queue] = {}

    @functools.wraps(func)
    def async_func(*args, **kwargs):
        try:
            async_queue = queue_dic[func.__hash__]
        except KeyError:
            async_queue = queue.Queue()
            Worker(que=async_queue)
            queue_dic[func.__hash__] = async_queue

        async_queue.put_nowait(functools.partial(func, *args, **kwargs))

    return async_func


@functools.cache
def get_config_yaml(config_yaml_path=Path(__file__).parent / "config.yaml") -> dict:
    with open(config_yaml_path, "r") as config_yaml_file_object:
        return yaml.safe_load(config_yaml_file_object)


class STDWriter:
    def __init__(self, log_func: typing.Callable):
        self.log_func = log_func
        self.buf = []

    def write(self, msg):
        if msg.endswith('\n'):
            self.buf.append(msg.removesuffix('\n'))
            self.log_func(''.join(self.buf))
            self.buf = []
        else:
            self.buf.append(msg)

    def flush(self):
        pass
