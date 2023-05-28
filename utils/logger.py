"""

Logger file for GPToAnki.py

"""
import re
from pathlib import Path
from tqdm import tqdm
import logging
from logging import handlers
import rtoml
import json

Path("logs.txt").touch(exist_ok=True)
log_formatter = logging.Formatter('%(asctime)s ##%(levelname)s %(funcName)s(%(lineno)d)## %(message)s')
file_handler = handlers.RotatingFileHandler(
        "logs.txt",
        mode='a',
        maxBytes=100000,
        backupCount=1,
        encoding=None,
        delay=0,
        )
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_formatter)

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(file_handler)
log_regex = re.compile(" ##.*?## ")


def coloured_log(color_asked):
    """used to print color coded logs"""
    col_red = "\033[91m"
    col_yel = "\033[93m"
    col_rst = "\033[0m"

    # all logs are considered "errors" otherwise the datascience libs just
    # overwhelm the logs

    if color_asked == "white":
        def printer(string, **args):
            if isinstance(string, dict):
                try:
                    string = rtoml.dumps(string)
                except Exception:
                    string = json.dumps(string)
            if isinstance(string, list):
                string = ",".join(string)
            string = str(string)
            log.info(string)
            tqdm.write(col_rst + string + col_rst, **args)
            return string
    elif color_asked == "yellow":
        def printer(string, **args):
            if isinstance(string, dict):
                try:
                    string = rtoml.dumps(string)
                except Exception:
                    string = json.dumps(string)
            if isinstance(string, list):
                string = ",".join(string)
            string = str(string)
            log.info(string)
            tqdm.write(col_yel + string + col_rst, **args)
            return string
    elif color_asked == "red":
        def printer(string, **args):
            if isinstance(string, dict):
                try:
                    string = rtoml.dumps(string)
                except Exception:
                    string = json.dumps(string)
            if isinstance(string, list):
                string = ",".join(string)
            string = str(string)
            log.info(string)
            tqdm.write(col_red + string + col_rst, **args)
            return string
    return printer

def get_log():
    "frequently called: read the most recent log entries and display it in the output field"
    logcontent = []
    with open("logs.txt", "r") as f:
        for line in f.readlines()[-100:]:
            line = line.strip()
            if not line:
                continue
            line = re.sub(log_regex, " >           ", line)
            logcontent.append(line)
    if not logcontent:
        return "Empty log"
    logcontent.reverse()
    return "\n".join(logcontent)


whi = coloured_log("white")
yel = coloured_log("yellow")
red = coloured_log("red")
