"""

Logger file for GPToAnki.py

"""
import time
from pathlib import Path
from tqdm import tqdm
import logging
import rtoml
import json

# adds logger, restrict it to X lines
Path("logs.txt").touch(exist_ok=True)
Path("logs.txt").write_text(
    "\n".join(
        Path("logs.txt").read_text().split("\n")[-100_000:]))
logging.basicConfig(filename="logs.txt",
                    filemode='a',
                    force=True,
                    format=f"{time.ctime()}: %(message)s")
log = logging.getLogger()
log.setLevel(logging.INFO)


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


whi = coloured_log("white")
yel = coloured_log("yellow")
red = coloured_log("red")
