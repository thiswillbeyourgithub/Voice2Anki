import pytesseract
import ftfy
import numpy as np
from tqdm import tqdm
import cv2
from bs4 import BeautifulSoup
import re
from textwrap import dedent
from joblib import Memory

from .logger import red, yel

memory = Memory(".ocr_cache", verbose=0)

# pre compiled regex
bbox_regex = re.compile(r'bbox\s(\d+)\s(\d+)\s(\d+)\s(\d+)')
confidence_regex = re.compile(r'x_wconf\s(\d+)')
newlines_regex = re.compile(r'\n\s*\n')

# colors
col_red = "\033[91m"
col_yel = "\033[93m"
col_rst = "\033[0m"

# msic
tesse_config = "--oem 3 --psm 11 -c preserve_interword_spaces=1"


@memory.cache
def get_text(img_path: str):
    img = cv2.imread(img_path, flags=1)

    # remove alpha layer if found
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # take greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # preprocess several times then OCR, keep the one with the highest median confidence
    preprocessings = {}

    # sharpen a bit for testing
    gray_sharp = cv2.addWeighted(gray, 1.5, cv2.GaussianBlur(gray, (0,0), 10), -0.5, 0)

    # source:
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#otsus-binarization
    # Otsu's thresholding
    _, sharpened = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    preprocessings["otsu1_nosharp"] = {"image": sharpened}
    _, sharpened = cv2.threshold(gray_sharp, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    preprocessings["otsu1_sharp"] = {"image": sharpened}

    # # Otsu's thresholding after Gaussian filtering
    # blur = cv2.GaussianBlur(gray,(5,5),0)
    # _, sharpened = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # preprocessings["otsu2_nosharp"] = {"image": sharpened}
    # blur = cv2.GaussianBlur(gray_sharp,(5,5),0)
    # _, sharpened = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # preprocessings["otsu2_sharp"] = {"image": sharpened}

    # # adaptative gaussian thresholding
    # sharpened = cv2.adaptiveThreshold(
    #         gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
    #         21, -5)
    # preprocessings["gauss_nosharp"] = {"image": sharpened}
    # sharpened = cv2.adaptiveThreshold(
    #         gray_sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
    #         21, -5)
    # preprocessings["gauss_sharp"] = {"image": sharpened}

    max_med = 0  # median
    max_mean = 0  # mean
    best_method = None
    for method, value in preprocessings.items():
        # use pytesseract to get the text
        hocr_temp = do_ocr(value["image"])  # cached
        preprocessings[method]["hocr"] = hocr_temp

        # load hOCR content as html
        soup = BeautifulSoup(hocr_temp, 'html.parser')

        # get all words
        all_words_temp = soup.find_all('span', {'class': 'ocrx_word'})

        # get confidence
        confidences = []
        for w in all_words_temp:
            confidence = get_w_conf(w)
            confidences.append(confidence)
        med = np.median(confidences)
        preprocessings[method]["median"] = med
        mean = np.sum(confidences) / len(confidences)
        preprocessings[method]["mean"] = mean
        if med >= max_med and mean >= max_mean:
            max_med = med
            max_mean = mean
            best_method = method
            hocr = hocr_temp
            all_words = all_words_temp

    red(f"Best preprocessing method: {best_method} with score {max_med:.2f} {max_mean:.2f}")
    for m, v in preprocessings.items():
        yel(f"* {m}: {v['median']:.2f}  {v['mean']:.2f}")
    del preprocessings

    # determine the average word height to use as threshold for the newlines
    # also average length of a letter
    heights = []
    lengths = []
    levels = []  # distance from top of the screen
    all_bbox = {}
    for word in all_words:
        all_bbox[word] = get_w_dim(word)
        bbox = all_bbox[word]
        levels.append(bbox[1])
        heights.append(bbox[3] - bbox[1])
        text = word.get_text()
        if " " not in text and len(text) >= 3 and len(text) <= 20:
            length_bbox = bbox[2] - bbox[0]
            lengths.append(length_bbox / len(text))
    char_width = np.median(lengths)  # unit: pixel per character
    median_height = np.median(heights)
    max_h = max(levels)
    min_h = min(levels)

    # figure out from the line indices how how much to ignore as same line
    sorted_lev = np.array(sorted(levels)[len(levels)//3:len(levels)*2//3])  # focus on middle third
    diff = (sorted_lev - np.roll(sorted_lev, +1))[1:-1]
    try:
        merge_thresh = np.quantile(diff[np.where(diff != 0)], 0.1)
    except Exception as err:
        yel(f"Set line merging threshold to 0 because caught exception: '{err}'")
        merge_thresh = 0
    red(f"Line merging threshold: {merge_thresh}")

    # figure out the height of the line to scan at each iteration
    try:
        newline_threshold = np.quantile(diff[np.where(diff > median_height)], 0.5)
    except Exception as err:
        yel(f"Set newline threshold to median_height because caught exception: '{err}'")
        newline_threshold = median_height
    red(f"Median line diff: {newline_threshold}, median height: {median_height}")

    # figure out the offset that maximizes the number of words per lines
    incr = 5
    offset_to_try = [i / incr * min_h / 2 for i in range(0, incr, 1)]
    scores = {}
    w_todo = all_words
    # skip the first section of the text because it's sometimes headers
    #w_todo = [w for w in all_words if get_w_dim(w)[1] > (max_h - min_h) / 5]
    for offset in offset_to_try:
        w_done = []
        scan_lines = [min_h + offset]
        while scan_lines[-1] < max_h + offset * 2:
            scan_lines.append(scan_lines[-1] + newline_threshold)
        temp = []
        for y_scan in scan_lines:
            buff = [w for w in w_todo if w not in w_done and all_bbox[w][1] <= y_scan + merge_thresh]
            if not buff:
                continue
            w_done.extend(buff)
            temp.append(len(buff))
        scores[offset] = sum(temp) / len(temp)

    for k, v in scores.items():
        if v == max(scores.values()):
            best_offset = k
            break
    red(f"Best offset: {best_offset})")

    # reset the best iterator
    w_done = []
    w_todo = [w for w in all_words]
    scan_lines = [min_h + best_offset]
    while scan_lines[-1] < max_h + best_offset * 2:
        scan_lines.append(scan_lines[-1] + newline_threshold)
    output_str = ''
    prev_y1 = None

    for y_scan in scan_lines:
        # keep only words not added that are in the the scan line
        buff = [w for w in w_todo if w not in w_done and all_bbox[w][1] < y_scan]
        if not buff:
            # go straight to next line because no words matched
            continue
        w_done.extend(buff)

        # sort words to make sure they are left to right
        ocr_words = sorted(buff, key=lambda x: all_bbox[x][0])

        # Extract text and format information
        line_text = ''
        for idx, word in enumerate(ocr_words):
            text = word.get_text()
            x0, y0, x1, y1 = all_bbox[word]

            if idx:
                prev_x1, prev_y1 = all_bbox[ocr_words[idx-1]][2:4]
                spaces_before = ' ' * int(max(1, (x0 - prev_x1) / char_width))
            else:
                spaces_before = ' ' * int(max(1, ((x0) / char_width)))

            confidence = get_w_conf(word)

            if confidence < 50:
                tqdm.write(f"* Low confidence: {confidence}: {text}")

            line_text += f'{spaces_before}{text}'

            if prev_y1 is not None and abs(y0 - prev_y1) >= newline_threshold:
                output_str += '\n'

        output_str += f'{line_text}\n'
        prev_y1 = y1

    # remove useless indentation
    output_str = dedent(output_str)

    # remove too many newlines
    while "\n\n" in output_str:
         output_str = output_str.replace("\n\n", "\n")
    #output_str = "\n".join([li for li in output_str.split("\n") if li.strip() != ""])
    #output_str = re.sub(newlines_regex, '\n', output_str)

    # just in case
    output_str = ftfy.fix_text(output_str)

    return output_str


def get_w_dim(ocrx_word):
    "return x0, y0, x1, y1"
    return [int(x) for x in re.findall(bbox_regex, ocrx_word["title"])[0]]


def get_w_conf(ocrx_word):
    "return x0, y0, x1, y1"
    out = re.findall(confidence_regex, ocrx_word["title"])[0]
    return int(out)


def do_ocr(img):
    return pytesseract.image_to_pdf_or_hocr(
            img,
            lang="fra",
            config=tesse_config,
            extension="hocr",
            )
