#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import configparser
import cv2
import imutils
import logging
import logging.handlers
import os
import time
import sys

from datetime import datetime
from multiprocessing import Process, Queue


def RGBtoGBR(*args, **kwargs):
    tab = kwargs.get('tab')

    tab.append(tab[0])
    tab = tab[1:]

    return tab


def check_motion(*args, **kwargs):
    conf = kwargs.get("conf")
    queue_check = kwargs.get("q_check")
    queue_write = kwargs.get("q_write")

    angle = conf.getfloat("default", "rotatePicture")
    weight = conf.getfloat("default", "weight")
    delta_thresh = conf.getint("default", "deltaThresh")
    min_area = conf.getint("default", "minArea")
    motion_count = conf.getint("default", "motionNumBeforeWritePic")
    resize = conf.getint("default", "resizeWidth")
    see = conf.getboolean("default", "see")
    which_frame = conf.get("default", "whichFrame")

    gaussian_blur = conf.get("default", "gaussianBlur")
    gaussian_blur = tuple([int(x.strip()) for x in gaussian_blur.split(",")])

    rec_color = conf.get("writing", "rectangleColorRGB")
    rec_color = [int(x.strip()) for x in rec_color.split(",")]
    rec_color = tuple(RGBtoGBR(tab=rec_color))

    avg = None
    counter = 0

    logging.debug("****** Starting check motion ******")
    while True:
        frame = queue_check.get()
        found = False

        if resize > 0:
            frame = imutils.resize(frame, width=resize)

        if angle != 0:
            frame = imutils.rotate_bound(frame, angle)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, gaussian_blur, 0)

        if avg is None:
            avg = blur.copy().astype("float")
            continue

        cv2.accumulateWeighted(blur, avg, weight)
        frame_delta = cv2.absdiff(blur, cv2.convertScaleAbs(avg))

        thresh = cv2.threshold(frame_delta, delta_thresh, 255,
                               cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        if which_frame == "gray":
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif which_frame == "blur":
            frame = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
        elif which_frame == "frameDelta":
            frame = cv2.cvtColor(frame_delta, cv2.COLOR_GRAY2BGR)
        elif which_frame == "thresh":
            frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue

            found = True
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), rec_color, 2)

        if found:
            counter += 1
        else:
            counter -= 1
            if counter < 0:
                counter = 0

        if counter >= motion_count:
            counter = motion_count

            if queue_write.full():
                logging.warning("Queue (size {}) is full !".format(
                    conf.get("default", "maxFrameInQueue")))

            queue_write.put(frame)

        if see:
            cv2.imshow('image', frame)
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                break


def create_image_directory(*args, **kwargs):
    conf = kwargs.get("conf")

    try:
        os.makedirs(conf.get("default", "recordPath"))
    except(FileExistsError):
        pass

    if not os.access(conf.get("default", "recordPath"), os.W_OK):
        logging.critical("You don't have write's permissions on \"{}\"".format(
            conf.get("default", "recordPath")
        ))
        sys.exit(1)


def get_params(*args, **kwargs):
    config_file = kwargs.get("config_file")

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        nargs='+',
                        default=config_file,
                        help="Input file config")

    return parser.parse_args()


def load_config(*args, **kwargs):
    conf = configparser.ConfigParser()
    conf.read(kwargs.get("conf"))

    if not conf.sections():
        print("{} is not valid or is an empty file !".format(
            kwargs.get("conf")), file=sys.stderr)
        sys.exit(1)

    return conf


def log_activity(*args, **kwargs):
    logger = logging.getLogger()
    conf = kwargs.get("conf")

    level = 0
    if conf.get("default", "logLevel") == "debug":
        level = 10
    elif conf.get("default", "logLevel") == "info":
        level = 20
    elif conf.get("default", "logLevel") == "warning":
        level = 30
    elif conf.get("default", "logLevel") == "error":
        level = 40
    elif conf.get("default", "logLevel") == "critical":
        level = 50
    else:
        config_file = kwargs.get("config_file")
        print("\"{}\" is an unknown log level !".format(
            conf.get("default", "logLevel")), file=sys.stderr)
        print("See your \"{}\" !".format(config_file))

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s :: %(levelname)s :: %(message)s")
    file_handler = logging.handlers.RotatingFileHandler(kwargs.get("log"), "a",
                                                        1000000, 1)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    steam_handler = logging.StreamHandler()
    steam_handler.setLevel(level)
    logger.addHandler(steam_handler)


def manage_process(*args, **kwargs):
    process_list = kwargs.get('processList')

    for process in process_list:
        process.start()

    time.sleep(0.1)

    critical_error = False
    while True:
        for process in process_list:
            if not process.is_alive():
                critical_error = True
                logging.critical("{} process has stopped !".format(
                    process.name))

        if critical_error:
            for process in process_list:
                if process.is_alive():
                    process.terminate()

            cv2.destroyAllWindows()

            break

        time.sleep(5)


def read_frames(*args, **kwargs):
    queue = kwargs.get("queue")

    conf = kwargs.get("conf")
    camera = conf.get("default", "camStream")

    try:
        camera = int(camera)
    except ValueError:
        pass

    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        logging.critical("Unable to open \"{}\" camera".format(camera))

    waiting = conf.getfloat("default", "waitAtStart")
    if waiting > 0:
        time.sleep(waiting)

    logging.debug("****** Starting get frame ******")
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        if queue.full():
            logging.warning("Queue (size {}) is full !".format(
                conf.get("default", "maxFrameInQueue")))
            timeToSleep = conf.getfloat("default", "timeSleepQueueIsFull")
            time.sleep(timeToSleep)
        else:
            queue.put(frame)

    frame.release()


def write_frames(*args, **kwargs):
    conf = kwargs.get("conf")
    queue = kwargs.get("queue")

    quality = conf.getint("default", "jpegQuality")
    record_path = conf.get("default", "recordPath")
    text_size = conf.getfloat("writing", "textSize")
    thickness = conf.getint("writing", "thickness")
    writeTextOnPicture = conf.getboolean("writing", "writeTimeOnPicture")

    pos_xy = conf.get("writing", "posXY")
    pos_xy = tuple([int(x.strip()) for x in pos_xy.split(",")])

    colorRGB = conf.get("writing", "colorRGB")
    colorRGB = [int(x.strip()) for x in colorRGB.split(",")]
    colorRGB = tuple(RGBtoGBR(tab=colorRGB))

    while True:
        frame = queue.get()
        d = datetime.now()
        filename_tmp = os.path.join(record_path,
                                    d.strftime("%Y%m%d_%H:%M:%S"))
        ext = ".jpg"

        num = 0
        filename = filename_tmp + ext
        while os.path.exists(filename):
            filename = filename_tmp + "_" + str(num) + ext
            num += 1

        if writeTextOnPicture:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, d.strftime("%d/%m/%Y %H:%M:%S"),
                        pos_xy, font, text_size, colorRGB,
                        thickness, cv2.LINE_AA)

        cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])


def main(*args, **kwargs):
    log_file = kwargs.get("log_file")
    parser = kwargs.get("parser")

    config_file = parser.input

    conf = load_config(conf=config_file)
    log_activity(log=log_file, conf=conf, config_file=config_file)
    create_image_directory(conf=conf)

    queue_frames_check = Queue(conf.getint("default", "maxFrameInQueue"))
    queue_frames_write = Queue(conf.getint("default", "maxFrameInQueue"))
    process_list = []
    process_list.append(Process(target=read_frames, name="read_frames",
                                args=(), kwargs={'conf': conf,
                                                 'queue': queue_frames_check}))
    process_list.append(Process(target=check_motion, name="check_motion",
                                args=(),
                                kwargs={'conf': conf,
                                        'q_check': queue_frames_check,
                                        'q_write': queue_frames_write}))
    process_list.append(Process(target=write_frames, name="write_frames",
                                args=(), kwargs={'conf': conf,
                                                 'queue': queue_frames_write}))
    manage_process(processList=process_list, conf=conf)


if __name__ == "__main__":
    config_file = "surveillance/surveillance/cfg/surveillance_sample.conf"
    log_file = "surveillance/surveillance/error.log"

    # config_file = "/etc/surveillance/surveillance.conf"
    # log_file = "/var/log/surveillance/errors.log"

    parser = get_params(config_file=config_file)
    main(config_file=config_file, log_file=log_file, parser=parser)
