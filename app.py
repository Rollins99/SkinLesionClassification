import logging
import os
import sys

from PIL import Image
from skinlesion.classify import Classify
from skinlesion.training.training import Training

logging.basicConfig(
    format='%(asctime)s.%(msecs)d [%(levelname)s] %(thread)d %(module)s.%(funcName)s.%(lineno)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG)

train = False
classify = False
test = False
server = False
filename = None

for i in range(1, len(sys.argv)):

    arg = sys.argv[i].lower()

    if arg == "train":
        train = True
        logging.info("'train' selected")
    elif arg.startswith("classify"):
        classify = True
        parts = arg.split("=")
        if len(parts) != 2:
            logging.error("Expecting 'classify=<filename>'")
            sys.exit(-4)

        filename = parts[1]
        logging.info("'classify' selected")
    elif arg == "test":
        test = True
        logging.info("'test' selected")
    elif arg == "server":
        server = True
        logging.info("'server' selected")
    else:
        logging.error("Invalid command line argument: {}. Stopping".format(sys.argv[i]))
        sys.exit(-2)

if not train and not classify and not test and not server:
    logging.error("Start the application with either 'train', 'classify', 'test' and/or 'server'. Stopping")
    sys.exit(-1)

if train:
    logging.info("Beginning training...")
    training_session = Training(datasets_dir="./datasets", batch_size=64, model_file="skin-lesions-resnet50.pt",
                                models_dir="./models", classes_file="classes.csv")
    training_session.prepare()
    training_session.train()

if classify:
    logging.info("Preparing classifier...")
    classifier = Classify()
    if filename is None or not os.path.exists(filename):
        logging.error("File for classification does not exist ")
        sys.exit(-3)

    predictions = classifier.classify(Image.open(filename), top_count=1)
    logging.info(predictions)

if test:
    logging.info("TODO: Running tests...")

if server:
    logging.info("TODO: Running server...")
