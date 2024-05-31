import logging
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

for i in range(1, len(sys.argv)):

    arg = sys.argv[i].lower()

    if arg == "train":
        train = True
        logging.info("'train' selected")
    elif arg == "classify":
        classify = True
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
    predictions = classifier.classify(Image.open("CHP_02_01_1.jpg"))
    logging.info(predictions)


