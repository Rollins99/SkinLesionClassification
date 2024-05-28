import logging
from skinlesion.training.training import Training

logging.basicConfig(format='%(asctime)s.%(msecs)d [%(levelname)s] %(thread)d %(module)s.%(funcName)s.%(lineno)d | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)

training_session = Training(data_dir="./datasets", batch_size=64, model_file="skin-lesions-resnet50-wpoz017.pt")

training_session.prepare()

training_session.train()
