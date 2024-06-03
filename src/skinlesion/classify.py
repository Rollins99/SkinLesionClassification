import logging
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.onnx
import torch.nn.functional as F

from torchvision import transforms
from PIL.Image import Image as PILImage
from skinlesion.model_classes import ModelClasses


class Classify:
    _instance = None

    def __init__(self, models_dir: str = "./models", model_file: str = "skin-lesions-resnet50.pt",
                 classes_file: str = "classes.csv"):

        self.models_dir = models_dir
        logging.info(f"Model directory={self.models_dir}")

        self.model_file = model_file
        logging.info(f"Model file={self.model_file}")

        self.classes_file = classes_file
        logging.info(f"Classes file={self.classes_file}")
        self.class_to_label = {}

        self.device = None
        self.num_classes = 0
        self.model = None
        self.transform = None
        self.prepared=False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Classify, cls).__new__(cls, *args, **kwargs)
        else:
            logging.info("Using singleton instance of Classify class")

        return cls._instance

    def prepare(self):
        logging.info("Preparing classify...")

        class_filename = os.path.join(self.models_dir, self.classes_file)
        self.class_to_label = ModelClasses.load(class_filename)

        logging.info("Loading model...")
        self.model = models.resnet50()
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_to_label))
        model_filename = os.path.join(self.models_dir, self.model_file)
        self.model.load_state_dict(torch.load(model_filename))

        logging.info("Model into eval mode")
        self.model.eval()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device {self.device}")
        self.model.to(self.device)

        logging.info("Prepare transform")
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.prepared=True

    def classify(self, image: PILImage, top_count: int = 0):
        if not self.prepared:
            raise AssertionError("The classify class has not been prepared. Run prepare()")

        # Must be RGB, transform to tensor
        image = self.transform(image.convert("RGB"))

        # Add extra dimension as we're not processing a batch
        image = image.unsqueeze(0)

        # send to device (e.g. cuda)
        image = image.to(self.device)

        # get output tensor
        output = self.model(image)

        # Compute the class probabilities using the softmax function
        percentages = torch.round((F.softmax(output, dim=1) * 100)).tolist()[0]

        # Prepare class and probability pairs for sorting
        class_prob_pairs = [(self.class_to_label[i], prob) for i, prob in enumerate(percentages)]
        sorted_pairs = sorted(class_prob_pairs, key=lambda x: x[1], reverse=True)

        # Print sorted pairs
        for pair in sorted_pairs:
            logging.debug(f'Class: {pair[0]}, Probability: {pair[1]:.2f}%')

        if top_count == 0:
            top_count = len(self.class_to_label)
        return sorted_pairs[:top_count]