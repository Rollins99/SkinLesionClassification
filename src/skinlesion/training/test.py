import csv
import os
from itertools import chain

import torch.nn as nn
import torchvision.models as models
import torch.onnx
import logging

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

from skinlesion.model_classes import ModelClasses


class Tester:
    def __init__(self, datasets_dir: str = "./datasets", batch_size: int = 64, models_dir: str = "./models",
                 model_file: str = "skin-lesions-resnet50.pt",
                 classes_file: str = "classes.csv"):

        logging.info("Creating Testing class...")

        self.datasets_dir = datasets_dir
        logging.info(f"Datasets Directory = {self.datasets_dir}")
        self.classes_file = classes_file
        logging.info(f"Classes File = {self.classes_file}")
        self.models_dir = models_dir
        logging.info(f"Models Directory = {self.models_dir}")
        self.model_file = model_file
        logging.info(f"Output Model File = {self.model_file}")
        self.batch_size = batch_size
        logging.info(f"Batch Size = {self.batch_size}")

        self.device = None
        self.test_transform = None
        self.test_dataset = None
        self.num_classes = 0
        self.model = None
        self.test_loader = None
        self.class_to_label = None
        self.prepared = False

    def prepare(self):
        logging.info("Preparing test...")

        logging.info("Creating transformer")

        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.test_dataset = datasets.ImageFolder(os.path.join(self.datasets_dir, 'test'),
                                                 transform=self.test_transform)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        logging.info(f"{len(self.test_loader)} batches in the testing dataset")

        self.num_classes = len(self.test_dataset.classes)
        logging.info(f"{self.num_classes} classes available in Testing dataset")

        class_filename = os.path.join(self.models_dir, self.classes_file)
        self.class_to_label = ModelClasses.load(class_filename)

        if len(self.class_to_label) != len(self.test_dataset.class_to_idx):
            logging.warning(
                f"The expected class count was {len(self.class_to_label)}, but the actual class count was {len(self.test_dataset.class_to_idx)}")

        logging.info("Loading model...")
        self.model = models.resnet50()
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_to_label))
        model_filename = os.path.join(self.models_dir, self.model_file)
        self.model.load_state_dict(torch.load(model_filename))

        logging.info("Model into eval mode")
        self.model.eval()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device {self.device}")

        if self.device == "cuda":
            torch.cuda.empty_cache()
            self.model.to(self.device)
            logging.info(f"CUDA memory allocated = {torch.cuda.memory_allocated()}")
        else:
            self.model.to(self.device)
        self.prepared = True

    def test(self):
        if not self.prepared:
            raise AssertionError("The testing class has not been prepared. Run prepare()")

        logging.info(f"Evaluating model...")
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                if total % 100 == 0 :
                    logging.info(f"Accuracy: {(100 * correct / total)}")

            logging.info(f"Final accuracy value: {(100 * correct / total)}")
