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


class Training:
    def __init__(self, datasets_dir: str = "./datasets", batch_size: int = 64, epochs: int = 10,
                 models_dir: str = "./models", model_file: str = "resnet50_skin_lesions.pt",
                 classes_file: str = "classes.csv"):

        logging.info("Creating Training class...")

        self.datasets_dir = datasets_dir
        logging.info(f"Datasets Directory = {self.datasets_dir}")
        self.models_dir = models_dir
        logging.info(f"Models Directory = {self.models_dir}")
        self.classes_file = classes_file
        logging.info(f"Classes File = {self.classes_file}")
        self.batch_size = batch_size
        logging.info(f"Batch Size = {self.batch_size}")
        self.epochs = epochs
        logging.info(f"Epochs = {self.epochs}")
        self.model_file = model_file
        logging.info(f"Output Model File = {self.model_file}")

        self.device = None
        self.train_transform = None
        self.valid_transform = None
        self.train_dataset = None
        self.valid_dataset = None
        self.num_classes = 0
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.valid_loader = None
        self.train_loader = None

    def prepare(self):
        logging.info("Preparing class...")

        logging.info("Creating transformers")

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomPerspective(),
            transforms.RandomRotation(20),
            transforms.RandomAffine(20),
            transforms.RandomAdjustSharpness(20),
            transforms.RandomAutocontrast(),
            transforms.RandomPosterize(20),
            transforms.RandomSolarize(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.valid_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_dataset = datasets.ImageFolder(os.path.join(self.datasets_dir, 'train'),
                                                  transform=self.train_transform)
        self.valid_dataset = datasets.ImageFolder(os.path.join(self.datasets_dir, 'val'),
                                                  transform=self.valid_transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)

        logging.info(f"{len(self.train_loader)} batches in the training dataset")
        logging.info(f"{len(self.valid_loader)} batches in the validation dataset")

        self.num_classes = len(self.train_dataset.classes)
        logging.info(f"{self.num_classes} classes available")

        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device {self.device}")

        if self.device == "cuda":
            torch.cuda.empty_cache()
            self.model.to(self.device)
            logging.info(f"CUDA memory allocated = {torch.cuda.memory_allocated()}")
        else:
            self.model.to(self.device)

    def train(self):

        # for epoch in range(self.epochs):
        #     self.train_epoch(epoch)
        #     self.evaluate_epoch(epoch)

        self.save_trained_model()

    def save_trained_model(self):
        model_filename = os.path.join(self.models_dir, self.model_file)
        logging.info(f"Saving model state to file {model_filename}")
        torch.save(self.model.state_dict(), self.model_file)

        class_filename = os.path.join(self.models_dir, self.classes_file)
        logging.info(f"Saving model classes to file {class_filename}")

        with open(class_filename, "w", newline='') as csv_file:
            for index, element in enumerate(self.train_dataset.classes):
                csv_file.write(f"{index},\"{element}\"\n")

    def evaluate_epoch(self, epoch):
        logging.info(f"Evaluating after Epoch {epoch + 1}...")

        self.model.eval()
        val_loss = 0.0
        corrects = 0
        with torch.no_grad():
            for inputs, labels in self.valid_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                predictions = torch.argmax(outputs, dim=1)
                corrects += torch.sum(predictions == labels.data)
        val_loss = val_loss / len(self.valid_loader.dataset)
        val_acc = corrects.double() / len(self.valid_loader.dataset)
        logging.info(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')

    def train_epoch(self, epoch):
        logging.info(f"Training Epoch {epoch + 1}...")
        self.model.train()
        running_loss = 0.0
        batch_num = 0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            if batch_num % 100 == 0:
                logging.info(f"Batch {batch_num + 1}, Running loss: {running_loss:.4f}")
            batch_num += 1
        epoch_loss = running_loss / len(self.train_loader.dataset)
        logging.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}")
