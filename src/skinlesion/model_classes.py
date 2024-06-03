import csv
import logging


class ModelClasses:

    @staticmethod
    def load(class_filename: str) -> dict:
        logging.info(f"Loading classes file {class_filename}")
        class_to_label = {}
        with open(class_filename, mode="r", newline="") as in_file:
            reader = csv.reader(in_file)
            for row in reader:
                class_to_label[int(row[0])] = row[1]
        return class_to_label

    @staticmethod
    def save(class_filename: str, classes: dict):
        with open(class_filename, "w", newline='') as csv_file:
            for index, element in enumerate(classes):
                csv_file.write(f"{index},\"{element}\"\n")
