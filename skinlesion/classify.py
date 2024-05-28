# import os
# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torch.onnx
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
#
# model = models.resnet50()
# model.fc = nn.Linear(model.fc.in_features, 14)
#
# model.load_state_dict(torch.load("../skin-lesions-resnet50.pt"))
# model.eval()
#
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device {device}")
# model.to(device)
#
# class_to_label = {}
#
# for key, value in test_dataset.class_to_idx.items():
#     class_to_label[value] = key
#
#
#
# outputs = model(inputs)
#
#     probability = (F.softmax(outputs, dim=1))
#     percentages = probability * 100
#     prediction = torch.argmax(percentages, dim=1)
#     predicted_class = class_to_label[prediction.item()]
#
#     actual_class = class_to_label[labels.item()]
#     if predicted_class == actual_class:
#         correct += 1
#     if reporting:
#         print(f"Predicted class: {predicted_class} with likelihood of {percentages[:, prediction.item()].item()}.")
#         print(f"Actual = {actual_class}")
#     if count % 100 == 0:
#         accuracy = correct / count * 100
#         print(f"Accuracy after {count} images: {accuracy}%")
