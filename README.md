# Walnut Detection
# Overview
This repository contains a computer vision project where I applied deep learning to distinguish between walnut shells and kernels. The aim was to develop a machine vision solution for automating the separation of kernels from shells, a process currently done manually in the industry, which is highly labor-intensive.

The challenge with automating this task is that both the shells and kernels are similar in color and texture, which makes traditional machine learning techniques less effective. However, deep learning, when provided with sufficient data, can perform significantly better in such scenarios.

# Project Details
Due to limited resources, the dataset initially consisted of only 10 images. I used 9 images for training and 1 for testing (Walnut10.jpg). The workflow began with image annotation using the LabelImg tool, with the original annotations and images stored in the data directory.

To enhance the dataset, I applied image augmentation using the Albumentations library, increasing the dataset from 9 to 100 images. Out of these, 80 images were used for training various YOLOv8 models (n, s, m, l), while the remaining 20 images were set aside for validation. 

# Training and Results
Given the limited dataset, the model performed exceptionally well. The trained weights are available in the weights directory. During training, I experimented with different image resolutions since the original image size (6944x9248) was quite large. After testing, a resolution of 1280x1280 pixels provided the best results.

The key performance metric for this project is detection speed. Using the script in code/time_test, I tested the model on an NVIDIA GeForce GTX 1650 GPU and achieved a detection time of 0.2 seconds per image. This suggests that the model is capable of supporting real-time kernel-shell separation.

# Future Improvements
While the current results are promising, additional data would likely improve the model's accuracy. Further experimentation with more advanced models such as YOLOv9 or YOLOv10 could also yield even better performance.

