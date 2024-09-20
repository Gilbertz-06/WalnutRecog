import os
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


# Read YOLO annotations
def read_yolo_annotations(annotation_path):
    with open(annotation_path, 'r') as file:
        boxes = []
        for line in file.readlines():
            class_label, x_center, y_center, width, height = [
                float(x) if float(x) != int(float(x)) else int(x)
                for x in line.replace('\n', '').split()
            ]
            boxes.append([x_center, y_center, width, height, class_label])
    return boxes


# Write YOLO annotations
def write_yolo_annotations(annotation_path, bboxes):
    with open(annotation_path, 'w') as file:
        for bbox in bboxes:
            x_center, y_center, width, height, class_label = bbox
            file.write(f'{class_label} {x_center} {y_center} {width} {height}\n')


# Transformations
transform = A.Compose([
    # A.HorizontalFlip(p=0.5),
    # A.RandomBrightnessContrast(p=0.5),
    # A.ShiftScaleRotate(p=0.5)
    # A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3)
    A.GaussNoise(p=0.8),
    # A.Blur(blur_limit=7, always_apply=False,p=1)

], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Directory wit images and annotations
images_dir = "../data/image"
annotations_dir = "../data/txt"

# Counter for unique augmentation filenames
augmentation_counter = 1

# Augment images and annotations
for image_name in os.listdir(images_dir):
    if image_name.endswith('.jpg'):
        # Read image
        image_path = os.path.join(images_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read corresponding annotation file
        annotation_name = image_name.replace('.jpg', '.txt')
        annotation_path = os.path.join(annotations_dir, annotation_name)
        bboxes = read_yolo_annotations(annotation_path)

        # Generate unique filenames for the augmented image and annotation
        base_name = os.path.splitext(image_name)[0]
        augmented_image_name = f"{base_name}_gn_{augmentation_counter}.jpg"
        augmented_annotation_name = f"{base_name}_gn_{augmentation_counter}.txt"

        # Extract class labels
        class_labels = [bbox[4] for bbox in bboxes]

        # Apply transformations
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']

        # Convert RGB to BGR for OpenCV
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)

        # Save augmented image with a unique name
        cv2.imwrite(f'../results/aug//{augmented_image_name}', transformed_image)

        # Save augmented annotations with a unique name
        write_yolo_annotations(f'../results/aug//{augmented_annotation_name}', transformed_bboxes)

        # Increment the counter for the next augmentation
        augmentation_counter += 1

print("Augmentation completed!")
