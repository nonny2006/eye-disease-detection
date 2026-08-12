import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Where the dataset lives, and the four class folders we need to loop through
DATA_DIR = "data/retina_dataset/dataset"
CLASS_NAMES = ["1_normal", "2_cataract", "2_glaucoma", "3_retina_disease"]

# Build two parallel lists: every image's file path, and the class it belongs to.
# We need this pairing before we can do a stratified split.
image_paths = []
labels = []

for class_name in CLASS_NAMES:
    class_folder = os.path.join(DATA_DIR, class_name)
    for filename in os.listdir(class_folder):
        image_paths.append(os.path.join(class_folder, filename))
        labels.append(class_name)

# First split: carve off 75% for training, leave 25% aside (X_temp/y_temp)
# to be split again into validation and test below.
# stratify=labels keeps each class's proportion consistent in both halves.
X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths, labels,
    test_size=0.25,
    stratify=labels,
    random_state=42
)

# Second split: divide the remaining 25% into validation (10% of total)
# and test (15% of total) -- 0.6 of 25% = 15%, 0.4 of 25% = 10%.
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.6,
    stratify=y_temp,
    random_state=42
)

# print("Train size:", len(X_train))
# print("Validation size:", len(X_val))
# print("Test size:", len(X_test))

# from collections import Counter
# print("Train class counts:", Counter(y_train))
# print("Validation class counts:", Counter(y_val))
# print("Test class counts:", Counter(y_test))


#LOADING AND RESIZING THE ACTUAL IMAGES
# Target size every image gets resized to -- matches what ResNet50 expects too,
# so we don't need a separate pipeline for the transfer learning comparison later.
IMAGE_SIZE = (224, 224)

def load_and_preprocess(paths):
    """Opens each image file, resizes it, and normalizes pixel values to 0-1."""
    images = []
    for path in paths:
        img = Image.open(path).convert("RGB")  # force 3 color channels, some images sneak in as grayscale/RGBA
        img = img.resize(IMAGE_SIZE)
        img_array = np.array(img) / 255.0  # scale 0-255 pixel values down to 0-1 for stable training - normalization
        images.append(img_array)
    return np.array(images)

# Actually load and preprocess each set now that we know which files belong where
X_train_images = load_and_preprocess(X_train)
X_val_images = load_and_preprocess(X_val)
X_test_images = load_and_preprocess(X_test)

# print("X_train_images shape:", X_train_images.shape)

# Map each class name to a number: 0, 1, 2, 3
label_to_index = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Convert each label list from text to numbers using that mapping
y_train_encoded = [label_to_index[label] for label in y_train]
y_val_encoded = [label_to_index[label] for label in y_val]
y_test_encoded = [label_to_index[label] for label in y_test]

# One-hot encode: turns e.g. 2 into [0, 0, 1, 0] -- required format for
# multi-class classification with softmax output
y_train_final = to_categorical(y_train_encoded, num_classes=4)
y_val_final = to_categorical(y_val_encoded, num_classes=4)
y_test_final = to_categorical(y_test_encoded, num_classes=4)

print("Example label before encoding:", y_train[0])
print("Example label after encoding:", y_train_final[0])
