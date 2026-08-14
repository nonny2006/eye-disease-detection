from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from data_loader import X_test_images, y_test, CLASS_NAMES
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Load the trained ResNet model
model = load_model("models/eye_disease_resnet.keras")

'''
Same fix as training: reverse our 0-1 normalization, apply ResNet's
own expected preprocessing instead -- otherwise evaluation would be
just as broken as that first training attempt was
'''
X_test_resnet = preprocess_input(X_test_images * 255)

predictions = model.predict(X_test_resnet)
predicted_classes = np.argmax(predictions, axis=1)

label_to_index = {name: idx for idx, name in enumerate(CLASS_NAMES)}
true_classes = [label_to_index[label] for label in y_test]

cm = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix (ResNet50):")
print(cm)

report = classification_report(true_classes, predicted_classes, target_names=CLASS_NAMES)
print("\nClassification Report (ResNet50):")
print(report)