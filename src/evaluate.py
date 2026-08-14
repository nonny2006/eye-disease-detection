# Load the trained model saved, and the test set we've kept untouched
# since Stage 3, this is the first time it's actually being used
from tensorflow.keras.models import load_model
from data_loader import X_test_images, y_test, CLASS_NAMES
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

model = load_model("models/eye_disease_cnn.keras")

'''
model.predict gives back probabilities for all 4 classes per image
(like [0.1, 0.7, 0.1, 0.1]) - argmax picks the class with the highest
probability, converting it into a single predicted class index
'''
predictions = model.predict(X_test_images)
predicted_classes = np.argmax(predictions, axis=1)

'''
y_test is still text labels ("2_cataract" etc), convert to the same
numeric index format as predicted_classes so we can compare them directly
'''
label_to_index = {name: idx for idx, name in enumerate(CLASS_NAMES)}
true_classes = [label_to_index[label] for label in y_test]

'''
Confusion matrix: rows = actual class, columns = predicted class.
The diagonal shows correct predictions; anything off the diagonal
shows exactly which classes the model confuses with each other.
'''
cm = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:")
print(cm)

'''
classification_report gives precision, recall, and f1-score PER CLASS --
this is what actually tells us if the model is just leaning on "normal"
(like we discussed with the 50% baseline) or genuinely learning all 4 classes
'''
report = classification_report(true_classes, predicted_classes, target_names=CLASS_NAMES)
print("\nClassification Report:")
print(report)