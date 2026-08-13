'''
Quick check: print the model architecture before training, so we can see
the shape of data flowing through each layer and confirm nothing looks off
'''
from model import build_model

model = build_model(num_classes=4)
model.summary()

'''
Import everything already built: the full data pipeline (which runs
automatically when this file is imported) and the model architecture
'''
from data_loader import X_train_augmented, y_train_final, X_val_images, y_val_final
from model import build_model

# Build the CNN using the architecture just defined
model = build_model(num_classes=4)

# Train the model
history = model.fit(
    X_train_augmented, y_train_final,
    validation_data=(X_val_images, y_val_final),
    epochs=20,
    batch_size=16
)

# Save the trained model so the Streamlit app can load it later
model.save("models/eye_disease_cnn.keras")
print("Model saved to models/eye_disease_cnn.keras")