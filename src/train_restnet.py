from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from data_loader import X_train_augmented, y_train_final, X_val_images, y_val_final
from tensorflow.keras.applications.resnet50 import preprocess_input

def build_resnet_model(num_classes=4):
    '''
    Load ResNet50 with its ImageNet-trained weights, but exclude its
    original final classification layer (which was built for 1000
    ImageNet classes, not our 4) -- include_top=False strips that off
    '''
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    '''
    Freeze the base model: its already-learned weights won't be updated
    during our training. We're borrowing its pattern-detection ability,
    not re-teaching it from scratch, that's the whole point of transfer
    learning, and it's what lets a small dataset like ours still work well.
    '''
    base_model.trainable = False

    return base_model

def build_resnet_model(num_classes=4):
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    '''
    GlobalAveragePooling2D: takes ResNet50's output (a grid of features,
    similar in spirit to your from-scratch model's last conv layer) and
    condenses it into a single number per feature channel by averaging.
    This is a common, lighter alternative to Flatten -- fewer parameters,
    less prone to overfitting than flattening a huge grid like we did before.
    '''
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),

        
        # A small dense layer to let the model combine ResNet's features
        # in a way that's useful for OUR specific 4 classes
        
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),

        # Same output layer logic as before: 4 classes, softmax probabilities
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

'''
Our data_loader normalized images to 0-1 for the from-scratch CNN, but
ResNet50 expects its own specific preprocessing (based on how it was
originally trained on ImageNet). We reverse the 0-1 scaling back to
0-255, then apply ResNet50's expected preprocessing instead.
'''
X_train_resnet = preprocess_input(X_train_augmented * 255)
X_val_resnet = preprocess_input(X_val_images * 255)


model = build_resnet_model(num_classes=4)

history = model.fit(
    X_train_resnet, y_train_final,
    validation_data=(X_val_resnet, y_val_final),
    epochs=10,
    batch_size=16
)

model.save("models/eye_disease_resnet.keras")
print("Model saved to models/eye_disease_resnet.keras")