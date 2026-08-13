# BUILDING THE CNN
from tensorflow.keras import layers, models

def build_model(num_classes=4):
    # Sequential = layers stacked one after another, each one feeding
    # straight into the next -- simplest architecture to build and reason about
    model = models.Sequential()

    '''
    Declares the shape of ONE image the model expects: 224x224 pixels,
    3 color channels. The number of images (batch size) is deliberately
    not specified here -- that can vary between training, validation,
    and single predictions later in the Streamlit app.
    '''
    model.add(layers.Input(shape=(224, 224, 3)))
    

    '''
    First layer -convolutional layer
    First conv layer: scans the image looking for simple, low-level patterns
    like edges and color transitions. 32 filters = 32 different patterns
    it learns to detect at this stage. Each filter produces its own "feature map" 
    a version of the image highlighting where that pattern was found.
    '''
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))

    '''
    Second Layer - Pooling Layer
    Shrinks the feature maps down by taking the max value in each 2x2
    block, halving the width and height. This keeps the strongest signals
    from the conv layer while reducing computation for everything after it,
    and helps the model generalize rather than fixate on exact pixel positions.
    '''
    model.add(layers.MaxPooling2D((2, 2)))

    '''
    Second conv layer: works on the output of the first pool, so it can
    start combining simple edges/textures into slightly more complex shapes.
    More filters (64 vs 32) because there are more possible combinations
    of simple features to detect at this stage.
    '''
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    '''
    Third conv layer: by this point the model is working with fairly
    abstract, combined features from the two layers before it. 128 filters
    continues the doubling pattern, more capacity for more complex
    combinations at this deepest stage.
    '''
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    '''
    Flatten: converts the 3D grid of features (height x width x filters)
    into a single flat list of numbers, so it can be fed into regular
    Dense layers, which only work with flat input.
    '''
    model.add(layers.Flatten())

    '''
    Dense layer: a normal fully-connected layer where every input number
    connects to every one of these 128 neurons. This is where the model
    actually starts combining all the detected features into a decision.
    '''
    model.add(layers.Dense(128, activation="relu"))

    '''
    Dropout: randomly "turns off" 50% of neurons during each training step.
    This forces the model not to rely too heavily on any single neuron,
    which helps prevent overfitting -- especially important given how
    small our dataset is.
    '''
    model.add(layers.Dropout(0.5))

    '''
    Output layer: one neuron per class (4 total). Softmax turns the raw
    outputs into probabilities that sum to 1 -- e.g. [0.05, 0.1, 0.8, 0.05]
    means the model is 80% confident on the 3rd class.
    '''
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",       # adjusts the model's weights during training, adapting the learning rate automatically
        loss="categorical_crossentropy",  # measures how wrong predictions are, for multi-class problems with one-hot labels
        metrics=["accuracy"]     # tracks accuracy during training so we can watch progress
    )

    return model