import tensorflow as tf
import os
import shutil

# Base directories
base_dirs = ['DataSet/train', 'DataSet/valid', 'DataSet/test']
output_dirs = ['ProcessedData/train', 'ProcessedData/valid', 'ProcessedData/test']

# Liste des classes
classes = [
    "Green Light", "Red Light", "Speed Limit 10", "Speed Limit 100", "Speed Limit 110",
    "Speed Limit 120", "Speed Limit 20", "Speed Limit 30", "Speed Limit 40", "Speed Limit 50",
    "Speed Limit 60", "Speed Limit 70", "Speed Limit 80", "Speed Limit 90", "Stop"
]

# Fonction pour copier les données
def process_directory(base_dir, output_dir):
    images_dir = os.path.join(base_dir, 'images')
    labels_dir = os.path.join(base_dir, 'labels')

    # Créer les répertoires pour chaque classe
    for class_name in classes:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

    # Parcourir les fichiers d'annotations
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            label_path = os.path.join(labels_dir, label_file)
            image_name = label_file.replace('.txt', '.jpg')  # Supposez les images en .jpg
            image_path = os.path.join(images_dir, image_name)

            if not os.path.exists(image_path):  # Vérifier si l'image existe
                print(f"Image non trouvée pour {label_file}, ignorée.")
                continue

            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    class_id = int(line.split()[0])  # ID de la classe
                    class_name = classes[class_id]

                    # Copier l'image dans le répertoire correspondant
                    dest_dir = os.path.join(output_dir, class_name)
                    shutil.copy(image_path, dest_dir)

# Traiter chaque répertoire (train, valid, test)
for base_dir, output_dir in zip(base_dirs, output_dirs):
    print(f"Traitement du répertoire : {base_dir}")
    process_directory(base_dir, output_dir)


from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_transfer_model(input_shape, num_classes):
    # Load pre-trained MobileNetV2 model without the top layer (for fine-tuning)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    # Unfreeze the last few layers for fine-tuning
    base_model.trainable = True
    # Optionally, unfreeze specific layers (e.g., the last 4 layers):
    # base_model.layers[-4:].trainable = True

    # Add custom layers on top of the pre-trained model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Pooling layer
    x = Dense(128, activation='relu')(x)  # Dense layer
    x = Dropout(0.5)(x)  # Dropout layer for regularization
    predictions = Dense(num_classes, activation='softmax')(x)  # Output layer

    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001),  # Lower learning rate for fine-tuning
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model



from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Rescaling for validation and test data (no augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)

# Generators
train_gen = train_datagen.flow_from_directory(
    'ProcessedData/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'
)

val_gen = val_datagen.flow_from_directory(
    'ProcessedData/valid',
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'
)



from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=5,  # Stop if the validation loss doesn't improve for 5 epochs
    restore_best_weights=True  # Restore the best weights when training stops
)

# Reduce learning rate when validation loss plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # Monitor the validation loss
    factor=0.2,  # Reduce the learning rate by a factor of 2
    patience=3,  # Wait for 3 epochs before reducing the learning rate
    min_lr=1e-6  # Minimum learning rate
)


input_shape = (224, 224, 3)
num_classes = len(train_gen.class_indices)  # Number of classes in the dataset

# Create the model
model = create_transfer_model(input_shape, num_classes)

# Train the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,  # Train for 30 epochs (you can adjust this)
    callbacks=[early_stopping, reduce_lr]  # Add early stopping and learning rate scheduler
)



# Test data generator
test_gen = val_datagen.flow_from_directory(
    'ProcessedData/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    shuffle=False  # Don't shuffle so predictions align with ground truth
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_gen)
print(f"Test Accuracy: {test_accuracy:.2f}")



model.save('road_violation_model3.keras')

import matplotlib.pyplot as plt

# Plot training vs. validation accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training vs. validation loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Générateur spécifique pour le test
test_datagen = ImageDataGenerator(rescale=1./255)

# Générateur de données de test
test_gen = test_datagen.flow_from_directory(
    'ProcessedData/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    shuffle=False
)

# Évaluer le modèle sur les données de test
test_steps = np.ceil(test_gen.samples / test_gen.batch_size).astype(int)
test_loss, test_accuracy = model.evaluate(test_gen, steps=test_steps)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Effectuer les prédictions
y_pred = model.predict(test_gen, steps=test_steps)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_gen.classes

# Fonction pour afficher la matrice de confusion
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matrice de Confusion", fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    # Ajouter les valeurs dans les cellules
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("Classe réelle", fontsize=14)
    plt.xlabel("Classe prédite", fontsize=14)
    plt.tight_layout()
    plt.show()

# Afficher la matrice de confusion
plot_confusion_matrix(y_true, y_pred_classes, test_gen.class_indices.keys())


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Define the prediction function
def predict_image(image_path, model, class_names):
    """
    Predict the classes for a single image.

    Args:
    - image_path: Path to the test image.
    - model: The trained multi-label classification model.
    - class_names: List of class names corresponding to the output vector.

    Returns:
    - predicted_classes: List of classes detected in the image.
    - probabilities: Probability scores for each class.
    """
    # Load and preprocess the image
    image = load_img(image_path, target_size=(224, 224))  # Resize to match model input size
    image_array = img_to_array(image) / 255.0  # Normalize pixel values to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict
    probabilities = model.predict(image_array)[0]  # Get predictions for the first (and only) image in the batch
    predicted_classes = [class_names[i] for i, prob in enumerate(probabilities) if prob > 0.5]  # Threshold = 0.5

    return predicted_classes, probabilities


from tensorflow.keras.models import load_model

# Load the trained model
model_path = '/content/road_violation_model3.keras'  # Path to the saved model file
model = load_model(model_path)


class_names = ["Green Light", "Red Light", "Speed Limit 10", "Speed Limit 100", "Speed Limit 110",
               "Speed Limit 120", "Speed Limit 20", "Speed Limit 30", "Speed Limit 40", "Speed Limit 50",
               "Speed Limit 60", "Speed Limit 70", "Speed Limit 80", "Speed Limit 90", "Stop"]


# Path to a test image
test_image_path = '/content/road822_png.rf.b3ba7f5457042a083bad8aac3fe5f819.jpg'

# Predict the classes for the test image
predicted_classes, probabilities = predict_image(test_image_path, model, class_names)

# Print the results
print(f"Predicted Classes: {predicted_classes}")
print("Class Probabilities:")
for i, prob in enumerate(probabilities):
    print(f"{class_names[i]}: {prob:.2f}")

