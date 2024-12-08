from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from glob import glob
import json
from tensorflow.keras.metrics import Precision, Recall


# Constants for Data Paths and Hyperparameters
IMG_SIZE = [224, 224, 3]
BATCH_SIZE = 32
EPOCHS = 10
train_dir = '/Users/saisumanthkoppolu/Desktop/Computervision/pythonProject/chest_xray/train'
test_dir = '/Users/saisumanthkoppolu/Desktop/Computervision/pythonProject/chest_xray/test'

def create_model():
    base_model = VGG16(input_shape=IMG_SIZE, weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False
    x = Flatten()(base_model.output)
    output_layer = Dense(len(glob(f"{train_dir}/*")), activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=output_layer)

def compile_model(model):
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
    )

def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.15,
        zoom_range=0.15,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    test_data = test_datagen.flow_from_directory(
        test_dir, target_size=(224, 224), batch_size=BATCH_SIZE, class_mode='categorical'
    )
    return train_data, test_data

def train_model(model, train_data, test_data):
    # In the train_model function
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True),  # Change to .keras
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)
    ]

    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=EPOCHS,
        steps_per_epoch=len(train_data),
        validation_steps=len(test_data),
        callbacks=callbacks
    )
    return history

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')  # Save the figure
    plt.show()

if __name__ == "__main__":
    model = create_model()
    compile_model(model)
    train_data, test_data = create_data_generators()
    history = train_model(model, train_data, test_data)
    plot_training_history(history)
    model.save('final_model.keras')

    # Save training history to JSON
    with open('training_history.json', 'w') as f:
        json.dump(history.history, f)
