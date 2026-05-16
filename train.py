
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_INITIAL = 12
EPOCHS_FINE = 10

# ==========================
# Data Augmentation
# ==========================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    zoom_range=0.25,
    horizontal_flip=True,
    width_shift_range=0.15,
    height_shift_range=0.15
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# ==========================
# Load Dataset
# ==========================
train_data = train_datagen.flow_from_directory(
    "dataset_final2/Training",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_datagen.flow_from_directory(
    "dataset_final2/Testing",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

print("Class Indices:", train_data.class_indices)

# ==========================
# Load MobileNetV2
# ==========================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

# ==========================
# Build Model
# ==========================
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(2, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================
# Callbacks
# ==========================
checkpoint = ModelCheckpoint(
    "best_brain_tumor_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

print("\n--- Initial Training ---\n")

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS_INITIAL,
    callbacks=[checkpoint, early_stop]
)

# ==========================
# Fine Tuning
# ==========================
print("\n--- Fine Tuning ---\n")

base_model.trainable = True

for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS_FINE,
    callbacks=[checkpoint, early_stop]
)

model.save("final_brain_tumor_model.keras")

# ==========================
# Plot Accuracy
# ==========================
acc = history.history["accuracy"] + history_fine.history["accuracy"]
val_acc = history.history["val_accuracy"] + history_fine.history["val_accuracy"]

plt.plot(acc)
plt.plot(val_acc)
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.show()

print("\nTraining Complete ✅")


