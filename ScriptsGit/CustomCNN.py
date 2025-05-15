# Imports

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Constants
BATCH_SIZE = 16
IMAGE_SIZE = (224, 224)

# Paths
base_dir = "/home/u243945/Knee Osteoarthritis Classification/Data_thesis"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# Image generators (no augmentation)
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=True,
    seed=SEED
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False,
    seed=SEED
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False,
    seed=SEED
)

# Define model builder
def build_model(hp):
    K.clear_session()
    model = Sequential()
    model.add(Input(shape=(224, 224, 1)))

    # Block 1
    model.add(Conv2D(filters=hp.Int('filters_1', 16, 64, step=16), kernel_size=3, activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(hp.Choice('l2_reg', [1e-4, 1e-5]))))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    # Block 2
    model.add(Conv2D(filters=hp.Int('filters_2', 32, 96, step=32), kernel_size=3, activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(hp.Choice('l2_reg', [1e-4, 1e-5]))))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    # Block 3
    model.add(Conv2D(filters=hp.Int('filters_3', 64, 128, step=32), kernel_size=3, activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(hp.Choice('l2_reg', [1e-4, 1e-5]))))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    # Optional 4th block (more depth)
    if hp.Boolean('use_block4'):
        model.add(Conv2D(filters=hp.Int('filters_4', 64, 128, step=32), kernel_size=3, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(hp.Choice('l2_reg', [1e-4, 1e-5]))))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(hp.Int('dense_units', 192, 448, step=64), activation='relu',
                    kernel_regularizer=regularizers.l2(hp.Choice('l2_reg', [1e-4, 1e-5]))))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1)))
    model.add(Dense(3, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-3, 1e-4])),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Initialize tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=80,
    factor=3,
    directory=os.path.join(base_dir, "models", "tuner_dir"),
    project_name='knee_cnn_tuning_cnn_new4'
)

# Early stopping
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# Tuning
tuner.search(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=100,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    callbacks=[early_stop]
)

# Best model
best_model = tuner.get_best_models(num_models=1)[0]
plot_model(best_model, to_file=os.path.join(base_dir, "models", "custom_cnn_architecture4.png"), show_shapes=True)

# Retrieve best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\nBest Hyperparameters:")
print(f"filters_1: {best_hps.get('filters_1')}")
print(f"filters_2: {best_hps.get('filters_2')}")
print(f"filters_3: {best_hps.get('filters_3')}")
if best_hps.get('use_block4'):
    print(f"filters_4: {best_hps.get('filters_4')}")
print(f"dense_units: {best_hps.get('dense_units')}")
print(f"dropout: {best_hps.get('dropout')}")
print(f"learning_rate: {best_hps.get('learning_rate')}")
print(f"l2_reg: {best_hps.get('l2_reg')}")

# Train final model
history = best_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    callbacks=[early_stop],
    verbose=1
)

# (rest of the evaluation and plots code follows as already present)
# Evaluate and visualize
train_loss, train_acc = best_model.evaluate(train_generator)
val_loss, val_acc = best_model.evaluate(val_generator)
test_loss, test_acc = best_model.evaluate(test_generator)
print(f"Train accuracy: {train_acc:.4f}")
print(f"Validation accuracy: {val_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# Save predictions
predictions = best_model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes
np.save(os.path.join(base_dir, "models", "custom_preds_cnn_new4.npy"), predictions)
np.save(os.path.join(base_dir, "models", "y_true_cnn_new4.npy"), y_true)

# Classification report
labels = list(test_generator.class_indices.keys())
report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
pd.DataFrame(report).transpose().to_csv(os.path.join(base_dir, "models", "classification_report_cnn_new4.csv"))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - Custom CNN")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "models", "confusion_matrix_cnn_new4.png"))
plt.close()

# ROC Curve
y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
fpr, tpr, roc_auc = {}, {}, {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
for i in range(3):
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curves - Custom CNN')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "models", "roc_curves_cnn_new4.png"))
plt.close()

# Combined accuracy and loss plot
best_epoch_acc = np.argmax(history.history['val_accuracy']) + 1
best_epoch_loss = np.argmin(history.history['val_loss']) + 1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.plot(history.history['loss'], label='Training loss', color='red')
ax1.plot(history.history['val_loss'], label='Validation loss', color='green')
ax1.axvline(best_epoch_loss - 1, color='blue', linestyle='--', label=f'Best Epoch (Loss): {best_epoch_loss}')
ax1.scatter(best_epoch_loss - 1, history.history['val_loss'][best_epoch_loss - 1], color='blue', s=50)
ax1.set_title('Loss over Epochs')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(history.history['accuracy'], label='Training Accuracy', color='red')
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='green')
ax2.axvline(best_epoch_acc - 1, color='blue', linestyle='--', label=f'Best Epoch (Acc): {best_epoch_acc}')
ax2.scatter(best_epoch_acc - 1, history.history['val_accuracy'][best_epoch_acc - 1], color='blue', s=50)
ax2.set_title('Accuracy over Epochs')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(base_dir, "models", "combined_accuracy_loss_plot_cnn_new4.png"))
plt.close()