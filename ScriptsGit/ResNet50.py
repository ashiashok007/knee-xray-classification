# ======== Imports ========
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pandas as pd
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.regularizers import l2

# ======== Reproducibility ========
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ======== Paths ========
base_dir = "/home/u243945/Knee Osteoarthritis Classification/Data_thesis"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")
models_dir = os.path.join(base_dir, "models")
os.makedirs(models_dir, exist_ok=True)

# ======== Data Generators ========
# Use consistent preprocessing across all datasets
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.15,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)
test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=16,
    class_mode='categorical', shuffle=True, seed=SEED
)
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(224, 224), batch_size=16,
    class_mode='categorical', shuffle=False, seed=SEED
)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=16,
    class_mode='categorical', shuffle=False
)
labels = list(test_generator.class_indices.keys())

# Calculate steps per epoch for proper training
train_steps = train_generator.samples // train_generator.batch_size + 1
val_steps = val_generator.samples // val_generator.batch_size + 1
test_steps = test_generator.samples // test_generator.batch_size + 1

# ======== Build Enhanced ResNet50 Model ========
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
base_model.trainable = False

# Enhanced model architecture with dual pooling
x = base_model.output
avg_pool = GlobalAveragePooling2D()(x)
max_pool = GlobalMaxPooling2D()(x)
x = Concatenate()([avg_pool, max_pool])
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
output = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# ======== Callbacks ========
early_stop = EarlyStopping(
    monitor='val_accuracy', 
    patience=10, 
    restore_best_weights=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=5, 
    min_lr=1e-6,
    verbose=1
)
checkpoint = ModelCheckpoint(
    os.path.join(models_dir, 'resnet50_checkpoint_sat_1.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ======== Stage 1: Train only the top layers ========
model.compile(
    optimizer=Adam(learning_rate=1e-4), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

print("Stage 1: Training top layers...")
history1 = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    validation_data=val_generator,
    validation_steps=val_steps,
    epochs=30,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# ======== Stage 2: Fine-tune the last ResNet block ========
print("Stage 2: Fine-tuning the last ResNet block...")
# Find the last conv block in ResNet50
for i, layer in enumerate(base_model.layers):
    if 'conv5_block3' in layer.name:
        last_resnet_block_idx = i
        break

# Freeze all layers except the last ResNet block
for layer in base_model.layers[:last_resnet_block_idx]:
    layer.trainable = False
for layer in base_model.layers[last_resnet_block_idx:]:
    layer.trainable = True

# Recompile with reduced learning rate
model.compile(
    optimizer=Adam(learning_rate=5e-5), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

history2 = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    validation_data=val_generator,
    validation_steps=val_steps,
    epochs=40,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# ======== Stage 3: Fine-tune more layers ========
print("Stage 3: Fine-tuning more layers...")
# Find the fourth block in ResNet50
for i, layer in enumerate(base_model.layers):
    if 'conv4_block1' in layer.name:
        fourth_block_idx = i
        break

# Freeze all layers except the last two ResNet blocks
for layer in base_model.layers[:fourth_block_idx]:
    layer.trainable = False
for layer in base_model.layers[fourth_block_idx:]:
    layer.trainable = True

# Recompile with even smaller learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

history3 = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    validation_data=val_generator,
    validation_steps=val_steps,
    epochs=50,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# ======== Save Final Model ========
model_save_path = os.path.join(models_dir, "resnet50_knee_classification_sat_1.h5")
model.save(model_save_path)
print(f"ResNet50 model saved at: {model_save_path}")

# ======== Evaluate and Predict ========
train_loss, train_acc = model.evaluate(train_generator, steps=train_steps, verbose=1)
val_loss, val_acc = model.evaluate(val_generator, steps=val_steps, verbose=1)
test_loss, test_acc = model.evaluate(test_generator, steps=test_steps, verbose=1)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Predictions
predictions = model.predict(test_generator, steps=test_steps, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes[:len(y_pred)]  # Ensure same length as predictions

# Save predictions
np.save(os.path.join(models_dir, "resnet50_predictions_sat_1.npy"), predictions)

# ======== Classification Report ========
report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(models_dir, "classification_report_resnet50_sat_1.csv"))


# ======== Confusion Matrix ========
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - ResNet50")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(models_dir, "confusion_matrix_resnet50_sat_1.png"), dpi=300)
plt.close()


# ======== ROC Curve ========
y_true_bin = label_binarize(y_true, classes=list(range(len(labels))))
fpr, tpr, roc_auc = {}, {}, {}
for i in range(len(labels)):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
colors = ['blue', 'green', 'red']
for i, color in zip(range(len(labels)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f"{labels[i]} (AUC = {roc_auc[i]:.3f})")
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('ROC Curves - ResNet50')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(models_dir, "roc_curve_resnet50_sat_1.png"), dpi=300)
plt.close()


# ======== Accuracy and Loss Plots ========
# Combine all training histories
history_combined = {
    'accuracy': (history1.history['accuracy'] + 
                 history2.history['accuracy'] + 
                 history3.history['accuracy']),
    'val_accuracy': (history1.history['val_accuracy'] + 
                     history2.history['val_accuracy'] + 
                     history3.history['val_accuracy']),
    'loss': (history1.history['loss'] + 
             history2.history['loss'] + 
             history3.history['loss']),
    'val_loss': (history1.history['val_loss'] + 
                 history2.history['val_loss'] + 
                 history3.history['val_loss']),
}

# Add stage separation lines
stage1_end = len(history1.history['accuracy'])
stage2_end = stage1_end + len(history2.history['accuracy'])

# Accuracy Plot
plt.figure(figsize=(12, 6))
plt.plot(history_combined['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history_combined['val_accuracy'], label='Validation Accuracy', color='green')
plt.axvline(stage1_end, color='r', linestyle='--', label='End of Stage 1')
plt.axvline(stage2_end, color='purple', linestyle='--', label='End of Stage 2')
best_epoch = np.argmax(history_combined['val_accuracy']) + 1
plt.axvline(best_epoch-1, color='orange', linestyle='-', label=f'Best Epoch ({best_epoch})')
plt.title('Training & Validation Accuracy - ResNet50')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(models_dir, "accuracy_plot_resnet50_sat_1.png"), dpi=300)
plt.close()

# Loss Plot
plt.figure(figsize=(12, 6))
plt.plot(history_combined['loss'], label='Train Loss', color='blue')
plt.plot(history_combined['val_loss'], label='Validation Loss', color='green')
plt.axvline(stage1_end, color='r', linestyle='--', label='End of Stage 1')
plt.axvline(stage2_end, color='purple', linestyle='--', label='End of Stage 2')
plt.axvline(best_epoch-1, color='orange', linestyle='-', label=f'Best Epoch ({best_epoch})')
plt.title('Training & Validation Loss - ResNet50')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(models_dir, "loss_plot_resnet50_sat_1.png"), dpi=300)
plt.close()

