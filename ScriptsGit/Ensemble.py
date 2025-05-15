# === Imports ===
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

# === Setup ===
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

base_dir = "/home/u243945/Knee Osteoarthritis Classification/Data_thesis"
models_dir = os.path.join(base_dir, "models")
os.makedirs(os.path.join(models_dir, "ensemble_new"), exist_ok=True)

# === Load Predictions ===
custom_preds = np.load(os.path.join(models_dir, "custom_preds_cnn_new4.npy"))
resnet_preds = np.load(os.path.join(models_dir, "resnet50_predictions_sat_1.npy"))
densenet_preds = np.load(os.path.join(models_dir, "improved_densenet_preds_sat_2.npy"))
y_true = np.load(os.path.join(models_dir, "y_true_cnn_new4.npy"))
labels = ['normal', 'osteopenia', 'osteoporosis']

# === Ensure Same Length ===
min_len = min(len(custom_preds), len(resnet_preds), len(densenet_preds), len(y_true))
custom_preds = custom_preds[:min_len]
resnet_preds = resnet_preds[:min_len]
densenet_preds = densenet_preds[:min_len]
y_true = y_true[:min_len]

# === Save Cleaned Predictions ===
np.save(os.path.join(models_dir, "ensemble_new", "custom_preds_cnn_new4.npy"), custom_preds)

# === Evaluation Function ===
def evaluate_ensemble(preds, method_name):
    y_pred = np.argmax(preds, axis=1)
    acc = np.mean(y_pred == y_true)
    print(f"{method_name} Accuracy: {acc:.4f}")

    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(models_dir, "ensemble_new", f"classification_report_new_{method_name}.csv"))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {method_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, "ensemble_new", f"confusion_matrix_new_{method_name}.png"), dpi=300)
    plt.close()

    y_true_bin = label_binarize(y_true, classes=list(range(len(labels))))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(labels)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i in range(len(labels)):
        plt.plot(fpr[i], tpr[i], lw=2, label=f"{labels[i]} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {method_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, "ensemble_new", f"roc_curve_new_{method_name}.png"), dpi=300)
    plt.close()

    # === Precision, Recall, F1 ===
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    prf_df = pd.DataFrame({
        'Label': labels,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    prf_df.to_csv(os.path.join(models_dir, "ensemble_new", f"precision_recall_f1_new{method_name}.csv"), index=False)

    plt.figure(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.25
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1-Score')
    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel('Score')
    plt.title(f'Precision, Recall, F1-Score - {method_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, "ensemble_new", f"precision_recall_f1_plot_new{method_name}.png"), dpi=300)
    plt.close()

    return acc, report

# === Ensemble Methods ===
average_preds = (custom_preds + resnet_preds + densenet_preds) / 3
weighted_preds = 0.24 * custom_preds + 0.38 * resnet_preds + 0.38 * densenet_preds

# Max Voting
max_ensemble = np.zeros_like(custom_preds)
for i in range(len(custom_preds)):
    combined = np.vstack([custom_preds[i], resnet_preds[i], densenet_preds[i]])
    max_indices = np.argmax(combined, axis=0)
    for j in range(len(max_indices)):
        max_ensemble[i, j] = combined[max_indices[j], j]
    max_ensemble[i] = max_ensemble[i] / np.sum(max_ensemble[i])

# Stacking
combined_features = np.hstack([custom_preds, resnet_preds, densenet_preds])
stk_preds = np.zeros_like(custom_preds)
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for train_idx, test_idx in kf.split(combined_features):
    meta_model = LogisticRegression(max_iter=1000, random_state=SEED)
    meta_model.fit(combined_features[train_idx], y_true[train_idx])
    stk_preds[test_idx] = meta_model.predict_proba(combined_features[test_idx])

# Majority Voting with Confidence
cls_custom = np.argmax(custom_preds, axis=1)
cls_resnet = np.argmax(resnet_preds, axis=1)
cls_densenet = np.argmax(densenet_preds, axis=1)
class_votes = np.vstack([cls_custom, cls_resnet, cls_densenet])
majority_vote = stats.mode(class_votes, axis=0, keepdims=True)[0].squeeze()
majority_confidence = np.mean(class_votes == majority_vote, axis=0)

# Save confidence scores
np.save(os.path.join(models_dir, "ensemble_new", "majority_confidence_scores_new.npy"), majority_confidence)
majority_preds = np.zeros_like(custom_preds)
for i, vote in enumerate(majority_vote):
    majority_preds[i, vote] = 1

# === Evaluate ===
custom_acc, _ = evaluate_ensemble(custom_preds, "custom_cnn")
resnet_acc, _ = evaluate_ensemble(resnet_preds, "resnet50")
densenet_acc, _ = evaluate_ensemble(densenet_preds, "densenet121")
avg_acc, _ = evaluate_ensemble(average_preds, "average_ensemble")
weighted_acc, _ = evaluate_ensemble(weighted_preds, "weighted_ensemble")
max_acc, _ = evaluate_ensemble(max_ensemble, "max_voting")
stk_acc, _ = evaluate_ensemble(stk_preds, "stacking")
majority_acc, _ = evaluate_ensemble(majority_preds, "majority_voting")

# === Summary Plot ===
methods = [
    "Custom CNN", 
    "ResNet50", 
    "DenseNet121", 
    "Average Ensemble", 
    "Weighted Ensemble", 
    "Max Voting", 
    "Stacking", 
    "Majority Voting"
]
accuracies = [
    custom_acc, 
    resnet_acc, 
    densenet_acc, 
    avg_acc, 
    weighted_acc, 
    max_acc, 
    stk_acc, 
    majority_acc
]
comparison_df = pd.DataFrame({
    'Method': methods,
    'Accuracy': accuracies
}).sort_values(by='Accuracy', ascending=False)
comparison_df.to_csv(os.path.join(models_dir, "ensemble_new", "ensemble_comparison_new.csv"), index=False)

plt.figure(figsize=(12, 8))
ax = sns.barplot(x='Accuracy', y='Method', data=comparison_df, palette='viridis')
plt.title('Ensemble Methods Comparison')
plt.xlim(min(accuracies) - 0.05, max(accuracies) + 0.05)
plt.grid(axis='x', alpha=0.3)
for i, acc in enumerate(comparison_df['Accuracy']):
    ax.text(acc + 0.005, i, f'{acc:.4f}', va='center')
plt.tight_layout()
plt.savefig(os.path.join(models_dir, "ensemble_new", "ensemble_comparison_new.png"), dpi=300)
plt.close()

# Markdown Summary
best_method = comparison_df.iloc[0]['Method']
best_accuracy = comparison_df.iloc[0]['Accuracy']
with open(os.path.join(models_dir, "ensemble_new", "ensemble_summary_new.md"), 'w') as f:
    f.write("# Knee Osteoarthritis Classification - Ensemble Results\n\n")
    f.write("## Best Performing Model\n")
    f.write(f"**{best_method}** with accuracy: **{best_accuracy:.4f}**\n\n")
    f.write("## Accuracy Comparison\n")
    for _, row in comparison_df.iterrows():
        f.write(f"- {row['Method']}: {row['Accuracy']:.4f}\n")
    f.write("\n## Ensemble Methods\n")
    f.write("1. **Average Ensemble**: Simple averaging of probabilities\n")
    f.write("2. **Weighted Ensemble**: Weighted average based on model accuracy\n")
    f.write("3. **Max Voting**: Per-class confidence voting\n")
    f.write("4. **Stacking**: Meta-learner on combined model outputs\n")
    f.write("5. **Majority Voting**: Hard voting among top predicted classes\n")
    f.write("\n## Observations\n")
    if best_method.lower().endswith("ensemble"):
        f.write("The ensemble method outperformed individual models, indicating effective fusion.\n")
    else:
        f.write("An individual model outperformed ensembles, suggesting possible redundancy.\n")
    f.write("\n## Future Improvements\n")
    f.write("- Optimize ensemble weights\n")
    f.write("- Use more diverse base models\n")
    f.write("- Apply test-time augmentation\n")
    f.write("- Try neural net meta-learners for stacking\n")
