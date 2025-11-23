import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import RejectOptionClassification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- CONFIG ---
protected_attribute = 'race'
privileged_groups = [{'race': 1}]
unprivileged_groups = [{'race': 0}]

# --- LOAD ---
compas = CompasDataset()  # uses built-in loader
# Inspect dataset metadata
print("Dataset shape:", compas.features.shape, "labels shape:", compas.labels.shape)
print("Protected attribute names:", compas.protected_attribute_names)
print("Privileged label (default):", compas.favorable_label, "Unfavorable:", compas.unfavorable_label)

# --- METRICS BEFORE MITIGATION ---
metric_orig = BinaryLabelDatasetMetric(compas,
                                      unprivileged_groups=unprivileged_groups,
                                      privileged_groups=privileged_groups)
print("Original mean difference (diff in selection):", metric_orig.mean_difference())

# Prepare a classifier (sklearn) pipeline
X = compas.features
y = compas.labels.ravel()

scaler = StandardScaler()
clf = LogisticRegression(solver='liblinear')

# Train-test split (simple)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, np.arange(len(y)), test_size=0.3, random_state=42, stratify=y)

# Build and train model
pipe = Pipeline([('scaler', scaler), ('clf', clf)])
pipe.fit(X_train, y_train)

# Wrap predictions back into BinaryLabelDataset for metrics
from aif360.datasets import BinaryLabelDataset
test_bld = BinaryLabelDataset(favorable_label=compas.favorable_label,
                              unfavorable_label=compas.unfavorable_label,
                              df=pd.DataFrame(np.hstack([X_test, y_test.reshape(-1,1)]),
                                              columns=list(compas.feature_names) + ['label']),
                              label_names=['label'],
                              protected_attribute_names=compas.protected_attribute_names)

# Generate predicted labels
y_pred = pipe.predict(X_test)
pred_bld = test_bld.copy()
pred_bld.labels = y_pred.reshape(-1,1)

# Compute classification metrics by group
classified_metric = ClassificationMetric(test_bld, pred_bld,
                                         unprivileged_groups=unprivileged_groups,
                                         privileged_groups=privileged_groups)

print("=== BEFORE MITIGATION ===")
print("False positive rate (unprivileged):", classified_metric.false_positive_rate(unprivileged=True))
print("False positive rate (privileged):", classified_metric.false_positive_rate(privileged=True))
print("False negative rate (unprivileged):", classified_metric.false_negative_rate(unprivileged=True))
print("False negative rate (privileged):", classified_metric.false_negative_rate(privileged=True))
print("Disparate impact:", classified_metric.disparate_impact())

# Visualize FPR/FNR disparity
groups = ['unprivileged','privileged']
fpr = [classified_metric.false_positive_rate(unprivileged=True),
       classified_metric.false_positive_rate(privileged=True)]
fnr = [classified_metric.false_negative_rate(unprivileged=True),
       classified_metric.false_negative_rate(privileged=True)]

fig, ax = plt.subplots(1,2, figsize=(12,4))
ax[0].bar(groups, fpr)
ax[0].set_title('False Positive Rate by Group')
ax[0].set_ylim(0,1)
ax[1].bar(groups, fnr)
ax[1].set_title('False Negative Rate by Group')
ax[1].set_ylim(0,1)
plt.tight_layout()
plt.savefig('fpr_fnr_by_group.png')
print("Saved 'fpr_fnr_by_group.png'")

# --- MITIGATION: Reweighing (preprocessing) ---
rw = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
rw.fit(compas)
compas_transf = rw.transform(compas)

# Train on reweighed dataset (use sample weights)
X_r = compas_transf.features
y_r = compas_transf.labels.ravel()
w_r = compas_transf.instance_weights

X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(X_r, y_r, w_r, test_size=0.3, random_state=42, stratify=y_r)

pipe2 = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(solver='liblinear'))])
pipe2.fit(X_tr, y_tr, sample_weight=w_tr)

# Evaluate
y_pred2 = pipe2.predict(X_val)
# rebuild BinaryLabelDataset for val
val_df = pd.DataFrame(np.hstack([X_val, y_val.reshape(-1,1)]),
                      columns=list(compas.feature_names) + ['label'])
val_bld = BinaryLabelDataset(favorable_label=compas.favorable_label,
                             unfavorable_label=compas.unfavorable_label,
                             df=val_df,
                             label_names=['label'],
                             protected_attribute_names=compas.protected_attribute_names)

pred_bld2 = val_bld.copy()
pred_bld2.labels = y_pred2.reshape(-1,1)

metric_post = ClassificationMetric(val_bld, pred_bld2,
                                   unprivileged_groups=unprivileged_groups,
                                   privileged_groups=privileged_groups)

print("=== AFTER REWEIGHING MITIGATION ===")
print("FPR (unpriv):", metric_post.false_positive_rate(unprivileged=True))
print("FPR (priv):", metric_post.false_positive_rate(privileged=True))
print("FNR (unpriv):", metric_post.false_negative_rate(unprivileged=True))
print("FNR (priv):", metric_post.false_negative_rate(privileged=True))
print("Disparate impact:", metric_post.disparate_impact())

# Save figures for the post-mitigation comparison
fpr_post = [metric_post.false_positive_rate(unprivileged=True),
            metric_post.false_positive_rate(privileged=True)]
fnr_post = [metric_post.false_negative_rate(unprivileged=True),
            metric_post.false_negative_rate(privileged=True)]

fig, ax = plt.subplots(1,2, figsize=(12,4))
ax[0].bar(groups, fpr_post)
ax[0].set_title('FPR by Group (post-mitigation)')
ax[1].bar(groups, fnr_post)
ax[1].set_title('FNR by Group (post-mitigation)')
plt.tight_layout()
plt.savefig('fpr_fnr_by_group_post.png')
print("Saved 'fpr_fnr_by_group_post.png'")

# End
print("Audit complete. Inspect PNGs and printed metrics.")