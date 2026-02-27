import numpy as np
import random

random.seed(42)
np.random.seed(42)

n_samples = 100

y_real = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])


y_predicted = []
for real_value in y_real:
    if real_value == 1:  # Rain
        # 80% chance of correctly predicting rain
        y_predicted.append(1 if random.random() < 0.8 else 0)
    else:  # No rain
        # 85% chance of correctly predicting no rain
        y_predicted.append(0 if random.random() < 0.85 else 1)

y_predicted = np.array(y_predicted)


for i in range(20):
    predicted = "Rain" if y_predicted[i] == 1 else "No Rain"
    real = "Rained" if y_real[i] == 1 else "Did Not Rain"
    
    if y_predicted[i] == 1 and y_real[i] == 1:
        situation = "True Positive (TP)"
    elif y_predicted[i] == 1 and y_real[i] == 0:
        situation = "False Positive (FP)"
    elif y_predicted[i] == 0 and y_real[i] == 1:
        situation = "False Negative (FN)"
    else:
        situation = "True Negative (TN)"
    
    print(f"{predicted:12} | {real:11} | {situation}")



# Calculate the components of the matrix
VP = np.sum((y_predicted == 1) & (y_real == 1))  # True Positives
VN = np.sum((y_predicted == 0) & (y_real == 0))  # True Negatives
FP = np.sum((y_predicted == 1) & (y_real == 0))  # False Positives
FN = np.sum((y_predicted == 0) & (y_real == 1))  # False Negatives



N = VP + VN + FP + FN  # Total samples

# Sensitivity (Recall) = VP / (VP + FN)
# Ability to correctly identify positives
sensitivity = VP / (VP + FN) if (VP + FN) > 0 else 0

# Specificity = VN / (FP + VN)
# Ability to correctly identify negatives
specificity = VN / (FP + VN) if (FP + VN) > 0 else 0

# Accuracy = (VP + VN) / N
# Overall proportion of correct predictions
accuracy = (VP + VN) / N if N > 0 else 0

# Precision = VP / (VP + FP)
# Of all positive predictions, how many were correct
precision = VP / (VP + FP) if (VP + FP) > 0 else 0

# F-score = 2 × (Precision × Sensitivity) / (Precision + Sensitivity)
# Harmonic mean of precision and sensitivity
f_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

print("\n" + "="*60)
print("SUMMARY OF METRICS")
print("="*60)
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F-Score: {f_score:.4f}")
print("="*60)
