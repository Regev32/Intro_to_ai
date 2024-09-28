import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import pandas as pd

# Create "Results" directory if it doesn't exist
os.makedirs('Results', exist_ok=True)

# Load and preprocess the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train_full = x_train_full / 255.0
x_test = x_test / 255.0

# Reshape data to include a channel dimension
x_train_full = np.expand_dims(x_train_full, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Split the original training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=10000, random_state=42)

# Define the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with the validation set and 10 epochs
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), verbose=1)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Accuracy (percentages of correctly classified instances): {test_accuracy * 100:.2f}%")

# Custom Classification Report without F1-score and support
y_pred = np.argmax(model.predict(x_test), axis=-1)
report = classification_report(y_test, y_pred, output_dict=True)

# Create a DataFrame to display the report
df = pd.DataFrame(report).transpose()

# Select only the precision and recall columns, convert them to percentages, and rename the index to "digit"
df = df.loc[df.index.map(lambda x: x.isdigit()), ['precision', 'recall']] * 100
df.index.name = "Digit"

# Save the DataFrame to a CSV file in the "Results" folder
df.to_csv('Results/classification_report.csv', float_format='%.2f%%')

# Save the training history (accuracy, loss, val_accuracy, val_loss) to a CSV file in the "Results" folder
epochs_df = pd.DataFrame({
    'epoch': range(1, len(history.history['accuracy']) + 1),
    'accuracy': np.array(history.history['accuracy']) * 100,
    'loss': history.history['loss'],
    'val_accuracy': np.array(history.history['val_accuracy']) * 100,
    'val_loss': history.history['val_loss']
})

# Save the DataFrame to a CSV file in the "Results" folder
epochs_df.to_csv('Results/epochs_results.csv', index=False, float_format='%.4f')

# Plotting and saving training accuracy values
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(ticks=np.arange(0, 11, 1))  # Adjust x-axis to go up to 10
plt.legend(loc='upper left')
plt.savefig('Results/model_accuracy.png')  # Save the accuracy figure to "Results" folder
plt.show()

# Plotting and saving training loss values
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(ticks=np.arange(0, 11, 1))  # Adjust x-axis to go up to 10
plt.legend(loc='upper left')
plt.savefig('Results/model_loss.png')  # Save the loss figure to "Results" folder
plt.show()

# ROC Curve and AUC for combined classes
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y_pred_prob = model.predict(x_test)

# Calculate micro-average ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_pred_prob.ravel())
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, label=f'Micro-Average ROC Curve (AUC = {roc_auc * 100:.2f}%)')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Combined Classes')
plt.legend(loc='best')
plt.savefig('Results/roc_curve.png')  # Save the ROC curve figure to "Results" folder
plt.show()
