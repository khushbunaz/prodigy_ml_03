import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Parameters
IMG_SIZE = (64, 64)  # Resize images to 64x64
TRAIN_DIR = "path_to_dataset/training_set"  # Replace with the path to the training_set
TEST_DIR = "path_to_dataset/test_set"       # Replace with the path to the test_set

def load_data(data_dir):
    """
    Load images and labels from the given directory.
    Assumes the directory contains subfolders for each class (e.g., 'cats' and 'dogs').
    """
    images = []
    labels = []
    for label, category in enumerate(["cats", "dogs"]):
        folder_path = os.path.join(data_dir, category)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                # Load and preprocess image
                img = load_img(img_path, target_size=IMG_SIZE)  # Load image and resize
                img_array = img_to_array(img).flatten()  # Flatten the image array for SVM
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels)

# Load training and testing data
print("Loading training data...")
X_train, y_train = load_data(TRAIN_DIR)
print("Loading testing data...")
X_test, y_test = load_data(TEST_DIR)

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Train the SVM model
print("Training SVM model...")
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)
print("Training complete.")

# Evaluate the model
print("Evaluating model...")
y_pred = svm_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test and visualize results
print("Visualizing predictions...")
indices = np.random.choice(len(X_test), 5, replace=False)
for idx in indices:
    img = X_test[idx].reshape(IMG_SIZE + (3,))
    true_label = "Cat" if y_test[idx] == 0 else "Dog"
    predicted_label = "Cat" if y_pred[idx] == 0 else "Dog"
    
    plt.imshow(img.reshape(IMG_SIZE[0], IMG_SIZE[1], 3))
    plt.title(f"True: {true_label}, Predicted: {predicted_label}")
    plt.axis("off")
    plt.show()
