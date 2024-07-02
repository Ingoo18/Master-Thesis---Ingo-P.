import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import seaborn as sns

#Directory of the images
data_dir = r'directory'

# Loading of the dataset
data = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    batch_size=None, 
    image_size=(256, 256),
    shuffle=True
)

##Check Image Size
# for images, labels in data.take(1):
#     print(f'Batch shape: {images.shape}')
#     print(f'First image shape in batch: {images[0].shape}')

#Check number of classes
items = os.listdir(data_dir)
class_amt = sum(os.path.isdir(os.path.join(data_dir, item)) for item in items)

#Check Class Names
class_names = data.class_names
print(class_names)


###Preprocessing###
    
#Scale down the data
data = data.map(lambda x, y: (x / 255.0, y))

# Conversion of the dataset to a NumPy array
def dataset_to_numpy(data):
    images, labels = [], []
    for image, label in data:
        images.append(image.numpy())
        labels.append(label.numpy())
    return np.array(images), np.array(labels)

X, y = dataset_to_numpy(data)

# Check class distribution
unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))

# Split the data into train and temp sets
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)

# Split the temp set into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

# One-hot encode the classes
y_train = tf.keras.utils.to_categorical(y_train, num_classes=class_amt)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=class_amt)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=class_amt)

# Convert back to TensorFlow datasets
train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Batch Size 
batch_size = 48

#Prefetch the datasets
train = train.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
val = val.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
test = test.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


###CNN-Model###

model = Sequential()
model.add(Conv2D(16, (1, 1), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (1, 1), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (1, 1), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(class_amt, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Model Training
history = model.fit(train, epochs=8, validation_data=val)

# Evaluation of the model on the test data
test_loss, test_accuracy = model.evaluate(test)
print(f"Test Accuracy: {test_accuracy}")

# Get predictions and true labels
y_pred_prob = model.predict(test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)


# Print classification report
print(classification_report(y_true, y_pred, target_names=class_names))

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

#Print confusion matrix
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# Plot Model Accuracy
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot Model Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

#Save the Trained Model
#model.save(r'directory')