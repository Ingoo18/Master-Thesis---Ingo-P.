import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load the model
model = tf.keras.models.load_model(r'directory')

# Display the model architecture
model.summary()

image_folder = r'directory'
image_files = [f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg'))]

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0  
    return img_array

preprocessed_images = [preprocess_image(os.path.join(image_folder, img_file)) for img_file in image_files]
preprocessed_images = np.vstack(preprocessed_images)

predictions = model.predict(preprocessed_images)
predicted_labels = np.argmax(predictions, axis=1)

data_dir = r'directory'
train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    image_size=(256, 256),
    batch_size=32
)

class_names = train_dataset.class_names


# Map predicted labels to class names
predicted_class_names = [class_names[label] for label in predicted_labels]

# Save predictions to a file
output_file = r'directory'

with open(output_file, 'w') as f:
    for img_file, class_name in zip(image_files, predicted_class_names):
        f.write(f"{img_file}: {class_name}\n")

print(f"Predictions saved to {output_file}")