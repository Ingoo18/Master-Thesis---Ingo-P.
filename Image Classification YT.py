import tensorflow as tf
import os
import cv2
import imghdr
from matplotlib import pyplot as plt
import numpy as np

# List available GPUs and set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Specify the directory containing your images
data_dir = r'C:\Users\ingop\Desktop\ICA_classification\_all_images' 

# List of image extensions to consider
image_exts = ['png']


#img = (cv2.imread(os.path.join(data_dir, 'study-ALG_sub-006_ses-Post_task-L2learning_properties_ic-2.png')))
#print(img.shape)

#show picture
#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

tf.data.Dataset
data = tf.keras.utils.image_dataset_from_directory(data_dir)
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()


###Preprocessing#### Scaling down the data for improved performance
data = data.map(lambda x,y: (x/255, y))
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()

#fig, ax = plt.subplots(ncols=4, figsize=(20,20))
#for idx, img in enumerate(batch[0][:4]):
 #   ax[idx].imshow(img)
  #  ax[idx].title.set_text(batch[1][idx])

len(data)

#testchange