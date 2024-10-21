import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import shutil
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

#trying to set up more GPU stuff, didnt work tho :(
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.65
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# GPU testing - trying not to cause my CPU a brain annurism (?)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("GPU is available and set for use.")
    except RuntimeError as e:
        print("Error setting up GPU:", e)
else:
    print("No GPU available. Using CPU.")

#naming datasets/paths - original dataset used for submission-wise work, limited_dataset, can put 50 files if i want, used to run code
#to check if it will run to completion.
original_dataset = 'Dataset 1\\Colorectal Cancer'
limited_dataset = 'path/'  # temp directory for the limited dataset


#Limited dataset created here, max_files_per_class = x, where x amount of files will be used per class
#Function to create a limited dataset
def create_limited_dataset(original_dir, limited_dir, max_files_per_class=50):
    if os.path.exists(limited_dir):
        shutil.rmtree(limited_dir)  # Remove if exists
    os.makedirs(limited_dir)

    for class_name in os.listdir(original_dir):
        class_dir = os.path.join(original_dir, class_name)
        if os.path.isdir(class_dir):
            limited_class_dir = os.path.join(limited_dir, class_name)
            os.makedirs(limited_class_dir)

            files = os.listdir(class_dir)[:max_files_per_class]
            for file_name in files:
                src = os.path.join(class_dir, file_name)
                dst = os.path.join(limited_class_dir, file_name)
                shutil.copy(src, dst)


#creates the limited dataset - used for debugging and testing to not take an hour of compiling -->

#create_limited_dataset(original_dataset, limited_dataset)

#---------------------------------------------------------------------------------------------------------
#                                          Hyper-parameters
#---------------------------------------------------------------------------------------------------------
img_size = (224, 224)
batch_size = 128
num_classes = 3
epochs = 40
learning_rate = 0.001

#image generator for loading image data and augmenting images, done to make training better, less stagnant behaviour
# should output better results for when we run testing in task 2
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

#change original_dataset to limited_dataset for using the limited dataset to train - [debugging]

train_generator = train_datagen.flow_from_directory(original_dataset, target_size=img_size, batch_size=batch_size,
                                                    class_mode='sparse'
                                                    )

#---------------------------------------------------------------------------------------
#                                                                                      |
#                                     CNN Management                                   |
#                                                                                      |
#---------------------------------------------------------------------------------------
#inputlayer
inputs = Input(shape=(224, 224, 3))
#ResNet50 model without top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)

#build
model = Model(inputs=inputs, outputs=outputs)
#freeze base layers in model
for layer in base_model.layers:
    layer.trainable = False
#compile
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#train using training generator
history = model.fit(train_generator, epochs=epochs, verbose=1)


#---------------------------------------------------------------------------------------------------------
#                                          Plotting
#---------------------------------------------------------------------------------------------------------
#plot accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
#plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()


plt.tight_layout()
plt.show()

#extract features using CNN encoder
encoder = Model(inputs=model.input, outputs=model.layers[-3].output)
features = encoder.predict(train_generator)

#get labels from training
labels = train_generator.classes

# get t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(features)

#plot t-SNE results
plt.figure(figsize=(8, 8))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis')
plt.colorbar(scatter)
plt.title('t-SNE Visualization of CNN Features')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()

# saving the encoder model to a file... Done using encoder.save and model.save just in case,
# unfamiliar with loading the model for future use so, to be safe done in different formats.
encoder.save('cnn_encoder_task1_lr0.001_fulldataset.h5')
model.save('my_model_task1_lr0.001_fulldaset.keras')
