import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(train_data_dir, validation_data_dir, img_height=224, img_width=224, batch_size=32):
    # Creating of ImageDataGenerators for data augmentation and normalization
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Loading and pre-processing of training and validation data
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    return train_generator, validation_generator