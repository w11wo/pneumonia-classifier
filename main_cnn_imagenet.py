import tensorflow as tf
from tensorflow.keras import layers, models
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255, 
    rotation_range=20, 
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    'chest_xray/train',
    batch_size=16,
    target_size=(128, 128),
    shuffle=True,
    class_mode='categorical'
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
val_generator = val_datagen.flow_from_directory(
    'chest_xray/val',
    batch_size=16,
    target_size=(128, 128),
    class_mode='categorical'
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
test_generator = test_datagen.flow_from_directory(
    'chest_xray/test',
    batch_size=16,
    target_size=(128, 128),
    class_mode='categorical'
)

base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(128, 128, 3)
)
base_model.trainable = False

model = base_model.output
model = tf.keras.layers.GlobalAveragePooling2D()(model)
model = tf.keras.layers.Dense(units=512, activation='relu')(model)
model = tf.keras.layers.Dropout(0.7)(model)
predictions = tf.keras.layers.Dense(units=2, activation='softmax')(model)
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(
    train_generator,
    epochs=15,
    verbose=1,
    validation_data=val_generator,
)

loss, test_acc = model.evaluate_generator(
    test_generator,
    steps=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=True,
    verbose=1
)

print(test_acc)

model.save('gpu_trained_model_local_augmented.h5')
