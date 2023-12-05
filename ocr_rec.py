from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

def create_ocr_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

train_dir = 'path/to/training/directory'
test_dir = 'path/to/testing/directory'
img_height, img_width = 64, 64
batch_size = 32
num_classes = 26

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

ocr_model = create_ocr_model((img_height, img_width, 3), num_classes)

ocr_model.fit(train_generator, epochs=10, validation_data=test_generator)

test_loss, test_acc = ocr_model.evaluate(test_generator)
print('Test accuracy:', test_acc)
