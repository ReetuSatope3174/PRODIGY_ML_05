import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Hyperparameters (adjust as needed)
IMAGE_SIZE = (224, 224)  # Target image size for VGG16
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 3
NUM_FOLDS = 10 #Adjust number of folds for k-fold cross-validation

# Data paths (replace with your actual paths)
DATA_DIR = "C:/Users/reetu/Downloads/food-101.zip.zip"

# Load pre-trained VGG16 model (excluding final layers)
# Freeze pre-trained layers for fine-tuning
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=IMAGE_SIZE + (3,))
vgg_model.trainable = False

# Define data generators for training and validation sets
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Define custom layers for food recognition and add dropout for regularization
inputs = Input(shape=IMAGE_SIZE + (3,))
x = vgg_model(inputs)
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)  # Dropout layer to prevent overfitting
outputs = Dense(len(train_generator.classes), activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE)

# Model checkpoint to save the best model during training
model_checkpoint = ModelCheckpoint(filepath='best_model.h5', save_best_only=True, monitor='val_acc')

# Train the model
model.fit(train_generator,
          epochs=EPOCHS,
          validation_data=val_generator,
          callbacks=[early_stopping, model_checkpoint])

# Function to recognize food from an image (assuming your model is saved as 'best_model.h5')
def predict_food(image_path):
  # Load the saved model
  model = tf.keras.models.load_model('best_model.h5')

  img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
  img_array = tf.keras.preprocessing.image.img_to_array(img)
  img_batch = img_array.expand_dims(axis=0)  # Add batch dimension
  pred = model.predict(img_batch)
  predicted_class = train_generator.class_indices.get(np.argmax(pred[0]))
  return predicted_class

# Load a calorie database (replace with your data source)
calorie_data = {'pizza': 250, 'apple': 95, 'pasta': 500}  # Example data

# Function to estimate calorie content
def estimate_calories(food_class):
  if food_class in calorie_data:
    return calorie_data[food_class]
  else:
    return "Calorie information not available"

