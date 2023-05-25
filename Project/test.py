# %%
from tensorflow.keras.layers import SeparableConv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import ActivityRegularization
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,Input,Flatten
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle as pkl
import pandas as pd

# %%
tf. __version__

# # %%
# import os

# # Set the environment variable to only use the GPU (assuming you have one GPU)
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# %%
# tf.compat.v1.disable_eager_execution()

# %%
# !pip install tensorflow==2.10
# !pip install numpy pickle pandas scikit-learn


# %%
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# # %%
# import tensorflow as tf
# import numpy as np

# def create_dataset(num_samples, input_shape):
#     # Generate random input data and corresponding labels
#     X = np.random.rand(num_samples, *input_shape)
#     y = np.random.randint(0, 10, size=(num_samples,))

#     # Create a TensorFlow Dataset
#     dataset = tf.data.Dataset.from_tensor_slices((X, y))

#     return dataset

# # Example usage
# num_samples = 1000  # Specify the number of samples in your dataset
# input_shape = (100,)  # Specify the input shape of your data

# # Create the dataset
# # dataset = create_dataset(num_samples, input_shape)
# # Create the dataset
# dataset = create_dataset(num_samples, input_shape)

# # Split the dataset into training and validation sets
# train_dataset = dataset.take(800)
# val_dataset = dataset.skip(800)

# # Define the model
# input_layer = tf.keras.layers.Input(shape=input_shape)
# x = tf.keras.layers.Dense(128, activation='relu')(input_layer)
# x = tf.keras.layers.Dense(128*2, activation='relu')(x)
# x = tf.keras.layers.Dense(128, activation='relu')(x)
# x = tf.keras.layers.Dense(128, activation='relu')(x)
# x = tf.keras.layers.Dense(128*2, activation='relu')(x)
# x = tf.keras.layers.Dense(128*2, activation='relu')(x)
# x = tf.keras.layers.Dense(128*2, activation='relu')(x)

# x = tf.keras.layers.Dense(128, activation='relu')(x)
# x = tf.keras.layers.Dense(128, activation='relu')(x)
# x = tf.keras.layers.Dense(128, activation='relu')(x)
# x = tf.keras.layers.Dense(128, activation='relu')(x)
# x = tf.keras.layers.Dense(128, activation='relu')(x)

# x = tf.keras.layers.Dense(16, activation='relu')(x)
# output_layer = tf.keras.layers.Dense(10, activation='softmax')(x)
# model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
# model.summary()



# %%
import tensorflow as tf
import numpy as np

def create_sample_dataset(num_samples, input_shape, num_classes):
    # Generate random input images and corresponding labels
    X = np.random.rand(num_samples, *input_shape)
    y = np.random.randint(0, num_classes, size=(num_samples,))

    # Convert the data to TensorFlow tensors
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.int32)

    # Create a TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    return dataset

# Example usage
num_samples = 1000  # Specify the number of samples in your dataset
input_shape = (32, 32, 3)  # Specify the input shape of your data
num_classes = 10  # Specify the number of classes in your dataset

# Create the sample dataset
dataset = create_sample_dataset(num_samples, input_shape, num_classes)

# Split the dataset into training and validation sets
train_dataset = dataset.take(800)
val_dataset = dataset.skip(800)

# Define the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()



# %%
# Train the model on the training dataset
batch_size = 1
epochs = 10
model.fit(train_dataset.batch(batch_size), epochs=epochs, validation_data=val_dataset.batch(batch_size))

# # %%
# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Fit the model on the training dataset
# batch_size = 32
# epochs = 10
# model.fit(train_dataset.batch(batch_size), epochs=epochs, validation_data=val_dataset.batch(batch_size))

# %% [markdown]
# <h1>Loading Dataset and images</h1>

# %%
train_df = pd.read_csv('./train.csv',index_col='id')
test_df = pd.read_csv('./test.csv',index_col='id')

# %%
datagen=ImageDataGenerator(rescale=1.0/255.0,validation_split=0.25)

# %%
train_generator = datagen.flow_from_directory(
    directory="./imageset/train/",
    batch_size=1,
    seed=42,
    shuffle=True,
    class_mode="sparse",
    target_size=(64, 64),
    subset="training"
)
validation_generator = datagen.flow_from_directory(
    directory="./imageset/train/",
    batch_size=1,
    seed=42,
    shuffle=True,
    class_mode="sparse",
    target_size=(64, 64),
    subset="validation"  # Specify that this is the validation set
)

# %%
# train_images = pkl.load(open('./train_images.pickle','rb'))
# test_images = pkl.load(open('./test_images.pickle','rb'))

# %%
# train_images = train_images[:100]
# train_df = train_df.iloc[:100]

# %%
# train_images.shape

# %%
# X_train,X_val,y_train,y_val = train_test_split(train_images,train_df['label'],random_state=32)

# %%
# X_train.shape,X_val.shape,y_train.shape

# %%
# train_images = None

# %% [markdown]
# <h1>Model Architecture</h1>

# %%
def Module_1(filters,previous_layer):
    x = SeparableConv2D(filters,kernel_size=(3,3),padding='same',activation='relu')(previous_layer)
    x = SeparableConv2D(filters,kernel_size=(3,3),padding='same',activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3,3),padding='same')(x)
    return x

# %%
def Module_2(filters,previous_layer):
    x = SeparableConv2D(filters,kernel_size=(3,3),padding='same',activation='relu')(previous_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2),padding='same')(x)
    x = SeparableConv2D(filters,kernel_size=(3,3),padding='same',activation='relu')(x)
    return x

# %%
def Module_3(filters,previous_layer):
    x = SeparableConv2D(filters,kernel_size=(3,3),padding='same',dilation_rate=(1,1),activation='relu')(previous_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2),padding='same')(x)
    return x

# %%
def load_model(input_shape=(224,224,3),lr=0.01,classes=10):
    input_layer = Input(shape = input_shape,name='Input_layer')
    x = Module_1(filters=64,previous_layer=input_layer)
    one = Module_2(filters=64,previous_layer=x)
    two = Module_2(filters=32,previous_layer=x)
    concat = concatenate([one, two], axis=3)
    concat = ActivityRegularization(l2=0.001)(concat)
    x = Module_3(filters=16,previous_layer=x)
    x = Flatten()(x)
    x = Dense(128,activation='relu')(x)
    output = Dense(classes,activation='softmax')(x)
    model = Model(inputs=input_layer,outputs=output) 
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model
    

# %%
model = load_model(input_shape=train_generator.image_shape)
# model = load_model()

# %%
model.summary()

# %%
plot_model(model,show_shapes=True, show_layer_names=True)

# %%
# Define the checkpoint path and early stopping
checkpoint_path = "model_checkpoint_age_1.h5"
checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)
# Loads the weights
# model.load_weights(checkpoint_path)

# %%
history = model.fit(train_generator,steps_per_epoch=len(train_generator),
                    epochs=100,
                    validation_data=validation_generator,
                    validation_steps=len(validation_generator),
                    verbose=1,
                    callbacks=[checkpoint,early_stopping])

# %%
def plot_history(history,metrics):
    plt.plot(history.history[metrics])
    plt.plot(history.history['val_'+metrics])
    plt.title(f'model {metrics}')
    plt.ylabel(f'{metrics}')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

# %%
plot_history(history,metrics='accuracy')

# %%
plot_history(history,metrics='loss')



