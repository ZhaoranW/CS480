# originally run in Kaggle


import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import img_to_array

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load data
train_df = pd.read_csv('/kaggle/input/cs-480-2024-spring/data/train.csv')
train_df = train_df.sample(frac=0.1, random_state=42)
target_columns = ['X4_mean', 'X11_mean', 'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean']
ancillary_columns = train_df.columns[1:164]

# Preprocess ancillary data
scaler = StandardScaler()
train_ancillary = scaler.fit_transform(train_df[ancillary_columns])

# Targets
train_targets = train_df[target_columns].values
print('train_targets', train_targets.shape)


# Initialize scalers for each target
scalers = [StandardScaler() for _ in range(train_targets.shape[1])]
# Standardize the training targets
standardized_train_targets = np.zeros_like(train_targets)
for i in range(train_targets.shape[1]):
    standardized_train_targets[:, i] = scalers[i].fit_transform(train_targets[:, i].reshape(-1, 1)).flatten()
print('standardized_train_targets', standardized_train_targets.shape)

train_targets = standardized_train_targets

# Custom data generator class
class DataGenerator(Sequence):
    def __init__(self, df, indices, img_dir, ancillary_data, targets, batch_size=32, shuffle=True):
        self.df = df.iloc[indices].reset_index(drop=True)
        self.img_dir = img_dir
        self.ancillary_data = ancillary_data[indices]
        self.targets = targets[indices]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]
        return self.__data_generation(batch_df, batch_indices)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_df, batch_indices):
        batch_images = np.array([self.__load_image(file_path) for file_path in batch_df['id']])
        batch_ancillary = self.ancillary_data[batch_indices]
        batch_targets = {
            'output1': self.targets[batch_indices, 0],
            'output2': self.targets[batch_indices, 1],
            'output3': self.targets[batch_indices, 2],
            'output4': self.targets[batch_indices, 3],
            'output5': self.targets[batch_indices, 4],
            'output6': self.targets[batch_indices, 5]
        }
        return (batch_images, batch_ancillary), batch_targets

    def __load_image(self, file_path):
        img = cv2.imread(f"/kaggle/input/cs-480-2024-spring/data/train_images/{file_path}.jpeg")
        img = cv2.resize(img, (128, 128))
        img = img.astype('float32') / 255.0
        img_array = img_to_array(img)
        return img_array

# Ensure correct train-validation split
train_idx, val_idx = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)



def build_model():
    # Image input
    img_input = Input(shape=(128, 128, 3))
    base_model = ResNet50(include_top=False, input_tensor=img_input, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Ancillary input
    ancillary_input = Input(shape=(train_ancillary.shape[1],))
    y = Dense(128, activation='relu')(ancillary_input)
    y = Dropout(0.3)(y)
    
    # Combine features
    combined = Concatenate()([x, y])
    z = Dense(128, activation='relu')(combined)
    z = Dropout(0.3)(z)
    
    # Separate output layers
    output1 = Dense(1, activation='linear', name='output1')(z)
    output2 = Dense(1, activation='linear', name='output2')(z)
    output3 = Dense(1, activation='linear', name='output3')(z)
    output4 = Dense(1, activation='linear', name='output4')(z)
    output5 = Dense(1, activation='linear', name='output5')(z)
    output6 = Dense(1, activation='linear', name='output6')(z)
    
    model = Model(inputs=[img_input, ancillary_input], outputs=[output1, output2, output3, output4, output5, output6])
    model.compile(optimizer=Adam(learning_rate=0.002), 
                  loss={
                      'output1': 'mean_squared_error', 
                      'output2': 'mean_squared_error', 
                      'output3': 'mean_squared_error', 
                      'output4': 'mean_squared_error', 
                      'output5': 'mean_squared_error', 
                      'output6': 'mean_squared_error'
                  })
    return model

model = build_model()




# Define Data Generators
train_generator = DataGenerator(train_df, train_idx, '/kaggle/input/cs-480-2024-spring/data/train_images', train_ancillary, train_targets)
val_generator = DataGenerator(train_df, val_idx, '/kaggle/input/cs-480-2024-spring/data/train_images', train_ancillary, train_targets, shuffle=False)

# Convert generators to tf.data.Dataset
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_signature=(
        (
            tf.TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, train_ancillary.shape[1]), dtype=tf.float32)
        ),
        {
            'output1': tf.TensorSpec(shape=(None,), dtype=tf.float32),
            'output2': tf.TensorSpec(shape=(None,), dtype=tf.float32),
            'output3': tf.TensorSpec(shape=(None,), dtype=tf.float32),
            'output4': tf.TensorSpec(shape=(None,), dtype=tf.float32),
            'output5': tf.TensorSpec(shape=(None,), dtype=tf.float32),
            'output6': tf.TensorSpec(shape=(None,), dtype=tf.float32)
        }
    )
)

val_dataset = tf.data.Dataset.from_generator(
    lambda: val_generator,
    output_signature=(
        (
            tf.TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, train_ancillary.shape[1]), dtype=tf.float32)
        ),
        {
            'output1': tf.TensorSpec(shape=(None,), dtype=tf.float32),
            'output2': tf.TensorSpec(shape=(None,), dtype=tf.float32),
            'output3': tf.TensorSpec(shape=(None,), dtype=tf.float32),
            'output4': tf.TensorSpec(shape=(None,), dtype=tf.float32),
            'output5': tf.TensorSpec(shape=(None,), dtype=tf.float32),
            'output6': tf.TensorSpec(shape=(None,), dtype=tf.float32)
        }
    )
)

# Train model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)


# Load test data
test_df = pd.read_csv('/kaggle/input/cs-480-2024-spring/data/test.csv')
# test_df = test_df.sample(frac=0.1, random_state=42)
test_img_dir = '/kaggle/input/cs-480-2024-spring/data/test_images'
target_columns = ['X4_mean', 'X11_mean', 'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean']
ancillary_columns = test_df.columns[1:164]

test_ancillary = scaler.fit_transform(test_df[ancillary_columns])
print(test_ancillary.shape)


# Custom data generator class for test data
class TestDataGenerator(Sequence):
    def __init__(self, df, img_dir, ancillary_data, batch_size=6391):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.ancillary_data = ancillary_data
        self.batch_size = batch_size
        self.indices = np.arange(len(self.df))

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]
        return self.__data_generation(batch_df, batch_indices)

    def __data_generation(self, batch_df, batch_indices):
        batch_images = np.array([self.__load_image(file_path) for file_path in batch_df['id']])
        batch_ancillary = self.ancillary_data[batch_indices]
        batch_ids = batch_df['id'].astype(str).values
        return (batch_images, batch_ancillary), batch_ids

    def __load_image(self, file_path):
        img = cv2.imread(f"/kaggle/input/cs-480-2024-spring/data/test_images/{file_path}.jpeg")
        img = cv2.resize(img, (128, 128))
        img = img.astype('float32') / 255.0
        img_array = img_to_array(img)
        return img_array

# Create the test data generator
test_generator = TestDataGenerator(test_df, test_img_dir, test_ancillary)

# Convert the test generator to tf.data.Dataset
test_dataset = tf.data.Dataset.from_generator(
    lambda: test_generator,
    output_signature=(
        (
            tf.TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, test_ancillary.shape[1]), dtype=tf.float32)
        ),
        tf.TensorSpec(shape=(None,), dtype=tf.string)
    )
)

# test_dataset = test_dataset.unbatch().batch(32)
print(test_dataset)

# Make predictions and capture the IDs
ids = []
predictions = []

for (data, batch_ids) in test_dataset:
    preds = model.predict(data)
    ids.extend([id.decode('utf-8') if isinstance(id, bytes) else id for id in batch_ids.numpy()])  # Decode if byte string
    predictions.extend(preds)


# Convert predictions to a NumPy array
predictions_array = np.array(predictions)

# Convert predictions to a NumPy array
predictions_array = np.array(predictions)
predictions_array = predictions_array.reshape(6, 6391)

# Check the shape of the predictions array
print(predictions_array.shape)

destandardized_predictions = {
    'X4_mean': scalers[0].inverse_transform(predictions[0]),
    'X11_mean': scalers[1].inverse_transform(predictions[1]),
    'X18_mean': scalers[2].inverse_transform(predictions[2]),
    'X26_mean': scalers[3].inverse_transform(predictions[3]),
    'X50_mean': scalers[4].inverse_transform(predictions[4]),
    'X3112_mean': scalers[5].inverse_transform(predictions[5])
}

print(destandardized_predictions) 

df = pd.DataFrame({
    'X4_mean': destandardized_predictions['X4_mean'].flatten(),
    'X11_mean': destandardized_predictions['X11_mean'].flatten(),
    'X18_mean': destandardized_predictions['X18_mean'].flatten(),
    'X26_mean': destandardized_predictions['X26_mean'].flatten(),
    'X50_mean': destandardized_predictions['X50_mean'].flatten(),
    'X3112_mean': destandardized_predictions['X3112_mean'].flatten()
})


# # Convert predictions to a DataFrame
df.insert(0, 'id', ids)

# # Save the predictions to a CSV file
df.to_csv('destandardized_predictions.csv', index=False)
print(df)

# Save the predictions to a CSV file
df.to_csv('/kaggle/working/prediction.csv', index=False)

# Use Kaggle's file system to download the file
from IPython.display import FileLink
FileLink(r'prediction.csv')
