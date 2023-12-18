import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

data_dir = 'dataset'  # Change this to your actual dataset directory
test_data_dir = 'test_data'  # Change this to your actual test dataset directory

image_size = (128, 64)  # Set the desired image size
# Load and preprocess the training data using ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
# Training data
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=32,
    class_mode='binary',  # Change to 'categorical' if you have more than two classes
    subset='training',
    classes=['fake', 'genuine']  # Add the classes you want to consider
)


# In[5]:


# Validation data
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=32,
    class_mode='binary',  # Change to 'categorical' if you have more than two classes
    subset='validation',
    classes=['fake', 'genuine']  # Add the classes you want to consider
)


# In[6]:


# Load and preprocess the test data using ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)
# Test data
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    batch_size=32,
    class_mode='binary',  # Change to 'categorical' if you have more than two classes
    shuffle=False,  # Set shuffle to False for evaluation
    classes=['fake', 'genuine']  # Add the classes you want to consider
)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define your model
model = Sequential()

# Convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten layer
model.add(Flatten())

# Dense layers
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  
import matplotlib.pyplot as plt

# Set the number of epochs
epochs = 25  # You can adjust this number as needed

# Initialize lists to store training, validation, and test accuracy
train_accuracy = []
#val_accuracy = []
test_accuracy = []

# Train the model
for epoch in range(epochs):
    # Fit the model to the training data
    history = model.fit(train_generator, epochs=1, validation_data=validation_generator)

    # Evaluate the model on the test data
    test_result = model.evaluate(test_generator)
    test_accuracy.append(test_result[1])

    # Append training accuracy for the epoch
    train_accuracy.append(history.history['accuracy'][0])

    # Append validation accuracy for the epoch
 #   val_accuracy.append(history.history['val_accuracy'][0])

    # Print and plot the accuracy
    print(f'Epoch {epoch + 1}/{epochs}, Training Accuracy: {train_accuracy[-1]}, Test Accuracy: {test_accuracy[-1]}')
   # print(f'Epoch {epoch + 1}/{epochs}, Training Accuracy: {train_accuracy[-1]}, Validation Accuracy: {val_accuracy[-1]}, Test Accuracy: {test_accuracy[-1]}')

# Plotting the accuracy graph
plt.plot(range(1, epochs + 1), train_accuracy, label='Training Accuracy')
#plt.plot(range(1, epochs + 1), val_accuracy, label='Validation Accuracy')
plt.plot(range(1, epochs + 1), test_accuracy, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy Over Epochs')
plt.legend()
plt.show()
# Get the training accuracy from the history object
train_accuracy = history.history['accuracy'][-1]
# Predict on the test data
predictions = model.predict(test_generator)

# Convert predicted probabilities to class labels
predicted_labels = [1 if pred > 0.5 else 0 for pred in predictions]  # Adjust threshold as needed

# Get the true labels from the generator
true_labels = test_generator.classes

# Display the actual and predicted labels
for true_label, predicted_label in zip(true_labels, predicted_labels):
    print(f"Actual Label: {true_label}, Predicted Label: {predicted_label}")
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'y_true' is the true labels and 'y_pred' is the predicted labels
y_true = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
y_pred = [0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,1,1,0,1,1,1,0,0,1,0,1,1,1,1,1]

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Display the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Genuine'], yticklabels=['Fake', 'Genuine'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print(classification_report(y_true, y_pred))

from ipywidgets import FileUpload,Label
from PIL import Image as PILImage
import io
from keras.preprocessing import image



model = Sequential()

# Convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten layer
model.add(Flatten())

# Dense layers
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Use 'softmax' if you have more than two classes

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Function to handle file upload and prediction
def handle_upload(change):
    # Get the uploaded image content
    img_content = list(uploader.value.values())[0]['content']
    
    # Create an in-memory file-like object from the uploaded content
    img = PILImage.open(io.BytesIO(img_content))
    
    # Resize the image to match the input shape expected by the model
    img = img.resize((64, 128))
    
    # Convert the image to a NumPy array
    img_array = image.img_to_array(img)
    
    # Expand the dimensions to match the input shape expected by the model
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the image data
    img_array /= 255.0
    
    # Use your trained model to predict
    prediction = model.predict(img_array)
    
    # Display the result
    if prediction > 0.5:
        result.value = "Prediction: Genuine"
    else:
        result.value = "Prediction: Fake"

# Create a file uploader widget
uploader = FileUpload(accept='image/*', multiple=False)
uploader.observe(handle_upload, names='value')

# Create a result label
result = Label()

# Display widgets
display(uploader)
display(result)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[17]:


#this is the code for mobileNetv2


# In[18]:


# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


# In[19]:


# Create the MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 64, 3))
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


print("Training Classes:", np.unique(y_train))
print("Validation Classes:", np.unique(y_test))

train_generator = datagen.flow(X_train, y_train, batch_size=32, subset='training', shuffle=True)
validation_generator = datagen.flow(X_train, y_train, batch_size=32, subset='validation', shuffle=True)



generator = datagen.flow(X_train, y_train, batch_size=32, subset='training', shuffle=True)

# Train the model
model.fit(generator, epochs=200)

# Evaluate the model (you can split your data into training and testing sets if needed)
accuracy = model.evaluate(generator)[1]
print(f"Test Accuracy: {accuracy}")
accuracy_percentage = accuracy * 100
print(f"Test Accuracy: {accuracy_percentage:.2f}%")


# In[ ]:


# Debugging: Print predictions for some images during training
debug_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 64),
    batch_size=1,  # Use a batch size of 1 to visualize individual predictions
    class_mode='binary',
    shuffle=False  # Ensure the order is preserved for debugging
)

for i in range(10):  # Print predictions for the first 10 images
    x, y = debug_generator.next()
    prediction = model.predict(x)
    print(f"True Label: {int(y[0])}, Predicted Label: {int(round(prediction[0][0]))}")


# In[ ]:


print(generator.class_indices)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]: