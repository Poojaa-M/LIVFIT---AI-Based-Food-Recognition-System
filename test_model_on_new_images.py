import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.applications.inception_v3 import decode_predictions

# Load your trained model (replace with the path to your trained model file)
model = load_model('model/inception_v3_best.h5')

# Set up the ImageDataGenerator for preprocessing (this matches what you did during training)
datagen = ImageDataGenerator(rescale=1./255)

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np

# Assuming you have this dictionary from your training phase:
class_indices = {
    0: 'Burger',
    1: 'Butter_Naan',
    2: 'Chai',
    3: 'Chapati',
    4: 'Chole_Bhature',
    5: 'Dal_Makhani',
    6: 'Dhokla',
    7: 'Fried_Rice',
    8: 'Idli',
    9: 'Jalebi',
    10: 'Kaathi_Rolls',
    11: 'Kadai_Panner',
    12: 'Kulfi',
    13: 'Masala_Dosa',
    14: 'Momos',
    15: 'Paani_Puri',
    16: 'Pakode',
    17: 'Pav_Bhaji',
    18: 'Pizza',
    19: 'Samosa',
}

def predict_image(image_path):
    # Load and resize the image to 150x150 (for InceptionV3 input)
    img = load_img(image_path, target_size=(150, 150))  # Resize to (150, 150)
    
    # Convert the image to a numpy array
    img_array = img_to_array(img)
    
    # Add batch dimension (1, 150, 150, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess the image for the InceptionV3 model
    img_array = preprocess_input(img_array)
    
    # Make prediction
    predictions = model.predict(img_array)
    
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    # Map the predicted index to the class name
    predicted_class_name = class_indices.get(predicted_class_index, "Unknown")
    
    print(f"Predicted class index: {predicted_class_index}")
    print(f"Predicted class name: {predicted_class_name}")

# Example usage
predict_image("C:/Users/Jayas/OneDrive/Desktop/Project/dokla.png")

