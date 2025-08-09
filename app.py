import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('model/inception_v3_best.h5')

# Dictionary mapping class indices to food details
class_nutrition_info = {
    0: {
        "name": "Burger",
        "calories": "400 kcal",
        "protein": "15 g",
        "fat": "25 g",
        "carbohydrates": "35 g",
        "fiber": "3 g",
        "digestion_time": "4-5 hours",
        "remarks": "High in protein and fat, especially if fried, slow to digest, depending on toppings and preparation.",
        "tips": "Choose a lean meat or vegetable patty option to make the burger healthy."
    },
    1: {
        "name": "Butter Naan",
        "calories": "200-250 kcal",
        "protein": "4-6 g",
        "fat": "8-12 g",
        "carbohydrates": "30-40 g",
        "fiber": "2-3 g",
        "digestion_time": "2-3 hours",
        "remarks": "Rich in carbs and fats due to butter; best paired with high-protein dishes.",
        "tips": "Opt for whole-wheat naan and reduce butter to lower calorie count."
    },
    2: {
        "name": "Chai",
        "calories": "90-120 kcal",
        "protein": "2-3 g",
        "fat": "2-5 g",
        "carbohydrates": "10-15 g",
        "fiber": "0-1 g",
        "digestion_time": "1-2 hours",
        "remarks": "Contains added sugar and milk fat, moderate calories.",
        "tips": "Use low-fat milk and less sugar to make it healthier."
    },
    3: {
        "name": "Chapati",
        "calories": "70-100 kcal",
        "protein": "2-3 g",
        "fat": "2-4 g",
        "carbohydrates": "15-20 g",
        "fiber": "2-3 g",
        "digestion_time": "1-2 hours",
        "remarks": "A staple food high in fiber and easily digestible.",
        "tips": "Use whole wheat flour and pair with vegetables for balanced nutrition."
    },
    4: {
        "name": "Chole Bhature",
        "calories": "400-450 kcal",
        "protein": "12-15 g",
        "fat": "15-20 g",
        "carbohydrates": "50-60 g",
        "fiber": "5-6 g",
        "digestion_time": "3-4 hours",
        "remarks": "High in carbs and fat, a heavy meal.",
        "tips": "Use baked bhature and less oil for a lighter version."
    },
    5: {
        "name": "Dal Makhani",
        "calories": "300-350 kcal",
        "protein": "12-15 g",
        "fat": "12-15 g",
        "carbohydrates": "20-25 g",
        "fiber": "5-6 g",
        "digestion_time": "3-4 hours",
        "remarks": "Rich in protein but high in fat due to cream and butter.",
        "tips": "Use less butter and cream; add more lentils for protein."
    },
    6: {
        "name": "Dhokla",
        "calories": "150-200 kcal",
        "protein": "6-8 g",
        "fat": "2-4 g",
        "carbohydrates": "20-25 g",
        "fiber": "2-3 g",
        "digestion_time": "1-2 hours",
        "remarks": "Light and easily digestible, good for snacks.",
        "tips": "Steam instead of frying, and pair with mint chutney."
    },
    7: {
        "name": "Fried Rice",
        "calories": "300-400 kcal",
        "protein": "6-8 g",
        "fat": "8-10 g",
        "carbohydrates": "40-50 g",
        "fiber": "2-3 g",
        "digestion_time": "3-4 hours",
        "remarks": "High in carbs and fats, depends on oil and veggies used.",
        "tips": "Use brown rice and add more vegetables; limit oil."
    },
    8: {
        "name": "Groundnut Burfi",
        "calories": "150-180 kcal per piece",
        "protein": "4-6 g",
        "fat": "8-10 g",
        "carbohydrates": "15-20 g",
        "fiber": "1-2 g",
        "digestion_time": "2-3 hours",
        "remarks": "Good source of protein and healthy fats, but calorie dense.",
        "tips": "Use jaggery instead of sugar for a healthier option."
    },
    9: {
        "name": "Idli",
        "calories": "60-80 kcal",
        "protein": "2-3 g",
        "fat": "0.5-1 g",
        "carbohydrates": "10-15 g",
        "fiber": "0.5-1 g",
        "digestion_time": "1-2 hours",
        "remarks": "Low-calorie and easily digestible.",
        "tips": "Pair with sambhar for a balanced meal."
    },
    10: {
        "name": "Jalebi",
        "calories": "200-250 kcal",
        "protein": "1-2 g",
        "fat": "10-12 g",
        "carbohydrates": "30-40 g",
        "fiber": "0-1 g",
        "digestion_time": "2-3 hours",
        "remarks": "High in sugar and fat, best as an occasional treat.",
        "tips": "Bake instead of frying; use jaggery for a healthier version."
    },
    11: {
        "name": "Kaathi Rolls",
        "calories": "200-300 kcal",
        "protein": "10-15 g",
        "fat": "8-12 g",
        "carbohydrates": "20-30 g",
        "fiber": "2-3 g",
        "digestion_time": "2-3 hours",
        "remarks": "Nutritional content varies by filling; a balanced snack.",
        "tips": "Use whole wheat wraps and limit oily fillings."
    },
    12: {
        "name": "Kadai Paneer",
        "calories": "250-300 kcal",
        "protein": "12-15 g",
        "fat": "15-18 g",
        "carbohydrates": "10-15 g",
        "fiber": "2-3 g",
        "digestion_time": "2-3 hours",
        "remarks": "Rich in protein and fat due to paneer and cream.",
        "tips": "Use low-fat paneer and less oil for a healthier dish."
    },
    13: {
        "name": "Kulfi",
        "calories": "100-150 kcal",
        "protein": "2-4 g",
        "fat": "8-10 g",
        "carbohydrates": "15-20 g",
        "fiber": "0-1 g",
        "digestion_time": "1-2 hours",
        "remarks": "High in sugar; best enjoyed occasionally.",
        "tips": "Use natural sweeteners for a healthier version."
    },
    14: {
        "name": "Masala Dosa",
        "calories": "200-250 kcal",
        "protein": "4-6 g",
        "fat": "6-8 g",
        "carbohydrates": "30-35 g",
        "fiber": "2-3 g",
        "digestion_time": "2-3 hours",
        "remarks": "Rich in carbs; balanced with sambhar and chutney.",
        "tips": "Reduce oil and include more vegetables in the filling."
    },
    15: {
        "name": "Momos",
        "calories": "40-60 kcal per piece",
        "protein": "1-2 g",
        "fat": "1-2 g",
        "carbohydrates": "5-8 g",
        "fiber": "0.5-1 g",
        "digestion_time": "1-2 hours",
        "remarks": "Light snack; depends on filling and preparation.",
        "tips": "Steam instead of frying and choose vegetable fillings."
    },
    16: {
        "name": "Paani Puri",
        "calories": "150-200 kcal",
        "protein": "2-3 g",
        "fat": "5-8 g",
        "carbohydrates": "20-25 g",
        "fiber": "1-2 g",
        "digestion_time": "1-2 hours",
        "remarks": "Hygienic preparation is crucial for safety.",
        "tips": "Use baked puris and avoid sugary tamarind water."
    },
    17: {
        "name": "Pakode",
        "calories": "200-250 kcal",
        "protein": "2-3 g",
        "fat": "10-12 g",
        "carbohydrates": "20-25 g",
        "fiber": "1-2 g",
        "digestion_time": "2-3 hours",
        "remarks": "High in fat due to frying.",
        "tips": "Bake or air-fry for a healthier option."
    },
    18: {
        "name": "Pav Bhaji",
        "calories": "300-350 kcal",
        "protein": "8-10 g",
        "fat": "10-15 g",
        "carbohydrates": "40-50 g",
        "fiber": "5-6 g",
        "digestion_time": "3-4 hours",
        "remarks": "High in carbs and fats due to butter.",
        "tips": "Use whole wheat pav and less butter."
    },
    19: {
        "name": "Pizza",
        "calories": "250-400 kcal per slice",
        "protein": "10-15 g",
        "fat": "10-20 g",
        "carbohydrates": "30-40 g",
        "fiber": "2-3 g",
        "digestion_time": "3-4 hours",
        "remarks": "High in fat and carbs; toppings determine nutritional value.",
        "tips": "Use whole wheat crust and vegetable toppings."
    },
    20: {
        "name": "Samosa",
        "calories": "150-200 kcal per piece",
        "protein": "2-4 g",
        "fat": "8-10 g",
        "carbohydrates": "20-25 g",
        "fiber": "1-2 g",
        "digestion_time": "2-3 hours",
        "remarks": "Deep-fried, high in fat and carbs.",
        "tips": "Bake or air-fry for a lighter version."
    },

}

def predict_image(img):
    # Resize the image to match model input size (InceptionV3 takes 150x150)
    img = cv2.resize(img, (150, 150))
    
    # Convert image to array and preprocess for the model
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # InceptionV3 preprocessing
    
    # Get predictions from the model
    predictions = model.predict(img_array)
    
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    # Get the food information from the index
    food_info = class_nutrition_info.get(predicted_class_index, None)
    
    return food_info

# Start video capture
cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Make prediction
    food_info = predict_image(frame)
    
    # Inside the prediction loop
    # Inside the prediction loop
     # Inside the prediction loop
    if food_info:
        food_name = food_info["name"]
        calories = food_info["calories"]
        protein = food_info["protein"]
        fat = food_info["fat"]
        carbohydrates = food_info["carbohydrates"]
        fiber = food_info["fiber"]
        digestion_time = food_info["digestion_time"]
        remarks = food_info["remarks"]
        tips = food_info["tips"]
        
        box_x, box_y = 10, 50  # Top-left corner of the box
        box_width, box_height = 600, 400  # Box dimensions
        
        overlay = frame.copy()
        
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 0, 0), -1)
        
        alpha = 0.6  # Transparency factor (0 = fully transparent, 1 = opaque)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        y_offset = box_y + 30  # Start inside the box
        line_spacing = 30  # Space between lines
        
        text_color = (255, 255, 255)  # White
        
        cv2.putText(frame, f'Food: {food_name}', (box_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        y_offset += line_spacing
        cv2.putText(frame, f'Calories: {calories}', (box_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        y_offset += line_spacing
        cv2.putText(frame, f'Protein: {protein}', (box_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        y_offset += line_spacing
        cv2.putText(frame, f'Fat: {fat}', (box_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        y_offset += line_spacing
        cv2.putText(frame, f'Carbs: {carbohydrates}', (box_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        y_offset += line_spacing
        cv2.putText(frame, f'Fiber: {fiber}', (box_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        y_offset += line_spacing
        cv2.putText(frame, f'Digestion Time: {digestion_time}', (box_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        y_offset += line_spacing
        cv2.putText(frame, f'Remarks: {remarks}', (box_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        y_offset += line_spacing
        cv2.putText(frame, f'Tips: {tips}', (box_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)


    else:
        cv2.putText(frame, 'Food: Unknown', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Display the resulting frame
    cv2.imshow('Webcam Prediction', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()  # Releases the webcam
cv2.destroyAllWindows()  # Closes all OpenCV windows


