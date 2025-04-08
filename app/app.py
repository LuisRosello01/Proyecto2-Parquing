from flask import Flask, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from segmentacio.segmentacio import segment_license_plate

# Cargar el modelo
model = load_model('caracter_model.keras')

# Mapeo de caracteres válidos
valid_characters = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def preprocess_character(char_img):
    # Convertir a escala de grises si no lo está
    if len(char_img.shape) == 3:  # Si tiene 3 canales (RGB)
        char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    # Redimensionar al tamaño esperado por el modelo
    char_img = cv2.resize(char_img, (32, 32))
    # Normalizar valores de píxeles
    char_img = char_img.astype('float32') / 255.0
    # Expandir dimensiones para que coincida con el formato esperado por el modelo
    char_img = np.expand_dims(char_img, axis=-1)  # Añadir canal
    char_img = np.expand_dims(char_img, axis=0)  # Añadir batch
    return char_img

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Verificar si se envió un archivo
    if 'image' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen'}), 400

    # Leer la imagen enviada
    file = request.files['image']
    image_path = "temp_license_plate.jpg"
    file.save(image_path)

    try:
        # Segmentar caracteres de la matrícula
        character_images = segment_license_plate(image_path)

        predictions = []
        for char_img in character_images:
            # Preprocesar cada carácter
            processed_char = preprocess_character(char_img)
            # Realizar la predicción
            prediction = model.predict(processed_char)
            predicted_class = np.argmax(prediction)
            predicted_character = valid_characters[predicted_class]
            confidence = np.max(prediction) * 100
            if confidence > 40:
                predictions.append({
                    'character': predicted_character,
                    'confidence': f'{confidence:.2f}%'
                })

        return jsonify(predictions)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)