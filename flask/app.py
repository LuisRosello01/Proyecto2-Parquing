from flask import Flask, request, jsonify
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from segmentacio.segmentacio import segmentar_matricula
from ultralytics import YOLO

# Cargar el modelo
ocr = load_model(os.path.join(os.path.dirname(__file__), 'caracter_model.keras'))
yolo = YOLO(os.path.join(os.path.dirname(__file__), 'best.pt'))

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

@app.route('/detectar_matricula', methods=['POST'])
def detectar_matricula():
    # Verificar si se envió un archivo
    if 'image' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen'}), 400
    
    # Leer la imagen enviada
    file = request.files['image']
    image_path = os.path.join(os.path.dirname(__file__), 'temp_coche.jpg')
    file.save(image_path)
        
    try:
        results = yolo.predict(source=image_path, conf=0.25, save=False)
        for result in results:
            for box in result.boxes:
                # Obtener coordenadas del cuadro delimitador
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Leer la imagen original
                original_image = cv2.imread(image_path)
                # Recortar la región de interés (ROI)
                cropped_image = original_image[y1:y2, x1:x2]
                # Guardar la imagen recortada
                cropped_image_path = os.path.join(os.path.dirname(__file__), 'temp_matricula.jpg')
                cv2.imwrite(cropped_image_path, cropped_image)
                return jsonify({'message': 'Imagen recortada guardada', 'path': cropped_image_path})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    # Verificar si se envió un archivo
    if 'image' in request.files:
        # Leer la imagen enviada directamente en memoria
        file = request.files['image']
        file_stream = file.read()
        np_image = np.frombuffer(file_stream, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    elif os.path.exists(os.path.join(os.path.dirname(__file__), 'temp_matricula.jpg')):
        # Usar la imagen temporal guardada previamente
        image_path = os.path.join(os.path.dirname(__file__), 'temp_matricula.jpg')
        image = cv2.imread(image_path)
    else:
        return jsonify({'error': 'No se envió ninguna imagen'}), 400

    try:
        # Segmentar caracteres de la matrícula
        caracters = segmentar_matricula(image)

        predictions = []
        for caracter in caracters:
            # Preprocesar cada carácter
            processed_char = preprocess_character(caracter)
            # Realizar la predicción
            prediction = ocr.predict(processed_char)
            predicted_class = np.argmax(prediction)
            predicted_character = valid_characters[predicted_class]
            confidence = np.max(prediction) * 100
            if confidence > 35:
                predictions.append({
                    'character': predicted_character,
                    'confidence': f'{confidence:.2f}%'
                })

        return jsonify(predictions)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)