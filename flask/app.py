from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import cv2
import os
import base64
from tensorflow.keras.models import load_model
from segmentacio.segmentacio import segmentar_matricula
from ultralytics import YOLO
from pymongo import MongoClient
from flask_cors import CORS
from datetime import datetime
import uuid
import json

# MongoDB Connection
client = MongoClient('mongodb://localhost:27017/')
db = client.parking_system
vehicles_collection = db.vehicles
config_collection = db.config

# Initialize config collection if empty
if config_collection.count_documents({}) == 0:
    config_collection.insert_many([
        {'key': 'rate_per_hour', 'value': '2.50'},
        {'key': 'max_capacity', 'value': '100'}
    ])

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
CORS(app)  # Enable CORS for all routes

@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

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
        # Leer la imagen original para tenerla disponible
        original_image = cv2.imread(image_path)
        if original_image is None:
            return jsonify({'error': 'No se pudo leer la imagen'}), 500
            
        results = yolo.predict(source=image_path, conf=0.25, save=False)
        
        for result in results:
            if len(result.boxes) > 0:  # Si se detectó al menos una matrícula
                for box in result.boxes:
                    # Obtener coordenadas del cuadro delimitador
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Recortar la región de interés (ROI)
                    cropped_image = original_image[y1:y2, x1:x2]
                    # Guardar la imagen recortada
                    cropped_image_path = os.path.join(os.path.dirname(__file__), 'temp_matricula.jpg')
                    cv2.imwrite(cropped_image_path, cropped_image)
                    
                    # Codificar la imagen recortada en base64 para devolverla en la respuesta
                    _, buffer = cv2.imencode('.jpg', cropped_image)
                    cropped_image_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    return jsonify({
                        'message': 'Imagen recortada guardada',
                        'path': cropped_image_path,
                        'image_data': cropped_image_base64
                    })
        
        # Si no se encontró ninguna matrícula, enviar la imagen original para que el OCR intente procesarla directamente
        # Codificar la imagen original en base64
        _, buffer = cv2.imencode('.jpg', original_image)
        original_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Guardar la imagen original como si fuera la recortada para mantener la consistencia del flujo
        cropped_image_path = os.path.join(os.path.dirname(__file__), 'temp_matricula.jpg')
        cv2.imwrite(cropped_image_path, original_image)
        
        return jsonify({
            'message': 'No se detectó ninguna matrícula, usando imagen completa',
            'path': cropped_image_path,
            'image_data': original_image_base64
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    # Verificar si se envió un archivo o imagen en base64
    if 'image' in request.files:
        # Leer la imagen enviada directamente en memoria
        file = request.files['image']
        file_stream = file.read()
        np_image = np.frombuffer(file_stream, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    elif 'image_data' in request.json:
        # Decodificar la imagen en base64
        try:
            image_data = base64.b64decode(request.json['image_data'])
            np_image = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        except Exception as e:
            return jsonify({'error': f'Error al decodificar la imagen: {str(e)}'}), 400
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

        # Obtener la matrícula completa
        license_plate = ''.join([p['character'] for p in predictions]) if predictions else ''
        
        # Verificar si se solicita registro automático
        if license_plate and request.args.get('register') == 'entry':
            response = register_entry(license_plate)
            return jsonify({
                'plate_recognition': predictions,
                'license_plate': license_plate,
                'entry_data': response
            })
        elif license_plate and request.args.get('register') == 'exit':
            response = register_exit(license_plate)
            return jsonify({
                'plate_recognition': predictions,
                'license_plate': license_plate,
                'exit_data': response
            })
        
        return jsonify({
            'plate_recognition': predictions,
            'license_plate': license_plate
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# New endpoints for parking management

@app.route('/entry', methods=['POST'])
def entry():
    if not request.json or 'license_plate' not in request.json:
        return jsonify({'error': 'Se requiere la matrícula del vehículo'}), 400
    
    license_plate = request.json['license_plate']
    return jsonify(register_entry(license_plate))

def register_entry(license_plate):
    # Verificar si el vehículo ya está en el parking
    if vehicles_collection.count_documents({'license_plate': license_plate, 'exit_time': None}) > 0:
        return {'error': 'Este vehículo ya se encuentra en el parking'}
    
    # Verificar capacidad del parking
    current_count = vehicles_collection.count_documents({'exit_time': None})
    max_capacity = int(config_collection.find_one({'key': 'max_capacity'})['value'])
    
    if current_count >= max_capacity:
        return {'error': 'Parking lleno'}
    
    # Generar ticket
    entry_time = datetime.now()
    ticket_id = str(uuid.uuid4())[:8]  # ID corto para el ticket
    
    # Insertar registro
    vehicle_data = {
        'license_plate': license_plate,
        'entry_time': entry_time,
        'ticket_id': ticket_id,
        'paid': False,
        'amount': None,
        'exit_time': None
    }
    
    vehicles_collection.insert_one(vehicle_data)
    
    return {
        'message': 'Entrada registrada correctamente',
        'ticket_id': ticket_id,
        'license_plate': license_plate,
        'entry_time': entry_time.strftime('%Y-%m-%d %H:%M:%S')
    }

@app.route('/exit', methods=['POST'])
def exit():
    if not request.json or 'license_plate' not in request.json:
        return jsonify({'error': 'Se requiere la matrícula del vehículo'}), 400
    
    license_plate = request.json['license_plate']
    return jsonify(register_exit(license_plate))

def register_exit(license_plate):
    # Buscar el vehículo en el parking
    vehicle = vehicles_collection.find_one({'license_plate': license_plate, 'exit_time': None})
    
    if not vehicle:
        return {'error': 'No se encontró ningún vehículo con esta matrícula en el parking'}
    
    entry_time = vehicle['entry_time']
    exit_time = datetime.now()
    
    # Calcular duración y coste
    duration_seconds = (exit_time - entry_time).total_seconds()
    duration_hours = duration_seconds / 3600
    
    rate_per_hour = float(config_collection.find_one({'key': 'rate_per_hour'})['value'])
    amount = rate_per_hour * duration_hours
    
    # Actualizar registro
    vehicles_collection.update_one(
        {'_id': vehicle['_id']},
        {'$set': {
            'exit_time': exit_time,
            'amount': round(amount, 2)
        }}
    )
    
    return {
        'message': 'Salida registrada correctamente',
        'license_plate': license_plate,
        'entry_time': entry_time.strftime('%Y-%m-%d %H:%M:%S'),
        'exit_time': exit_time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration_hours': round(duration_hours, 2),
        'amount': round(amount, 2),
        'ticket_id': vehicle['ticket_id'],
        'status': 'Pendiente de pago'
    }

@app.route('/payment', methods=['POST'])
def payment():
    if not request.json or 'ticket_id' not in request.json:
        return jsonify({'error': 'Se requiere el ID del ticket'}), 400
    
    ticket_id = request.json['ticket_id']
    
    # Buscar el ticket
    vehicle = vehicles_collection.find_one({'ticket_id': ticket_id, 'exit_time': {'$ne': None}})
    
    if not vehicle:
        return jsonify({'error': 'Ticket no encontrado o vehículo aún en parking'}), 404
    
    if vehicle['paid']:
        return jsonify({
            'message': 'El ticket ya ha sido pagado', 
            'license_plate': vehicle['license_plate']
        })
    
    # Procesar el pago
    vehicles_collection.update_one(
        {'_id': vehicle['_id']},
        {'$set': {'paid': True}}
    )
    
    return jsonify({
        'message': 'Pago procesado correctamente',
        'license_plate': vehicle['license_plate'],
        'amount_paid': vehicle['amount'],
        'ticket_id': ticket_id
    })

@app.route('/verify_ticket/<ticket_id>', methods=['GET'])
def verify_ticket(ticket_id):
    # Buscar el ticket
    vehicle = vehicles_collection.find_one({'ticket_id': ticket_id})
    
    if not vehicle:
        return jsonify({'error': 'Ticket no encontrado'}), 404
    
    # Convertir ObjectId a string para que sea JSON serializable
    vehicle['_id'] = str(vehicle['_id'])
    
    # Formatear fechas
    if vehicle['entry_time']:
        vehicle['entry_time'] = vehicle['entry_time'].strftime('%Y-%m-%d %H:%M:%S')
    if vehicle['exit_time']:
        vehicle['exit_time'] = vehicle['exit_time'].strftime('%Y-%m-%d %H:%M:%S')
    
    # Agregar estado
    vehicle['status'] = 'Pagado' if vehicle['paid'] else 'Pendiente de pago'
    
    return jsonify(vehicle)

@app.route('/admin/statistics', methods=['GET'])
def statistics():
    # Estadísticas generales
    current_vehicles = vehicles_collection.count_documents({'exit_time': None})
    completed_visits = vehicles_collection.count_documents({'exit_time': {'$ne': None}, 'paid': True})
    
    # Calcular ingresos totales
    revenue_cursor = vehicles_collection.find({'paid': True}, {'amount': 1})
    total_revenue = sum(doc['amount'] for doc in revenue_cursor if doc.get('amount'))
    
    # Estadísticas del día actual
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    entries_today = vehicles_collection.count_documents({'entry_time': {'$gte': today_start}})
    exits_today = vehicles_collection.count_documents({'exit_time': {'$gte': today_start}})
    
    return jsonify({
        'current_vehicles': current_vehicles,
        'completed_visits': completed_visits,
        'total_revenue': round(total_revenue, 2),
        'entries_today': entries_today,
        'exits_today': exits_today
    })

@app.route('/admin/config', methods=['GET', 'PUT'])
def manage_config():
    if request.method == 'GET':
        config = {doc['key']: doc['value'] for doc in config_collection.find({}, {'key': 1, 'value': 1, '_id': 0})}
        return jsonify(config)
    
    elif request.method == 'PUT':
        if not request.json:
            return jsonify({'error': 'No se enviaron datos'}), 400
        
        # Actualizar configuraciones
        for key, value in request.json.items():
            config_collection.update_one(
                {'key': key},
                {'$set': {'value': str(value)}}
            )
        
        return jsonify({'message': 'Configuración actualizada correctamente'})

@app.route('/vehicles', methods=['GET'])
def get_vehicles():
    # Parámetros de paginación
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    
    # Filtros opcionales
    filters = {}
    if 'status' in request.args:
        if request.args['status'] == 'in':
            filters['exit_time'] = None
        elif request.args['status'] == 'out':
            filters['exit_time'] = {'$ne': None}
    
    if 'paid' in request.args:
        filters['paid'] = request.args['paid'].lower() == 'true'
    
    # Obtener vehículos con paginación
    total = vehicles_collection.count_documents(filters)
    
    vehicles = list(vehicles_collection.find(filters)
                   .sort('entry_time', -1)
                   .skip((page -1) * per_page)
                   .limit(per_page))
    
    # Convertir ObjectId a string y formatear fechas
    for vehicle in vehicles:
        vehicle['_id'] = str(vehicle['_id'])
        if vehicle['entry_time']:
            vehicle['entry_time'] = vehicle['entry_time'].strftime('%Y-%m-%d %H:%M:%S')
        if vehicle['exit_time']:
            vehicle['exit_time'] = vehicle['exit_time'].strftime('%Y-%m-%d %H:%M:%S')
    
    return jsonify({
        'vehicles': vehicles,
        'total': total,
        'page': page,
        'per_page': per_page,
        'pages': (total + per_page - 1) // per_page
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True)