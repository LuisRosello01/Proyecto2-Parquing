import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
import cv2
from pathlib import Path
import sys

# Constantes
IMG_HEIGHT = 50  # Reducción de altura a 50px
IMG_WIDTH = 140  # Ajuste proporcional del ancho
BATCH_SIZE = 32
EPOCHS = 30
CHARS = "0123456789 BCDFGHJKLMNPQRSTVWXYZ"  # Caracteres permitidos (dígitos + espacio + letras sin vocales)
CHAR_TO_INDEX = {char: idx for idx, char in enumerate(CHARS)}
INDEX_TO_CHAR = {idx: char for idx, char in enumerate(CHARS)}
MAX_LENGTH = 8  # 4 dígitos + 1 espacio + 3 letras

def cargar_datos(data_dir="dades"):
    """Carga las imágenes y sus correspondientes etiquetas."""
    print("Cargando datos...")
    # Ruta completa al directorio de datos
    data_path = os.path.join(os.path.dirname(__file__), data_dir)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encuentra el directorio de datos: {data_path}")
    
    images = []
    labels = []
    
    # Patrón para extraer el texto de la matrícula del nombre del archivo
    pattern = r"matricula_([0-9]{4}\s[A-Z]{3})_\d+\.jpg"
    
    # Encontrar todos los archivos de imágenes
    for img_file in os.listdir(data_path):
        if img_file.endswith('.jpg'):
            match = re.match(pattern, img_file)
            if match:
                # Extraer texto de la matrícula
                plate_text = match.group(1)
                
                # Cargar y preprocesar la imagen
                img_path = os.path.join(data_path, img_file)
                img = load_img(img_path, color_mode="grayscale", target_size=(IMG_HEIGHT, IMG_WIDTH))
                img = img_to_array(img) / 255.0  # Normalizar a [0,1]
                
                images.append(img)
                labels.append(plate_text)
    
    print(f"Se han cargado {len(images)} imágenes con sus etiquetas.")
    return np.array(images), np.array(labels)

def preprocesar_etiquetas(labels):
    """Convierte las etiquetas de texto a formato one-hot para secuencias."""
    # Crear matriz de etiquetas one-hot
    y = np.zeros((len(labels), MAX_LENGTH, len(CHARS)), dtype=np.float32)
    
    for i, label in enumerate(labels):
        for j, char in enumerate(label):
            if j < MAX_LENGTH and char in CHAR_TO_INDEX:
                # One-hot encoding para cada carácter
                y[i, j, CHAR_TO_INDEX[char]] = 1.0
    
    return y

def crear_modelo():
    """Crea un modelo CNN para OCR."""
    # Entrada
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    
    # Convoluciones
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Reshape para LSTM - Arreglando el problema de dimensionalidad
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Dividir las características para cada posición de carácter
    outputs = []
    for i in range(MAX_LENGTH):
        char_output = layers.Dense(64, activation='relu')(x)
        char_output = layers.Dense(len(CHARS), activation='softmax', name=f'char_{i}')(char_output)
        outputs.append(char_output)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Corrigiendo el problema de métricas - un diccionario de métricas para cada salida
    losses = {}
    metrics = {}
    for i in range(MAX_LENGTH):
        losses[f'char_{i}'] = 'categorical_crossentropy'
        metrics[f'char_{i}'] = 'accuracy'
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=losses,
        metrics=metrics
    )
    
    return model

def entrenar_modelo(model, X_train, y_train, X_val, y_val):
    """Entrena el modelo y devuelve el historial de entrenamiento."""
    # Preparar las etiquetas para cada salida
    y_train_outputs = [y_train[:, i, :] for i in range(MAX_LENGTH)]
    y_val_outputs = [y_val[:, i, :] for i in range(MAX_LENGTH)]
    
    # Callback para guardar el mejor modelo
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        "mejor_modelo_ocr.h5",
        monitor='val_loss',
        save_best_only=True
    )
    
    # Callback para early stopping
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train_outputs,
        validation_data=(X_val, y_val_outputs),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint_cb, early_stopping_cb]
    )
    
    return history

def predecir_matricula(model, img):
    """Predice la matrícula a partir de una imagen."""
    if isinstance(img, str):
        # Cargar desde ruta
        img = load_img(img, color_mode="grayscale", target_size=(IMG_HEIGHT, IMG_WIDTH))
        img = img_to_array(img) / 255.0
    
    img = np.expand_dims(img, axis=0)  # Añadir dimensión de batch
    
    # Realizar predicción
    predictions = model.predict(img)
    
    # Obtener el índice de máxima probabilidad para cada carácter
    plate_text = ""
    for pred in predictions:
        char_idx = np.argmax(pred[0])
        plate_text += INDEX_TO_CHAR[char_idx]
    
    return plate_text

def evaluar_modelo(model, X_test, y_test):
    """Evalúa el modelo y muestra métricas de rendimiento."""
    y_test_outputs = [y_test[:, i, :] for i in range(MAX_LENGTH)]
    
    # Evaluar el modelo
    results = model.evaluate(X_test, y_test_outputs, verbose=1)
    
    print("Resultados de evaluación:")
    total_accuracy = 0
    for i, result in enumerate(results[MAX_LENGTH:]):  # Saltando las pérdidas
        print(f"Precisión carácter {i+1}: {result:.4f}")
        total_accuracy += result
    
    print(f"Precisión media: {total_accuracy/MAX_LENGTH:.4f}")
    
    # Mostrar algunas predicciones
    n_samples = min(10, len(X_test))
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    for idx in indices:
        img = X_test[idx]
        true_label = ""
        for i in range(MAX_LENGTH):
            true_idx = np.argmax(y_test[idx, i])
            true_label += INDEX_TO_CHAR[true_idx]
        
        pred_label = predecir_matricula(model, img)
        
        print(f"Real: {true_label}, Predicción: {pred_label}")

def mostrar_curvas_entrenamiento(history):
    """Muestra gráficos del historial de entrenamiento."""
    # Extraer la primera salida para simplificar el gráfico
    loss = history.history['char_0_loss']
    val_loss = history.history['val_char_0_loss']
    acc = history.history['char_0_accuracy']
    val_acc = history.history['val_char_0_accuracy']
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(loss, label='Entrenamiento')
    plt.plot(val_loss, label='Validación')
    plt.title('Pérdida')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(acc, label='Entrenamiento')
    plt.plot(val_acc, label='Validación')
    plt.title('Precisión')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('entrenamiento_ocr.png')
    plt.close()

def guardar_modelo(model, modelo_path="modelo_ocr.h5"):
    """Guarda el modelo entrenado."""
    model.save(modelo_path)
    print(f"Modelo guardado en: {modelo_path}")

def esperar_confirmacion(mensaje="Presiona Enter para continuar o 'q' para salir: "):
    """Solicita confirmación del usuario para continuar."""
    resp = input(mensaje)
    if resp.lower() == 'q':
        print("Operación cancelada por el usuario.")
        sys.exit(0)
    return

if __name__ == "__main__":
    # Comprobar si existe la GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"Se utilizará GPU: {physical_devices}")
        # Permitir crecimiento de memoria según sea necesario
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    else:
        print("No se detectó GPU. Se utilizará CPU.")
    
    try:
        print("\n=== 🚗 OCR para Matrículas - Entrenamiento ===\n")
        esperar_confirmacion("Presiona Enter para iniciar la carga de datos o 'q' para salir: ")
        
        # Cargar datos
        X, y_text = cargar_datos()
        
        if len(X) == 0:
            print("No se encontraron imágenes. Por favor, genere datos primero.")
            exit()
        
        print(f"Se han cargado {len(X)} imágenes.")
        esperar_confirmacion("Datos cargados. Presiona Enter para continuar con el preprocesamiento o 'q' para salir: ")
        
        # Preprocesar etiquetas
        print("\nPreprocesando etiquetas...")
        y = preprocesar_etiquetas(y_text)
        print("Etiquetas preprocesadas correctamente.")
        esperar_confirmacion("Preprocesamiento completado. Presiona Enter para dividir los datos o 'q' para salir: ")
        
        # Dividir datos
        print("\nDividiendo datos en conjuntos de entrenamiento, validación y prueba...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        print(f"Datos de entrenamiento: {X_train.shape}")
        print(f"Datos de validación: {X_val.shape}")
        print(f"Datos de prueba: {X_test.shape}")
        esperar_confirmacion("División de datos completada. Presiona Enter para crear el modelo o 'q' para salir: ")
        
        # Crear modelo
        print("\nCreando modelo CNN...")
        model = crear_modelo()
        print("\nResumen del modelo:")
        model.summary()
        esperar_confirmacion("\nModelo creado. Presiona Enter para iniciar el entrenamiento o 'q' para salir: ")
        
        # Entrenar modelo
        print("\n🏋️ Iniciando entrenamiento...\n")
        history = entrenar_modelo(model, X_train, y_train, X_val, y_val)
        print("\nEntrenamiento completado.")
        esperar_confirmacion("Entrenamiento finalizado. Presiona Enter para generar gráficos o 'q' para salir: ")
        
        # Mostrar curvas de entrenamiento
        print("\nGenerando gráficos de entrenamiento...")
        mostrar_curvas_entrenamiento(history)
        print("Gráficos guardados como 'entrenamiento_ocr.png'")
        esperar_confirmacion("Gráficos generados. Presiona Enter para evaluar el modelo o 'q' para salir: ")
        
        # Evaluar modelo
        print("\n🔍 Evaluando modelo...")
        evaluar_modelo(model, X_test, y_test)
        esperar_confirmacion("Evaluación completada. Presiona Enter para guardar el modelo o 'q' para salir: ")
        
        # Guardar modelo
        print("\nGuardando modelo...")
        guardar_modelo(model)
        
        print("\n✅ ¡Entrenamiento completado con éxito!")
        esperar_confirmacion("Proceso finalizado. Presiona Enter para salir: ")
        
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        esperar_confirmacion("Se ha producido un error. Presiona Enter para salir: ")
