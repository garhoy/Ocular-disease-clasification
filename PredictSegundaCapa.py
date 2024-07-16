import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import os
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Usar GPU 2

def load_and_preprocess_image(img_path, target_size=(512, 512)):
    img = image.load_img(img_path, target_size=target_size)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.0
    return img_tensor

def get_random_images(img_dir, max_images=30):
    all_img_files = []
    for root, _, files in os.walk(img_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                all_img_files.append(os.path.join(root, file))
    return random.sample(all_img_files, min(max_images, len(all_img_files)))

def extract_real_label(img_name):
    if "ARMD" in img_name:
        return "ARMD"
    else:
        return "Catarata"

def predict_and_save_images(model, img_paths, save_dir, classes):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for img_path in img_paths:
        img_tensor = load_and_preprocess_image(img_path)
        prediction = model.predict(img_tensor)
        pred_class_idx = int(prediction > 0.5)
        pred_label = classes[pred_class_idx]

        img_name = os.path.basename(img_path)
        real_label = extract_real_label(img_name)

        # Cargar la imagen con OpenCV
        img = cv2.imread(img_path)
        img = cv2.resize(img, (512, 512))

        # Configuración del texto
        text = f'Prediccion: {pred_label}\nEtiqueta Real: {real_label}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        color = (255, 255, 255)  # Blanco
        thickness = 2
        line_type = cv2.LINE_AA

        # Agregar texto a la imagen
        y0, dy = 30, 30
        for i, line in enumerate(text.split('\n')):
            y = y0 + i * dy
            cv2.putText(img, line, (10, y), font, font_scale, color, thickness, line_type)

        # Guardar la imagen
        save_path = os.path.join(save_dir, os.path.basename(img_path))
        cv2.imwrite(save_path, img)
        print(f'Imagen guardada en {save_path} con la predicción: {pred_label} y etiqueta real: {real_label}')

if __name__ == '__main__':
    model_path = 'modelo_secundario.h5'
    img_dir = 'OtherTemp'  # Directorio donde están las imágenes a predecir
    save_dir = 'Predicciones_SegundoModelo'
    classes = ['Catarata', 'ARMD']  # Clases para el segundo modelo

    # Cargar el modelo entrenado
    model = load_model(model_path)

    # Obtener imágenes aleatorias del conjunto de datos
    random_img_paths = get_random_images(img_dir, max_images=5)

    # Predecir y guardar imágenes con predicciones
    predict_and_save_images(model, random_img_paths, save_dir, classes)
