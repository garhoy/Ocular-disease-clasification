import os
import shutil
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocessor
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Usar GPU 2

def setup_data_generators(base_dir="DataTrainMix", target_size=(512, 512), batch_size=32):
    preprocessor = densenet_preprocessor
    datagen = ImageDataGenerator(
        preprocessing_function=preprocessor,
        validation_split=0.2
    )

    validation_generator = datagen.flow_from_directory(
        base_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        classes=['normal', 'diabetes', 'glaucoma', 'miopia', 'other']
    )

    return validation_generator

def clasificar_y_mover_imagenes(model, generator, target_dir):
    steps = np.ceil(generator.samples / generator.batch_size)
    predictions = model.predict(generator, steps=steps)
    
    for i, (img_path, pred) in enumerate(zip(generator.filepaths, predictions)):
        class_idx = np.argmax(pred)
        if class_idx == 4:  # Si es la clase "Other"
            shutil.copy(img_path, os.path.join(target_dir, os.path.basename(img_path)))

if __name__ == '__main__':
    base_dir_nivel_1 = 'DataTrainCascada'
    batch_size = 32
    target_size = (512, 512)
    model_path = 'ModeloDenseCascada.h5'

    # Cargar el modelo entrenado
    model_nivel_1 = load_model(model_path)

    # Configurar generador de datos
    validation_generator_nivel_1 = setup_data_generators(base_dir_nivel_1, target_size, batch_size)

    # Crear directorio temporal para imágenes "Other"
    other_temp_dir = 'OtherTemp'
    os.makedirs(other_temp_dir, exist_ok=True)

    # Clasificar y mover imágenes "Other"
    clasificar_y_mover_imagenes(model_nivel_1, validation_generator_nivel_1, other_temp_dir)
