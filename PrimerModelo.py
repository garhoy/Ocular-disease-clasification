print(__doc__)
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Add, GlobalAveragePooling2D, Activation, Input
from tensorflow.keras.applications import InceptionV3, EfficientNetB7, DenseNet121
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import time
import numpy as np
from tensorflow.keras.optimizers import Adam
import itertools
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.nan)
from sklearn.metrics import confusion_matrix
from tensorflow.keras.regularizers import l2
import cv2
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocessor
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocessor
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocessor
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image  # Importar la librería de imagen
from PIL import Image, ImageDraw, ImageFont
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Usar GPU 2

def setup_data_generators(base_dir="DataTrainMix", target_size=(512, 512), batch_size=32, model_name="custom"):
    if model_name == "Dense":
        preprocessor = densenet_preprocessor
    else:
        preprocessor = densenet_preprocessor

    datagen = ImageDataGenerator(
        preprocessing_function=preprocessor,  # Usa la función de preprocesamiento seleccionada
        validation_split=0.2  # Divide los datos en entrenamiento y validación
    )

    train_generator = datagen.flow_from_directory(
        base_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',  # Cambiado a 'categorical' para clasificación multiclase
        subset='training',
        shuffle=True,
        classes=['normal', 'diabetes', 'glaucoma', 'miopia', 'other']
    )

    for images, labels in train_generator:
        print(labels)
        break

    validation_generator = datagen.flow_from_directory(
        base_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',  # Cambiado a 'categorical'
        subset='validation',
        shuffle=False,
        classes=['normal', 'diabetes', 'glaucoma', 'miopia', 'other']
    )

    for images, labels in validation_generator:
        print(labels)  # Imprime las etiquetas
        break  # Solo necesitamos un lote para verificar

    return train_generator, validation_generator

def build_custom_model():
    global input_shape, num_classes

    print(f"Construyendo modelo personalizado para {num_classes} clases.")
    inputs = Input(shape=input_shape)
    # Capa inicial
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    # Aumentando la profundidad progresivamente
    filter_sizes = [64, 128, 256, 512]
    for filters in filter_sizes:
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # Añadir capas convolucionales adicionales sin MaxPooling para conservar características hasta la última capa
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)

    # Capa final
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    if num_classes == 2:
        print("Configurando el modelo para clasificación binaria.")
        outputs = Dense(1, activation='sigmoid')(x)
        loss_function = 'binary_crossentropy'
    else:
        print("Configurando el modelo para clasificación multiclase.")
        outputs = Dense(num_classes, activation='softmax')(x)
        loss_function = 'categorical_crossentropy'

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=loss_function, metrics=['accuracy'])
    print(f"Modelo compilado con la función de pérdida: {loss_function}")
    return model

def build_model(model_name="Custom"):
    global input_shape, num_classes
    print("Número de clases:", num_classes)  # Debugging: Imprime el número de clases
    inputs = Input(shape=input_shape)

    if model_name == "Dense":
        print("Building Dense")
        base_model = DenseNet121(weights='imagenet', include_top=False, input_tensor=inputs)
        base_model.trainable = False
        x = base_model.output
    else:
        print("Building custom")
        return build_custom_model()

    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    if num_classes == 2:
        outputs = Dense(1, activation='sigmoid')(x)  # Correcto para clasificación binaria
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        for i in range(0, 5):
            print("Compilando modelo para clasificación binaria.")
    else:
        outputs = Dense(num_classes, activation='softmax')(x)  # Correcto para clasificación multiclase
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        for i in range(0, 5):
            print("Compilando modelo para clasificación multiclase.")

    return model

def train(model, train_generator, validation_generator):
    checkpoint = ModelCheckpoint('ModeloDenseCascada.h5', monitor='val_accuracy', verbose=1, save_weights_only=False, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=45, verbose=1, mode='max', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=25, verbose=1, mode='max', min_lr=1e-6)

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // 32, # ERa 8 antes
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // 32,
        epochs=1000,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )

    return history


def plot_training_history(history, name, save_loss=True, save_accuracy=True):
    # Save the accuracy graph
    if save_accuracy:
        plt.figure(figsize=(6, 4))
        plt.plot(history.history['accuracy'], label='Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        if name == "DosClases":
            plt.savefig("AccuracyDenseDosClases.png")
            plt.close()
        else:
            plt.savefig("AccuracyCustomEnfermedades.png")
            plt.close()

    # Save the loss graph
    if save_loss:
        plt.figure(figsize=(6, 4))
        plt.plot(history.history['loss'], label='Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        if name == "DosClases":
            plt.savefig("LossDenseDosClases.png")
            plt.close()
        else:
            plt.savefig("LossCustomEnfermedades.png")
            plt.close()

def get_pred_labels(model, generator, num_classes):
    steps = np.ceil(generator.samples / generator.batch_size)
    predictions = model.predict(generator, steps=steps)

    if num_classes == 2:
        y_score = (predictions > 0.5).astype(int).flatten()  # Umbral de 0.5 para clasificación binaria
    else:
        y_score = np.argmax(predictions, axis=-1)  # Para clasificación multiclase

    y_test = generator.classes

    return y_score, y_test

def plot_confusion_matrix(cm, name, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if name == "DosClases":
        plt.savefig("ConfusionDenseDosClases.png")
        plt.close()
    else:
        plt.savefig("ConfusionCustomEnfermedades.png")
        plt.close()

def save_predictions_with_labels(model, img_dir, save_dir, num_classes, max_images=30):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_files = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, fname))]
    img_files = img_files[:max_images]

    for img_path in img_files:
        img = image.load_img(img_path, target_size=(512, 512))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.0

        prediction = model.predict(img_tensor)
        pred_class_idx = np.argmax(prediction) if num_classes > 2 else int(prediction > 0.5)
        pred_label = ['normal', 'diabetes', 'glaucoma', 'miopia', 'other'][pred_class_idx]

        pil_img = Image.open(img_path).resize((512, 512))
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype("arial", 20)
        real_label = os.path.basename(os.path.dirname(img_path))
        text = f'Predicción: {pred_label}\nReal: {real_label}'
        draw.text((10, 10), text, font=font, fill='white')

        save_path = os.path.join(save_dir, os.path.basename(img_path))
        pil_img.save(save_path)
        print(f'Imagen guardada en {save_path} con la predicción: {pred_label} y etiqueta real: {real_label}')
        # Imprimir la predicción y la etiqueta real
        print(f'Imagen: {img_path}, Predicción: {pred_label}, Real: {real_label}')

if __name__ == '__main__':
    base_dir_nivel_1 = 'DataTrainCascada'
    batch_size = 32
    target_size = (512, 512)
    input_shape = (512, 512, 3)
    num_classes = 5

    # Primer nivel: Clasificación global de 5 clases
    train_generator_nivel_1, validation_generator_nivel_1 = setup_data_generators(base_dir_nivel_1, target_size, batch_size)

    model_nivel_1 = build_model("Dense")
    model_nivel_1.summary()
    start_time = time.time()

    #Comento el train y añado el load model sino
    history_nivel_1 = train(model_nivel_1, train_generator_nivel_1, validation_generator_nivel_1)
    end_time = time.time()

    plot_training_history(history_nivel_1, "5Clases")
    y_score_1, y_test_1 = get_pred_labels(model_nivel_1, validation_generator_nivel_1, 5)

    cnf_matrix = confusion_matrix(y_test_1, y_score_1)
    print("Matriz de confusión:", cnf_matrix)
    
    plot_confusion_matrix(cnf_matrix, "CincoClases", classes=['normal', 'diabetes', 'glaucoma', 'miopia', 'other'], normalize=True, title='Normalized confusion matrix')
    print(f"Total training time: {(end_time - start_time) / 3600} hours")

    # Guardar predicciones con etiquetas reales para 30 imágenes
    save_predictions_with_labels(model_nivel_1, 'DataTrainCascada', 'Predicciones', num_classes, max_images=30)
