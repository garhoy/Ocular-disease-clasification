import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocessor


os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # Usar GPU 2

def setup_data_generators_second_phase(base_dir="OtherTemp", target_size=(512, 512), batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        base_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',  # Solo dos clases: ARMD y Cataratas
        subset='training',
        shuffle=True
    )

    validation_generator = datagen.flow_from_directory(
        base_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    return train_generator, validation_generator

def build_custom_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    filter_sizes = [64, 128, 256, 512]
    for filters in filter_sizes:
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    if num_classes == 2:
        outputs = Dense(1, activation='sigmoid')(x)
        loss_function = 'binary_crossentropy'
    else:
        outputs = Dense(num_classes, activation='softmax')(x)
        loss_function = 'categorical_crossentropy'
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=loss_function, metrics=['accuracy'])
    return model

def train(model, train_generator, validation_generator, model_name):
    checkpoint = ModelCheckpoint(model_name, monitor='val_accuracy', verbose=1, save_weights_only=False, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, verbose=1, mode='max', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=25, verbose=1, mode='max', min_lr=1e-6)
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=1000,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )

    return history

def get_pred_labels(model, generator, num_classes):
    steps = np.ceil(generator.samples / generator.batch_size)
    predictions = model.predict(generator, steps=steps)

    if num_classes == 2:
        y_score = (predictions > 0.5).astype(int).flatten()
    else:
        y_score = np.argmax(predictions, axis=-1)

    y_test = generator.classes

    return y_score, y_test

def train_and_evaluate_second_model(base_dir_nivel_2, target_size, batch_size, input_shape):
    # Configurar generadores de datos
    train_generator_nivel_2, validation_generator_nivel_2 = setup_data_generators_second_phase(base_dir_nivel_2, target_size=target_size, batch_size=batch_size)

    # Entrenar el segundo modelo
    num_classes = 2
    model_nivel_2 = build_custom_model(input_shape, num_classes)
    model_nivel_2.summary()

    history_nivel_2 = train(model_nivel_2, train_generator_nivel_2, validation_generator_nivel_2, 'modelo_secundario.h5')

    # Evaluar el modelo en el conjunto de validación
    y_score_2, y_test_2 = get_pred_labels(model_nivel_2, validation_generator_nivel_2, num_classes)

    for i in range(len(y_test_2)):
        pred_label = 'ARMD' if y_score_2[i] == 0 else 'Cataratas'
        real_label = 'ARMD' if y_test_2[i] == 0 else 'Cataratas'
        print(f"Predicción: {pred_label}, Etiqueta verdadera: {real_label}")

    return model_nivel_2, validation_generator_nivel_2

if __name__ == '__main__':
    base_dir_nivel_2 = 'SecondPhase'
    batch_size = 32
    target_size = (512, 512)
    input_shape = (512, 512, 3)
    #load the model
    # Segundo nivel: Clasificación de ARMD y Cataratas
    model_nivel_2, validation_generator_nivel_2 = train_and_evaluate_second_model(base_dir_nivel_2, target_size, batch_size, input_shape)
