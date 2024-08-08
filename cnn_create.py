import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Defina o caminho das imagens e dos rótulos
image_dir = '/content/image'
label_dir = '/content/labels'

# Lista de nomes das categorias (ID das classes deve corresponder a essas posições)
target_names = ['Alessandro', 'Leonir', 'Thiago', 'Yago']

# Função para ler rótulos a partir dos arquivos de texto
def load_labels(label_dir):
    labels = {}
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(label_dir, label_file), 'r') as file:
                # Leia o conteúdo e pegue apenas o primeiro valor
                label = int(file.read().split()[0])
                image_file = label_file.replace('.txt', '.jpg')
                labels[image_file] = label
    return labels

# Carregar os rótulos
labels = load_labels(label_dir)

# Converter os dados em um DataFrame
data = {'filepath': [os.path.join(image_dir, filename) for filename in labels.keys()],
        'label': [label for label in labels.values()]}
df = pd.DataFrame(data)

# Dividir os dados em treino e teste
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Definir a aumentação de dados
datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    rescale=1./255
)

# Criar o gerador de dados para treinamento
train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepath',
    y_col='label',
    target_size=(300, 300),
    batch_size=32,
    class_mode='raw'
)

# Criar o gerador de dados para teste (sem aumento, apenas normalização)
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filepath',
    y_col='label',
    target_size=(300, 300),
    batch_size=32,
    class_mode='raw',
    shuffle=False
)

# Definindo o modelo
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    Dropout(0.5),
    tf.keras.layers.Dense(len(target_names), activation='softmax')  # Alterar de acordo com o número de categorias
])

# Compilando o modelo
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Treinando o modelo usando o gerador de dados
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator
)

# Avaliando o modelo
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print("\nAcurácia no conjunto de teste:", test_acc)



model.save('/content/model.h5')

# Converter o modelo Keras para TFLite
def convert_to_tflite(keras_model_path, tflite_model_path):
    # Carregar o modelo Keras
    model = tf.keras.models.load_model(keras_model_path)

    # Criar um conversor TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Converter o modelo
    tflite_model = converter.convert()

    # Salvar o modelo TFLite
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Modelo TFLite salvo em: {tflite_model_path}")

# Caminho para salvar o modelo TFLite
tflite_model_path = '/content/model.tflite'

# Converter e salvar o modelo TFLite
convert_to_tflite('/content/model.h5', tflite_model_path)
