import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape


data = pd.read_csv(r"C:\Users\Hp\Desktop\Suicidal_redit.csv", encoding="utf-8")
print(data.head())

posts = data['Post'].astype(str).tolist()
labels = data['Label'].astype(str).tolist()


y = np.array([1 if label.lower() == 'suicidal' else 0 for label in labels])


X_train, X_test, y_train, y_test = train_test_split(posts, y, test_size=0.2, random_state=42)


max_features = 10000
max_sequence_length = 100
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding='post')


embedding_dim = 100
input_shape = (max_sequence_length, embedding_dim, 1)


model = Sequential()

model.add(Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(Reshape((max_sequence_length, embedding_dim, 1)))  # for Conv2D compatibility


model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))


model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))


model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))


model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))


model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.build(input_shape=(None, max_sequence_length))

model.summary()


batch_size = 32
epochs = 5

history = model.fit(X_train_pad, y_train,
                    validation_data=(X_test_pad, y_test),
                    epochs=epochs,
                    batch_size=batch_size)


loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")
from sklearn.metrics import classification_report


y_pred_prob = model.predict(X_test_pad)


y_pred = (y_pred_prob > 0.5).astype(int)


print(classification_report(y_test, y_pred))
