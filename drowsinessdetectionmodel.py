import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical

training_data = {
    "C:/Users/dhars/OneDrive/Desktop/Train/Closed_Eyes",
    "C:/Users/dhars/OneDrive/Desktop/Train/Open_Eyes"
}

def load_images(training_data):
    images = []
    labels = []
    for i, folder in enumerate(training_data): 
        for filename in os.listdir(folder):
            try:
                img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (48,48))
                images.append(img)
                labels.append(i)
            except Exception as e:
                print(f"Error loading image {os.path.join(folder,filename)}: {e}")
    return np.array(images), np.array(labels)

images, labels = load_images(training_data)
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
x_train = x_train.reshape(x_train.shape[0], 48,48).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0], 48,48).astype('float32')/255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1, validation_data=(x_test, y_test))
model.save("drowsiness_detection_model.h5")