import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#load the data
data = datasets.mnist.load_data()

#split the training data into 75% train and 25% test.
#  first generate random mask for splitting data (True=train, False=test)
mask = np.full(len(data[0][0]),True)
mask[:int(len(data[0][0])/4)] = False
np.random.shuffle(mask)

train_images = data[0][0][mask]
train_labels = data[0][1][mask]
test_images = data[0][0][mask==False]
test_labels = data[0][1][mask==False]

#reshape data appropriate for CNN
train_images = np.reshape(train_images, (len(train_images),28,28,1))
test_images = np.reshape(test_images, (len(test_images),28,28,1))

#normalize data between 0 and 1
N = data[0][0].max()
train_images = train_images/N
test_images = test_images/N

#setup the model
model = models.Sequential()
model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (2, 2), activation='relu'))
#fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

#compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#train the model
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
#measure losses and accuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)

#plot results
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

