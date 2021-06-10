import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt

def main():
  batch_size = 128 #number of samples per gradient
  num_classes = 10 #number of output classes
  epochs = 12 #number of iterations

  # input image dimensions
  img_rows, img_cols = 28, 28 #dimensions of the input image

  # the data, split between train and test sets
  (x_train, y_train), (x_test, y_test) = mnist.load_data() #data split between train and test sets

  if K.image_data_format() == 'channels_first': #if it is channels_first, reshape it accordingly
      x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
      x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
      input_shape = (1, img_rows, img_cols)
  else:
      x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
      x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
      input_shape = (img_rows, img_cols, 1)

  x_train = x_train.astype('float32') #setting the data type as float
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  print('x_train shape:', x_train.shape)
  print(x_train.shape[0], 'train samples')
  print(x_test.shape[0], 'test samples')

  # convert class vectors to binary class matrices
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)

  print("Running the baseline code")
  CNN(input_shape, num_classes, x_train, y_train, batch_size, epochs, x_test, y_test, (3,3),6,16)
  print("Running the code after changing the kernel size of the baseline code to (5,5) and then running the code")
  CNN(input_shape, num_classes, x_train, y_train, batch_size, epochs, x_test, y_test, (5,5),6,16)
  print("Running the code after changing the number of feature maps of the baseline code to 10 in the first convolution layer and changing the number of feature maps of the baseline code to 20 in the second convolution layer with kernel size (3 * 3)")
  CNN(input_shape, num_classes, x_train, y_train, batch_size, epochs, x_test, y_test, (3,3),10,20)

def CNN(input_shape, num_classes, x_train, y_train, batch_size, epochs, x_test, y_test, kernel_size, feature_maps1, feature_maps2):
  model = Sequential() #Creating model object and setting the parameters
  model.add(Conv2D(feature_maps1, kernel_size=kernel_size,
                  activation='relu',
                  input_shape=input_shape))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(feature_maps2, kernel_size, activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(120, activation='relu'))
  model.add(Dense(84, activation='relu'))

  model.add(Dense(num_classes, activation='softmax')) #Classification is done here

  # https://keras.io/optimizers/ 
  model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(lr=0.1, rho=0.95, decay=0.0),
                metrics=['accuracy']) #Model is compiled here

  history = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test)) #history object gets the output results
  score = model.evaluate(x_test, y_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])
  plot_graphs(history,score)

def plot_graphs(history,score): #this is a function to plot graphs
  plt.plot(history.history['loss']) #training loss plot
  plt.plot(history.history['val_loss']) #testing loss plot
  plt.title('Model Error Results')
  plt.ylabel('Error')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper right')
  plt.show()

  print('Test loss/Test error:', score[0])
  print('Test accuracy:', score[1])  

main()
