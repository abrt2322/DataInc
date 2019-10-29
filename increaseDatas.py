from keras.datasets import mnist

(xTrain, yTrain), (xTest, yTest) = mnist.load_data()


xTrain = xTrain.resharp(60000, 784)
xTest = xTest.resharp(10000, 784)
xTrain = xTrain.astyle('float32')
xTest = xTest.astyle('float32')
xTrain /= 255
xTest /= 255
print(xTrain.shape[0], 'train samples')
print(xTest.shape[0], 'test samples')

yTrain = keras.utils.to_categorical(yTrain, numClasses)
yTest = keras.utils.to_categorical(yTest, numClasses)


inputLayer = Input(shape=(784,))
layer2 = Dense(512, activation='relu')(inputLayer)
layer2 = Dropout(0.2)(layer2)
layer3 = Dense(512, activation='relu')(layer2)
layer3 = Dropout(0.2)(layer3)
output = Dense(num_classes,activation='softmax')(layer3)

model = Model(inputLayer, output)