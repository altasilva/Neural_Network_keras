from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('admission_dataset.csv')

# y is the targed variable
y = df['Chance of Admit ']
#x are another columns
x = df.drop('Chance of Admit ', axis=1)

x_train, x_test = x[0:300], x[300:]
y_train, y_test = x[0:300], x[300:]

print(x_train.shape[1])

#Making the  neural network architecture:
model = Sequential()
model.add(Dense(units=3, activation='relu', input_dim=7))
model.add(Dense(units=1, activation='linear'))

#Training of neural network:
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
result = model.fit(x_train, y_train, epochs=200, batch_size=32, validation_data=(x_test, y_test))

#Plotting the training hitory graph
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('History of train')
plt.ylabel('Cost Function')
plt.xlabel('Epochs of train')
plt.legend(['Error train', 'Error test'])
plt.show()




