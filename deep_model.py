from keras.layers import Dense, Conv1D, Flatten, BatchNormalization, MaxPool1D
from keras.models import Sequential
from keras.optimizers import Adam

class DeepModel:
    """docstring for DeepModel."""
    model = Sequential()

    def __init__(self):
        self.model.add(Conv1D(filters=512, kernel_size=5, padding='same', input_shape=(250,6)))
        self.model.add(Conv1D(filters=512, kernel_size=5, padding='same'))
        self.model.add(Conv1D(filters=512, kernel_size=5, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool1D())
        self.model.add(Conv1D(filters=512, kernel_size=5, padding='same'))
        self.model.add(Conv1D(filters=512, kernel_size=5, padding='same'))
        self.model.add(Conv1D(filters=512, kernel_size=5, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool1D())
        self.model.add(Conv1D(filters=512, kernel_size=5, padding='same'))
        self.model.add(Conv1D(filters=512, kernel_size=5, padding='same'))
        self.model.add(Conv1D(filters=512, kernel_size=5, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool1D())
        self.model.add(Conv1D(filters=512, kernel_size=5, padding='same'))
        self.model.add(Conv1D(filters=512, kernel_size=5, padding='same'))
        self.model.add(Conv1D(filters=512, kernel_size=5, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool1D())
        self.model.add(Conv1D(filters=512, kernel_size=5, padding='same'))
        self.model.add(Conv1D(filters=512, kernel_size=5, padding='same'))
        self.model.add(Conv1D(filters=512, kernel_size=5, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool1D())
        self.model.add(Flatten())
        self.model.add(Dense(1, activation='relu'))


        optimizer = Adam(lr=0.0001, decay=0.000001)
        self.model.compile(optimizer=optimizer,
                           loss='mean_squared_error',
                           metrics=['mse'])

    def train(self, input_data, output_label, test_data, test_label):
        self.model.fit(input_data, output_label, validation_data=[test_data, test_label], epochs=100, batch_size=100)

    def predict(self, input_instance):
        self.model.predict([input_instance])
