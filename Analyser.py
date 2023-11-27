import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model

class Analyser():
    '''
        Creates an analysing tool for pandas dataframes.
        Provide the dataset in the form of a dataframe, 
        feature names as a list and target name to use.
        The tool is an easy way to create basic neural
        networks of different depths and sizes.
    '''
    def __init__(self, dataframe:pd.DataFrame, features: list, target: str):
        self.dataframe = dataframe
        self.features = features # features names, list of strings
        self.target = target # name of the target column
        self.train_test_split_done = False
        self.save_file = 'best_model'

        
        
    
    def show_boxplots(self):
        number_of_features = len(self.features)
        fig, axs = plt.subplots(2,number_of_features // 2 + 1)
        fig.set_figheight(8)
        fig.set_figwidth(number_of_features * 2)
        n = 0

        for i in range(number_of_features):
            if i % 2 == 0:
                axs[0,n].boxplot(self.dataframe[self.features[i]])
                axs[0,n].set_title(self.features[i])
            else:
                axs[1,n].boxplot(self.dataframe[self.features[i]])
                axs[1,n].set_title(self.features[i])
                n+=1
    
    def show_histograms(self):
        number_of_features = len(self.features)
        fig, axs = plt.subplots(number_of_features // 2 + 1, 2)
        fig.set_figheight(number_of_features * 2)
        fig.set_figwidth(8)
        n = 0

        for i in range(number_of_features):
            if i % 2 == 0:
                axs[n,0].hist(self.dataframe[self.features[i]])
                axs[n,0].set_title(self.features[i])
            else:
                axs[n,1].hist(self.dataframe[self.features[i]])
                axs[n,1].set_title(self.features[i])
                n+=1

    def transform_data(self):
            self.x = MinMaxScaler().fit_transform(self.dataframe[self.features].to_numpy())
            self.y = OneHotEncoder(sparse=False).fit_transform(self.dataframe[self.target].to_numpy().reshape(-1,1))

    def create_model(self, hNum:int, max_depth = 4,two_way = False, activation='relu', softmax = False):
        
        '''
        hNum - number of neurons on the biggest hidden layer
        max_depth - max number of hidden layers
        two_way - if set to True layers are built as follows: first layer is twice the size of 
               the number of features and every following layer is double that 
               until they reach hNum, from that point the layers are half te size 
               of the previous until they reach target size,
               else the network starts with the biggest layer each following layer
               is of the previous until they reach target size
        activation - activation functions as in keras
        softmax - should a softmax layer be used as the output
        '''

        self.transform_data()
       
        model = tf.keras.Sequential()
        model.add(layers.InputLayer(input_shape = self.x.shape[1]))
        depth = 1
        
        if two_way:
            i = 0
            neurons = 2 * len(self.x[0])
            while hNum >= 2 * len(self.y[0]) and depth < max_depth:
                if neurons < hNum:
                    model.add(layers.Dense(neurons, activation='relu', name=f'hidden_layer_{i}'))
                    neurons = neurons*2
                    depth += 1
                    i += 1
                else:
                    model.add(layers.Dense(hNum, activation='relu', name=f'hidden_layer_{i}'))
                    hNum = hNum/2
                    depth += 1
                    i += 1
        else:
            while hNum >= 2 * len(self.y[0]) and depth < max_depth:
                model.add(layers.Dense(hNum))
                hNum = hNum/2
                depth += 1

        model.add(layers.Dense(len(self.y[0]), name='target', activation=activation))
        
        if softmax:
            model.add(layers.Softmax())

        model.summary()

        self.model = model

    def train_model(self, 
                    epochs, 
                    optimizer='adam', 
                    loss='binary_crossentropy', 
                    metrics=['accuracy'], 
                    test_size = 0.2, 
                    are_callbacks = False,
                    save = False, 
                    save_file='best_model',
                    patience_early = 10, 
                    patience_reduce = 3,
                    monitor_early = 'val_loss', 
                    monitor_reduce = 'val_loss', 
                    reduce_factor = 0.1, 
                    min_lr=0.00001, 
                    verbose_fit=0,
                    verbose_callbacks=1):
        
        '''
        are_callbacks - should there be callbacks added to the fit
        save - should the model be saved
        save_file - name of the file to which the best model should be saved
        patience_early - patience setting for the early stopping callback
        patience_reduce - patience setting for the reduce LR callback
        monitor_early - monitor setting for the early stopping callback
        monitor_reduce - monitor setting for the reduce LR callback
        verbose_fit - verbose setting for fit method
        verbose_callbacks - verbose setting for callbacks
        all the other variables work as in: 
            train_test_split - test_size, 
            keras_model_fit - epochs, loss, metrics 
            keras_callbacks - patience, moinitor_early, monitor_reduce
                              reduce_factor, min_lr
        '''
        
        self.save_file = save_file
        self.are_callbacks = are_callbacks
        callbacks = [
                        EarlyStopping(monitor=monitor_early, patience=patience_early, restore_best_weights=True, mode='auto', verbose=verbose_callbacks),
                        ReduceLROnPlateau(monitor=monitor_reduce, factor=reduce_factor, patience=patience_reduce, min_lr=min_lr, mode="auto", verbose=verbose_callbacks)
                    ]
        if not self.train_test_split_done:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=test_size)
            self.train_test_split_done = True

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        if are_callbacks:
            print('Training model with callbacks.')
            self.results = self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=epochs, callbacks=callbacks, verbose=verbose_fit)
        else:
            print('Training model.')
            self.results = self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=epochs, verbose=verbose_fit)
            print('History saved to self.results.')
        
        if save:     
            self.model.save(save_file)
            print(f'Best model saved to {save_file}. History saved to self.results.')
        
        
    def predict(self, model):
        self.y_pred = model.predict(self.x_test)

    def show_history(self):
        for key in self.results.history.keys():
            plt.plot(self.results.history[key], label=key)
        plt.title('History')
        plt.xlabel('Epochs')
        plt.legend()

    def show_results(self, load = False, load_path = ''):
        '''
        load - should the best model be loaded from the save_file
        load_path - path if model should be loaded from a different file than the last saved
        '''
        
        if load:
            if len(load_path) > 0:
                model = load_model(load_path)
                self.predict(model)
                ConfusionMatrixDisplay.from_predictions(self.y_test.argmax(axis=1), self.y_pred.argmax(axis=1))
                print(classification_report(self.y_test.argmax(axis=1), self.y_pred.argmax(axis=1)))
            else:
                model = load_model(self.save_file)
                self.predict(model)
                ConfusionMatrixDisplay.from_predictions(self.y_test.argmax(axis=1), self.y_pred.argmax(axis=1))
                print(classification_report(self.y_test.argmax(axis=1), self.y_pred.argmax(axis=1)))
        else:    
            self.y_pred = self.model.predict(self.x_test)
            print(classification_report(self.y_test.argmax(axis=1), self.y_pred.argmax(axis=1)))
            ConfusionMatrixDisplay.from_predictions(self.y_test.argmax(axis=1), self.y_pred.argmax(axis=1));
