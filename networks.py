import tensorflow as tf
from functools import partial
import keras_tuner as kt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#Function that encodes the sequence in order to feed the network
def sentence_encoder(data, vocab_size=1000):

    encoder_0 = tf.keras.layers.TextVectorization(max_tokens=vocab_size, standardize=None)
    encoder_0.adapt(data)

    return encoder_0


#Function return a model given a tuner and its parametes
def get_model(tuner, best_hp):
    # Build the model with the optimal hyperparameters
    model= tuner.hypermodel.build(best_hp)
    print(model.summary())

    return model


#Function that return test results given a trained model
def get_test_metrics(model, test_data, test_label):
    test_loss, test_acc = model.evaluate(test_data, test_label)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)

    result = model.predict(test_data)
    result = np.where(result > 0.5, 1, 0)

    print(classification_report(test_label, result))

    print(confusion_matrix(test_label, result))


def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])


def plot_acuracy_loss(history):
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plot_graphs(history, 'accuracy')
    plt.ylim(None, 1)
    plt.subplot(1, 2, 2)
    plot_graphs(history, 'loss')
    plt.ylim(0, None)


### CNN functions ###

#Function builds CNN model to tuner
def model_builder(encoder, hp):

    # hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    hp_dropout_rate = hp.Float('dropout_1', min_value=0.0, max_value=0.5, default=0.25, step=0.05)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Conv1D(filters=32, kernel_size=8, activation='relu'),
    tf.keras.layers.MaxPool1D(pool_size=2),
    tf.keras.layers.Dropout(rate=hp_dropout_rate),
    # tf.keras.layers.Dense(units=hp_units, activation='relu'), #>>>>>Hiperparametro
    tf.keras.layers.GlobalMaxPool1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])


    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate), #>>>>>Hiperparametro
              metrics=['accuracy'])

    return model


### Tuner functions ###

#Funtion creates Keras Tuner 
def network_tuner(encoder, project_name):
    build_model = partial(model_builder, encoder)

    # Instantiate the tuner
    tuner = kt.Hyperband(build_model, # the hypermodel
                        objective='val_accuracy', # objective to optimize
    max_epochs=50,
    factor=3, # factor which you have seen above 
    directory='tuner', # directory to save logs 
    project_name=project_name)

    return tuner

#Function searches best network
def search_network(tuner, train_data, train_labels, valid_data, valid_labels):

    #Print search space
    print(tuner.search_space_summary())

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    # Perform hypertuning
    tuner.search(train_data, train_labels, epochs=50, validation_data = (valid_data, valid_labels), callbacks=[stop_early])
    
    # Getting the best hyperparameters
    best_hp = tuner.get_best_hyperparameters()[0]

    return best_hp 

