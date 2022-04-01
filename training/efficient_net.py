# https://neptune.ai/blog/how-to-build-a-light-weight-image-classifier-in-tensorflow-keras

import os
import time
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import Model
from keras.applications.efficientnet import EfficientNetB1
from keras.layers import Dense, BatchNormalization, LeakyReLU, Softmax
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def preprocess_data(image_size, train_path):
    batch_size = 32

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=0.3,
		subset="training",
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=0.3,
		subset="validation",
    )

    return train_ds, val_ds

def preprocess_test(image_size, eval_path):
    eval_ds = tf.keras.preprocessing.image_dataset_from_directory(
        eval_path,
        seed=42,
        image_size=image_size,
        batch_size=32,
    )
    
    return eval_ds, []

def make_model(input_shape, dense_count, n_classes):
    backbone = EfficientNetB1(include_top = False,
                            input_shape = input_shape,
                            pooling = 'avg')
    
    model = Sequential()
    model.add(backbone)
    
    model.add(Dense(dense_count))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    
    model.add(Dense(n_classes))
    model.add(Softmax())
    return model


def unfreeze(model: Model):
    block_to_unfreeze_from = 5
    trainable_flag = False

    for layer in model.layers[0].layers:
        if layer.name.find('bn') != -1:
            layer.trainable = True
        else:
            layer.trainable = trainable_flag
            
        if layer.name.find(f'block{block_to_unfreeze_from}') != -1:
            trainable_flag = True
            layer.trainable = trainable_flag

    return model


def compile_model(model: Model, learning_rate) -> Model:
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), 
        loss='sparse_categorical_crossentropy',
        metrics=['acc']
    )
    return model


def train(model: Model, train_ds, val_ds, epochs, save_dir, log_dir=None):
    history = model.fit(
        train_ds, 
        epochs=epochs, 
        validation_data=val_ds, 
        use_multiprocessing = True,
        workers = 11
        )

    model.save(save_dir)
    train_acc_1, train_loss_1, val_acc_1, val_loss_1 = show_history(history)

    # unfreezing all layers in CNN
    for layer in model.layers:
        layer.trainable = True

    history = model.fit(
        train_ds, 
        epochs=epochs, 
        validation_data=val_ds, 
        use_multiprocessing = True,
        workers = 11
        )
    train_acc_2, train_loss_2, val_acc_2, val_loss_2 = show_history(history)
    model.save(save_dir)
    
    return model, train_acc_1, train_loss_1, val_acc_1, val_loss_1, train_acc_2, train_loss_2, val_acc_2, val_loss_2



def show_history(history):
    print(f"train_acc: {history.history['acc']}")
    print(f"val_acc: {history.history['val_acc']}")
    print(f"train_loss: {history.history['loss']}")
    print(f"val_loss: {history.history['val_loss']}")
    
    # # summarize history for accuracy
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig("accuracy_plot.png")

    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig("loss_plot.png")
    
    return history.history['acc'], history.history['loss'], history.history['val_acc'], history.history['val_loss']


def try_model(epochs, learning_rate, n_classes, dense_count, input_shape, image_size, train_path, eval_path, save_dir, log_dir = None):
    train_ds, val_ds = preprocess_data(image_size, train_path)
    x_test, y_test = preprocess_test(image_size, eval_path)
    model = make_model(input_shape, dense_count, n_classes)
    model = unfreeze(model)
    model = compile_model(model, learning_rate)
    model.summary()
    model, train_acc_1, train_loss_1, val_acc_1, val_loss_1, train_acc_2, train_loss_2, val_acc_2, val_loss_2 = train(model, train_ds, val_ds, epochs, save_dir, log_dir)

    # Evaluating model on validation data
    score = model.evaluate(x_test)
    print(f'Test loss: {score[0]}')
    print(f'Test accuracy: {score[1]}')
    print(f'Evaluation scores: {score}')
    
    return train_acc_1, train_loss_1, val_acc_1, val_loss_1, train_acc_2, train_loss_2, val_acc_2, val_loss_2, score[0], score[1]
    

def main():
    epoch_options = [1,2,3,5,8]
    learning_rate_options = [0.05,0.01,0.001,0.0001]

    repetitions_per_model = 1
    n_classes = 4
    dense_count = 256
    input_shape = (128, 128, 3)
    image_size = (128, 128)
    train_path = "/home/janneke/Documents/Image_analysis/image_analysis_eindopdracht/data/blood_cells/ALL/"
    eval_path = "/home/janneke/Documents/Image_analysis/image_analysis_eindopdracht/data/blood_cells/EVAL/"
    log_file = "training.log"
    with open(log_file, "w") as logs:
        logs.write("datetime_start,training_time,epochs,learning_rate,repetition,train_data,train_acc_1,train_loss_1,val_acc_1,val_loss_1,train_acc_2,train_loss_2,val_acc_2,val_loss_2,test_loss,test_acc\n")

    for epoch_index, epochs in enumerate(epoch_options):
        for learning_rate_index, learning_rate in enumerate(learning_rate_options):
            for repetition in range(repetitions_per_model):
                save_dir = f'/home/janneke/Documents/Image_analysis/image_analysis_eindopdracht/training/models_run2/efficient_net_model_{epoch_index}_{learning_rate_index}_{repetition}'
                
                training_start = datetime.now()
                training_start_time = time.time()
                train_acc_1, train_loss_1, val_acc_1, val_loss_1, train_acc_2, train_loss_2, val_acc_2, val_loss_2, test_loss, test_acc = try_model(epochs, learning_rate, n_classes, dense_count, input_shape, image_size, train_path, eval_path, save_dir)
                
                with open(log_file, "a") as logs:
                    print(100*"-")
                    print(save_dir)
                    logs.write(f'{training_start},{time.time()-training_start_time},{epochs},{learning_rate},{repetition},{train_path},{train_acc_1},{train_loss_1},{val_acc_1},{val_loss_1},{train_acc_2},{train_loss_2},{val_acc_2},{val_loss_2},{test_loss},{test_acc}\n')


if __name__ == "__main__":
    main()