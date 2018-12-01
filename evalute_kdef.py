import numpy as np
from data_loader_kdef import data_loader
import tensorflow as tf
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

from keras.callbacks import Callback

class WeightsSaver(Callback):
    def __init__(self, model, N):
        self.model = model
        self.N = N
        self.epoch = 0

    def on_batch_end(self, epoch, logs={}):
        if self.epoch % self.N == 0:
            name = 'checkpoint/weights%08d.h5' % self.epoch
            self.model.save_weights(name)
        self.epoch += 1

input_image_size = (127, 94)
label_indices = {"neutral":0, "angry":1, "contempt":2, "disgusted":3, "fearful":4, "happy":5, "sad":6, "surprised":7}
labels_ordered = list(label_indices)
num_classes = len(label_indices)

channel_means = np.array([147.12697, 160.21092, 167.70029])
train_test_split = 0.7

data = data_loader(label_indices = label_indices, 
           channel_means = channel_means, 
           train_test_split = train_test_split,
           input_image_size = input_image_size, 
           data_path = '../data')

# train_data_dir = "data/train"
# validation_data_dir = "data/val"
# nb_train_samples = 4125
# nb_validation_samples = 466 
batch_size = 16
epochs = 200
model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (input_image_size[0], input_image_size[1], 3))
# for layer in model.layers[:5]:
#     layer.trainable = False
x = model.output
x = Flatten()(x)
# x = Dense(50, activation="relu")(x)
# x = Dense(50, activation="relu")(x)
# # x = Dropout(0.5)(x)
# x = Dense(20, activation="relu")(x)

# x = Dense(100, activation="relu")(x)
# x = Dense(200, activation="relu")(x)

#     top_layer_model.add(Dense(256, input_shape=(512,), activation='relu'))
#     top_layer_model.add(Dense(256, input_shape=(256,), activation='relu'))
#     top_layer_model.add(Dropout(0.5))
#     top_layer_model.add(Dense(128, input_shape=(256,)))
#     top_layer_model.add(Dense(NUM_CLASSES, activation='softmax'))


predictions = Dense(num_classes, activation="softmax")(x)
model_final = Model(input = model.input, output = predictions)
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), metrics=["accuracy"])

train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

test_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.0,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

# train_generator = train_datagen.flow_from_directory(
# train_data_dir,
# target_size = (input_image_size[0], input_image_size[1]),
# batch_size = batch_size, 
# class_mode = "categorical")

# validation_generator = test_datagen.flow_from_directory(
# validation_data_dir,
# target_size = (input_image_size[0], input_image_size[1]),
# class_mode = "categorical")

# checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
# early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

# filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]

# model_final.fit_generator(
# train_datagen.flow(data.train.X, data.train.y, batch_size=32),steps_per_epoch=len(data.train.X) / 32, epochs=epochs,callbacks=[WeightsSaver(model, 20)])

# model_final.fit(data.train.X, data.train.y,epochs=epochs)
model_final.fit(data.train.X, data.train.y,
                        # validation_data=(data.train.X, data.train.y),
                        nb_epoch=epochs, batch_size=batch_size,callbacks=[WeightsSaver(model, 20)])
    # Evaluate
score = top_layer_model.evaluate(data.test.X,
                                     data.test.y, batch_size=batch_size)

# samples_per_epoch = nb_train_samples,
# epochs = epochs,
# validation_data = validation_generator,
# nb_val_samples = nb_validation_samples,
# callbacks = [checkpoint, early])
# model.compile
# batches=0
# for e in range(epochs) :
#     print('Epoch',e)
    # batches = 0
    
    # batches += 1
# for x_batch,y_batch in train_datagen.flow(data.train.X,data.train.y,batch_size=batch_size) :
    # print("here")
    # model_final.save_weights('checkpoint1.hdf5')
        # if batches >= len(data.train.X) / 32 :
        #     # we need to break the loop by hand because
        #     # the generator loops indefinitely
        #     break

