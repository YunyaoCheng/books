#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets,layers,models

#一，准备数据
## http://d0evi1.com/tensorflow/datasets_performance/ 介绍了pipline
BATCH_SIZE = 100

def load_image(img_path, size=(32,32)):
    label = tf.constant(1,tf.int8) \
        if tf.strings.regex_full_match(img_path,".*automobile.*") \
            else tf.constant(0,tf.int8)
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img) #注意此处为jpeg格式
    img = tf.image.resize(img,size)/255.0
    return(img,label)

## num_parallel_calls一般设置为cpu核心数量
ds_train = tf.data.Dataset.list_files('./data/cifar2/train/*/*.jpg') \
    .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
            .prefetch(tf.data.experimental.AUTOTUNE)
ds_test = tf.data.Dataset.list_files('./data/cifar2/test/*/*.jpg') \
    .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .batch(BATCH_SIZE) \
            .prefetch(tf.data.experimental.AUTOTUNE)

plt.figure(figsize=(8,8))
for i,(img,label) in enumerate(ds_train.unbatch().take(9)):
    ax = plt.subplot(3,3,i+1)
    ax.imshow(img.numpy())
    ax.set_title("label = %d" % label)
    ax.set_xticks([])
    ax.set_yticks([])
#plt.show()

for x, y in ds_train.take(1):
    print(x.shape, y.shape)

#二，定义模型
tf.keras.backend.clear_session()

inputs = layers.Input(shape=(32,32,3))
x = layers.Conv2D(32, kernel_size=(3,3))(inputs) # 896=3*32*9+32
x = layers.MaxPool2D()(x)
x = layers.Conv2D(64, kernel_size=(5,5))(x) # 51264=32*64*25+64
x = layers.MaxPool2D()(x)
x = layers.Dropout(rate=0.1)(x)
x = layers.Flatten()(x) # 1600=5*5*64
x = layers.Dense(32, activation='relu')(x) # 51232=1600*32*1+32
outputs = layers.Dense(1, activation='sigmoid')(x) # 33=32+1

model = models.Model(inputs=inputs, outputs=outputs)

model.summary()
#三，训练模型
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join('data', 'autograph', )

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    logdir, 
    histogram_freq=1
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.binary_crossentropy,
    metrics=['accuracy']
)

history = model.fit(
    ds_train,
    epochs=1,
    validation_data=ds_test,
    callbacks=[tensorboard_callback],
    workers=4
)
#四，评估模型
from tensorboard import notebook
notebook.list()


#五，使用模型
#六，保存模型
