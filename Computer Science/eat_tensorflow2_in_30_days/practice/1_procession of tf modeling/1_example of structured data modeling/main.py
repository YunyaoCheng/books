#一，准备数据
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

dftrain_raw = pd.read_csv('./data/titanic/train.csv')
dftest_raw = pd.read_csv('./data/titanic/test.csv')
dftrain_raw.head(10)

ax = dftrain_raw['Survived'].value_counts().plot(kind = 'bar', 
    figsize = (12, 8), fontsize = 15, rot = 0)
ax.set_xlabel('Servived', fontsize = 15)
ax.set_ylabel('Counts', fontsize = 15)
plt.show()

ax = dftrain_raw['Age'].plot(kind = 'hist',
    bins = 20, color = 'purple', figsize = (12, 8), fontsize = 15)
ax.set_xlabel('Age', fontsize = 15)
ax.set_ylabel('Frequency', fontsize = 15)
plt.show()

ax = dftrain_raw.query('Survived == 0')['Age'].plot(kind = 'density', 
    figsize = (12, 8), fontsize = 15)
dftrain_raw.query('Survived == 1')['Age'].plot(kind = 'density',
    figsize = (12, 8), fontsize = 15) #不需要再ax=，因为ax就是定义的画布
ax.legend(['Survived==0','Survived==1'],fontsize = 12) #legend可以直接访问查询
ax.set_ylabel('Density',fontsize = 15)
ax.set_xlabel('Age',fontsize = 15)
plt.show()

def preprocessing(dfdata):

    dfresult = pd.DataFrame()

    #Pclass
    dfPclass = pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['Pclass_' + str(x) for x in dfPclass.columns ]
    dfresult = pd.concat([dfresult,dfPclass],axis = 1)
    
    #Sex
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult, dfSex], axis = 1)

    #Age
    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')
    
    #SibSp, Parch, Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    #Carbin
    dfresult['Cabin_null'] = pd.isna(dfdata['Cabin']).astype('int32')

    #Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'], dummy_na = True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult, dfEmbarked], axis = 1)
    return dfresult

x_train = preprocessing(dftrain_raw)
y_train = dftrain_raw['Survived'].values

x_test = preprocessing(dftest_raw)
y_test = dftest_raw['Survived'].values

print("x_train.shape =", x_train.shape )
print("x_test.shape =", x_test.shape )

#二，定义模型
tf.keras.backend.clear_session() #清空

model = models.Sequential()
model.add(layers.Dense(20,activation = 'relu',input_shape=(15,)))
model.add(layers.Dense(10,activation = 'relu' ))
model.add(layers.Dense(1,activation = 'sigmoid' ))

print(model.summary())

#三，训练模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['AUC'])

history = model.fit(x_train, 
                    y_train,
                    batch_size=64,
                    epochs=30,
                    validation_split=0.2)

#四，评估模型
def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-') 
    plt.title('Training and validation' + metric)
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()   

plot_metric(history, 'loss')   

plot_metric(history, 'AUC')

model.evaluate(x = x_test, y = y_test) #在测试集上的评价，loss和AUC

#五，使用模型
model.predict(x_test[:10])

model.predict_classes(x_test[:10])

#六，保存模型
##1，Keras方式保存，仅兼容python
##H5 格式文件保存的是： Model stucture 和 Model weights
##JSON 和 YAML 格式问价保存的是： Model stucture
##保存模型
model.save('./practice/1_procession of tf modeling/' +
    '1_example of structured data modeling/keras_model.h5')
del model

##加载模型
model = models.load_model('./practice/1_procession of tf modeling/' +
    '1_example of structured data modeling/keras_model.h5')
model.evaluate(x_test, y_test)

##保存模型结构
json_str = model.to_json()
model_json = models.model_from_json(json_str)

##保存权重
model.save_weights('./practice/1_procession of tf modeling/' +
    '1_example of structured data modeling/keras_model_weight.h5')

##加载模型结构
model_json = models.model_from_json(json_str)
model_json.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['AUC'])

##加载模型权重
model_json .load_weights('./practice/1_procession of tf modeling/' +
    '1_example of structured data modeling/keras_model_weight.h5')
model_json.evaluate(x_test, y_test)

##2，Tensorflow原生方式
model.save_weights('./practice/1_procession of tf modeling/' +
    '1_example of structured data modeling/tf_model_weights.ckpt',
    save_format='tf')

model.save('./practice/1_procession of tf modeling/' +
    '1_example of structured data modeling/tf_model_savedmodel',
    save_format='tf')
print('export saved model.')

model_loaded = tf.keras.models.load_model('./practice/1_procession of tf modeling/' +
    '1_example of structured data modeling/tf_model_savedmodel')
model_loaded.evaluate(x_test, y_test)