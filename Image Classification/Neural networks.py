#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from keras.utils import np_utils


# In[2]:


train=[]
lab=[]
A=r'C:\Users\kkbal\OneDrive\Desktop\kk\Kaggle competition'
for i in os.listdir(A):
    if i.startswith('Cat'):
        new_path=os.path.join(A,i)        
        for j in os.listdir(new_path):
            if j.startswith('train'):
                new_pa=os.path.join(new_path,j)
                for k in os.listdir(os.path.join(A,new_path,new_pa)):
                        label=k.split('.')[0]
                        lab.append(label)
                        if k.endswith('jpg'):
                            im=cv2.imread(os.path.join(new_pa,k))
                            image=cv2.resize(im,(64,64))
                            train.append(image)


# In[216]:


from sklearn.preprocessing import LabelEncoder
encode=LabelEncoder()
label=encode.fit_transform(lab)
print(pd.unique(lab))
print(pd.unique(label))


# In[4]:


labels=np_utils.to_categorical(label,2)
labels.shape


# In[5]:


print(labels)


# In[6]:


test=[]
B=r'C:\Users\kkbal\OneDrive\Desktop\kk\Kaggle competition'
for i in os.listdir(B):
    if i.startswith('Cat'):
        new=os.path.join(B,i)
        for j in os.listdir(os.path.join(B,new)):
            if j.startswith('test1'):
                new_path=os.path.join(new,j)
                for k in os.listdir(os.path.join(new,new_path)):
                    if k.endswith('jpg'):
                        img=cv2.imread(os.path.join(new_path,k))
                        image=cv2.resize(img,(64,64))
                        test.append(image)


# In[7]:


test=np.array(test)
test.shape


# In[8]:


train=np.array(train)
print(train.shape)
print(labels.shape)


# In[9]:


index=75
plt.imshow(train[index])


# In[10]:


from sklearn.model_selection import train_test_split
X_train,Y_train,X_test,Y_test=train_test_split(train,labels,random_state=42,test_size=0.05)


# In[11]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[12]:


xtrain=X_train.reshape(X_train.shape[0],-1).T.astype('float32')
ytrain=Y_train.reshape(Y_train.shape[0],-1).T.astype('float32')
xtest=X_test.reshape(X_test.shape[0],-1).T.astype('float32')
ytest=Y_test.reshape(Y_test.shape[0],-1).T.astype('float32')
print(xtrain.shape)
print(xtest.shape)
print(ytest.shape)
print(ytrain.shape)


# In[13]:


x=xtrain/225
y=ytrain/225
x.shape


# In[14]:


xtrain=tf.convert_to_tensor(xtrain,tf.float32)
xtest=tf.convert_to_tensor(xtest,tf.float32)


# In[15]:


def initialize_parameters(xtrain):
    np.random.seed(1)
    w1=tf.Variable(np.random.randn(25,xtrain.shape[0]),dtype='float32')
    b1=tf.Variable(tf.zeros((25,1)))
    w2=tf.Variable(np.random.randn(12,25),dtype='float32')
    b2=tf.Variable(tf.zeros((12,1)))
    w3=tf.Variable(np.random.rand(2,12),dtype='float32')
    b3=tf.Variable(tf.zeros((2,1)))
    
    parameters={
        'W1':w1,
        'B1':b1,
        'W2':w2,
        'B2':b2,
        'W3':w3,
        'B3':b3}
    return parameters


# In[16]:


parameters=initialize_parameters(xtrain)


# In[17]:


def forward_propagation(xtrain,parameters):
    w1=parameters['W1']
    b1=parameters['B1']
    w2=parameters['W2']
    b2=parameters['B2']
    w3=parameters['W3']
    b3=parameters['B3']
    z1=tf.add(tf.matmul(w1,xtrain),b1)
    A1=tf.keras.activations.relu(z1)
    z2=tf.add(tf.matmul(w2,A1),b2)
    A2=tf.keras.activations.relu(z2)
    z3=tf.add(tf.matmul(w3,A2),b3)
    
    return z3


# In[18]:


z3=forward_propagation(xtrain,parameters)


# In[19]:


def cost_function(z3,parameters,xtest):
    z3=forward_propagation(xtrain,parameters)
    cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=z3, labels=xtest))
    return cost


# In[20]:


def model(x,xtest,y,ytest,learning_rate):
    
    n_x,m=xtrain.shape
    
    ## Intializing parameters
    
    parameters=initialize_parameters(xtrain)
    
    ## forward propagation
    
    forward_propagation(xtrain,parameters)
    
    ## computing cost
    
    cost_function(xtrain,parameters,xtest)
    
    ## Backward propagation
    opt=tf.keras.optimizers.Adam()
    cost=[]
    for i in range(50):
        opt.minimize(lambda:cost_function(xtrain,parameters,xtest),var_list=[parameters])
        cost.append(cost_function(xtrain,parameters,xtest))
        print(cost_function(xtrain,parameters,xtest))
    parameters=parameters
    plt.plot(np.squeeze(cost))
    plt.ylabel('Cost')
    plt.title('learning rate='+str(learning_rate))
    plt.xlabel('number of iterations')
    plt.show()
    
    return parameters


# In[21]:


parameters=model(x,xtest,y,ytest,learning_rate=0.000001)


# In[22]:


def prediction(parameters,xtest,xtrain):
    z3=forward_propagation(xtrain,parameters)
    predict=tf.keras.activations.sigmoid(z3)
    correct_prediction=tf.equal(tf.argmax(predict,0),tf.argmax(xtest,0))
    print(tf.argmax(predict,0),tf.argmax(xtest,0))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))
    return accuracy.numpy()


# In[23]:


prediction(parameters,xtest,xtrain)


# In[24]:


prediction(parameters,ytest,ytrain)


# In[142]:


## Prediction
index=test[110]
plt.imshow(index)
sh=index/225
sh=sh.reshape((1,64*64*3)).T.astype('float32')
print(sh.shape)
z3=forward_propagation(sh,parameters)
z3=tf.keras.activations.sigmoid(z3)
predict=tf.argmax(z3,0)
np.squeeze(predict)


# In[26]:


index=test[100]
plt.imshow(index)
index=index.reshape(1,64*64*3).T.astype('float32')
z3=forward_propagation(index,parameters)
z3=tf.keras.activations.sigmoid(z3)
tf.argmax(z3).numpy()


# In[27]:


## Sequential modelling

from keras.layers import Dense,BatchNormalization
from keras.models import Sequential,load_model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras import regularizers


# In[28]:


xtrain=X_train.reshape(X_train.shape[0],-1)
ytrain=Y_train.reshape(Y_train.shape[0],-1)
xtest=X_test.reshape(X_test.shape[0],-1)
ytest=Y_test.reshape(Y_test.shape[0],-1)
print(xtrain.shape)
print(xtest.shape)
print(ytest.shape)
print(ytrain.shape)


# In[29]:


xtrain=xtrain/225
ytrain=ytrain/225


# In[30]:


model=Sequential()
model.add(Dense(100,activation='relu',input_shape=(12288,)))
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(2,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[31]:


model.summary()


# In[32]:


model.fit(xtrain,xtest,epochs=32)


# In[33]:


model.evaluate(xtrain,xtest)


# In[34]:


### Trying different activation functions
def model(activation_function):
    
    activation_result={}
    for activation_function in activation_function:
        model=Sequential()
        model.add(Dense(100,activation=activation_function,input_shape=(12288,)))
        model.add(Dense(100,activation=activation_function))
        model.add(Dense(50,activation=activation_function))
        model.add(Dense(2,activation='sigmoid'))
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        history=model.fit(xtrain,xtest,validation_data=(ytrain,ytest),epochs=32)
        activation_result[activation_function]=history
    val_loss={k:v.history['val_loss'] for k,v in activation_result.items()}
    val_loss_dataframe=pd.DataFrame(val_loss)
    val_loss_dataframe.plot(title='Loss Per Activation Function')


# In[35]:


activation_function=['tanh','sigmoid','relu']
model(activation_function)


# In[36]:


## It looks likte relu activation function works better


# In[37]:


early_stopping=EarlyStopping(monitor='val_loss',patience=5)
model_save=ModelCheckpoint('best_model.hdf5',save_best_only=True)


# In[38]:


## Let's build a model with normalized batches.
model=Sequential()
model.add(Dense(100,activation='relu',input_shape=(12288,)))
model.add(BatchNormalization())
model.add(Dense(100,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(50,activation='relu'))
model.add(Dense(2,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()


# In[39]:


history=model.fit(xtrain,xtest,validation_data=(ytrain,ytest),epochs=64,callbacks=[early_stopping])


# In[40]:


## Plotting the loss function
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('loss for training and testing data')
plt.legend(['Train','Test'])
plt.show()


# In[41]:


model=Sequential()
model.add(Dense(100,activation='relu',input_shape=(12288,),kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Dense(100,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(50,activation='relu'))
model.add(Dense(2,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()


# In[42]:


history=model.fit(xtrain,xtest,validation_data=(ytrain,ytest),epochs=64,callbacks=[early_stopping])


# In[43]:


plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.legend(['Test','Train'])
plt.title('loss for training and testing data with regularization')
plt.show()


# In[44]:


model=Sequential()
model.add(Dense(100,activation='relu',input_shape=(12288,)))
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(2,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(xtrain,xtest,validation_data=(ytrain,ytest),epochs=64,callbacks=[early_stopping,model_save])


# In[45]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss in and validation set')
plt.legend(['Train','Test'])
plt.show()


# In[46]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')


# In[47]:


## Using the best model 
model.fit(xtrain,xtest,validation_data=[ytrain,ytest],callbacks=[model_save])


# In[48]:


model.evaluate(xtrain,xtest)


# In[55]:


### Using mini batch to fit the model
model=Sequential()
model.add(Dense(100,activation='relu',input_shape=(12288,)))
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(2,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()


# In[56]:


history=model.fit(xtrain,xtest,validation_data=(ytrain,ytest),epochs=64,callbacks=[early_stopping],batch_size=2048)


# In[57]:


plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.show()


# In[58]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()


# In[59]:


## Loading the best model
best=load_model('best_model.hdf5')


# In[60]:


def best_model(optimizer='adam',activation='relu'):
    best.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return best


# In[61]:


### Hyper parameter tuning:
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV,cross_val_score


model=KerasClassifier(build_fn=best_model,epochs=32,batch_size=1024,verbose=0)


# In[62]:


optimizers=['sgd','adam','RMSprop']
epoch=[10,12,15]
batch=[500,100,1500]
act=['relu','tanh']
params=dict(optimizer=optimizers,epochs=epoch,batch_size=batch,activation=act)
params


# In[63]:


random_search=RandomizedSearchCV(model,params,cv=3)
random_results=random_search.fit(xtrain,xtest)


# In[64]:


random_results


# In[65]:


print(random_results.best_params_)
print(random_results.best_score_)


# In[67]:


best.summary()


# In[221]:


## Predicting the test data:
test.shape


# In[222]:


## Converting the pixel values
testing=test.reshape(test.shape[0],-1)
print(testing.shape)


# In[223]:


pred=best.predict(testing)
final=tf.argmax(pred,1)
new=pd.DataFrame(final)


# In[224]:


final_new=new.rename(columns={0:'labels'})
final_new.groupby(['labels']).labels.count()


# In[225]:


pic=12499
check=testing[pic]
check=check.reshape(check.shape[0],-1).T
pred=best.predict(check)
pred_1=tf.argmax(pred,1).numpy()
if pred_1==1:
    print('It is a Dog!')
else:
    print('It is a Cat!')
plt.imshow(test[pic])
plt.show()


# In[226]:


final=best.evaluate(xtrain,xtest)[1]
print('Final Model has accuracy of',final)


# In[227]:


Predictions=final_new['labels'].replace({0:'Cat',1:'Dog'})
Predictions


# In[ ]:




