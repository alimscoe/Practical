#!/usr/bin/env python
# coding: utf-8

# # Title of the Assignment 01

# Linear regression by using Deep Neural network: Implement Boston
# housing price prediction problem by Linear regression using Deep Neural network. Use Boston
# House price prediction dataset.

# Step 1: Load the dataset

# In[6]:


import pandas as pd


# In[7]:


df= pd.read_csv("C:/Users/ALIM/OneDrive/Documents/archive[1]/HousingData.csv")


# In[8]:


df.head()


# Step 2: Preprocess the data

# In[9]:


from sklearn.preprocessing import StandardScaler


# In[10]:


X = df.drop("MEDV", axis=1)


# In[11]:


y= df["MEDV"]


# In[12]:


scalar = StandardScaler() # sacale the input features scaler


# In[13]:


scalar.fit_transform(X)


# Step 3: Split the dataset

# In[14]:


from sklearn.model_selection import train_test_split


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[18]:


print(X_train.shape,y_train.shape) 
print(X_test.shape,y_test.shape)


# Step 4: Define the model architecture

# In[19]:


from keras.layers import Dense, Dropout


# In[20]:


from keras.models import Sequential


# In[21]:


model = Sequential() # define model architecture


# In[22]:


model.add(Dense(64, input_dim=13, activation="relu"))


# In[23]:


model.add(Dropout(0.2))


# In[24]:


model.add(Dense(32, activation="relu"))


# In[25]:


model.add(Dense(1))


# In[26]:


print(model.summary) #display model summery


# Step 5: Compile the model

# In[27]:


import tensorflow as tf


# In[28]:


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
loss=tf.keras.losses.mean_squared_error,
metrics=[tf.keras.metrics.mean_absolute_error])


# Step 6: Train the model

# In[29]:


from keras.callbacks import EarlyStopping


# In[30]:


earlystopping = EarlyStopping(monitor="val_loss",patience=5)


# In[31]:


history = model.fit(X_train, y_train,validation_split=0.2, 
                    epochs=100, batch_size=32, callbacks=[earlystopping])


# In[32]:


import matplotlib.pyplot as plt 


# In[33]:


plt.plot(history.history["loss"])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss') 
plt.legend(['Training','Validation'])
plt.show()


# # Title of the Assignment: 03
#     
# 

# Convolutional neural network (CNN).Use MNIST Fashion Dataset and
# create a classifier to classify fashion clothing into categories.

# In[34]:


from tensorflow import  keras


# In[35]:


import numpy as np 


# In[36]:


import matplotlib.pyplot as plt


# In[37]:


## Load the dataset


# In[38]:


fasion_mnist = keras.datasets.fashion_mnist


# In[39]:


(train_image,train_labels),(test_image,test_labels) = fasion_mnist.load_data()


# In[40]:


#Normalize the images


# In[41]:


train_image = train_image/255.0


# In[42]:


test_image = test_image/255.0


# In[43]:


# Define the model


# In[44]:


model = keras.Sequential([
keras.layers.Flatten(input_shape=(28, 28)),
keras.layers.Dense(128, activation='relu'),
keras.layers.Dense(10, activation='softmax')])


# In[45]:


# compile model


# In[46]:


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


# In[47]:


# train model


# In[48]:


model.fit(train_image,train_labels,epochs=10)


# In[49]:


# Evaluate the model


# In[50]:


test_loss,test_acc = model.evaluate(test_image,test_labels)


# In[51]:


print("Test Loss: ", test_loss,"Test Acc:", test_acc)


# In[52]:


# Make predictions


# In[53]:


prediction = model.predict(test_image)


# In[54]:


#  Show some example images and their predicted labels


# In[55]:


num_rows=5
num_cols=5


# In[56]:


num_images = num_rows * num_cols


# In[57]:


plt.figure(figsize=(2*4*num_cols,4*num_rows))


# In[71]:


for i in range(num_images):
    plt.subplot(num_rows,2*num_cols,2*i+1)
    plt.imshow(test_image[i], cmap= 'gray')
    plt.axis('off')
    
    plt.subplot(num_rows,2*num_cols,2*i+2)
    plt.bar(range(10),prediction[i])
    plt.xticks(range(10))
    
    
  

   
 




# # Assignment no 2

# In[92]:


import pandas as pd


# In[94]:


df=pd.read_csv("C:/Users/ALIM/OneDrive/Desktop/Data Sets/imdb_top_1000.csv")


# In[95]:


df


# In[96]:


import numpy as np


# In[97]:


from keras.datasets import imdb


# In[98]:


from keras.utils import pad_sequences


# In[99]:


from keras.utils import pad_sequences


# In[100]:


from keras.models import Sequential


# In[101]:


from keras.layers import Embedding,Bidirectional,LSTM, Dense


# In[102]:


(X_train,y_train),(X_test,y_test) = imdb.load_data()


# In[103]:


max_len = 250


# In[104]:


X_train = pad_sequences(X_train,maxlen = max_len)


# In[105]:


X_test = pad_sequences(X_test,maxlen = max_len)


# In[106]:


model = Sequential()


# In[107]:


model.add(Embedding(input_dim=10000, output_dim=128, input_length=max_len))


# In[108]:


model.add(Bidirectional(LSTM(64, return_sequences=True)))


# In[109]:


model.add(Bidirectional(LSTM(32)))


# In[110]:


model.add(Dense(1, activation='sigmoid'))


# In[111]:


# Compile the model


# In[112]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[113]:


# Train the model


# In[116]:


history = model.fit(X_train, y_train, epochs=100, batch_size=128, validation_split=0.2)


# In[108]:


loss, acc = model.evaluate(X_test, y_test, batch_size=128)
print(f'Test accuracy: {acc:.4f}, Test loss: {loss:.4f}')


# In[ ]:




