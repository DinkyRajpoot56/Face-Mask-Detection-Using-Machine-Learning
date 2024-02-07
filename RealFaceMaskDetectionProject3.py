#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
haar_data = cv2.CascadeClassifier("C:\\Users\\kush\\Downloads\\haarcascade_frontalface_default.xml")


# In[2]:


img=cv2.imread('C:\\Users\\kush\\Desktop\\360_F_243123463_zTooub557xEWABDLk0jJklDyLSGl2jrr.jpg')


# In[3]:


haar_data.detectMultiScale(img)


# In[ ]:


capture=cv2.VideoCapture(0)
data = []
while True:
    flag, img = capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 4)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50,50))
            print(len(data))
            if len(data) < 400:
                data.append(face)
        cv2.imshow('result',img)
        if cv2.waitKey(2) == 27 or len(data) >=200:
            break
capture.release()
cv2.destroyAllWindows()


# In[5]:


import numpy as np


# In[6]:


x = np.array([3,2,54,6])


# In[7]:


x


# In[8]:


x[0:2]


# In[9]:


x=np.array([[3,4,54,67,8,8],[1,2,2,4,5,7],[4,5,3,5,6,7],[1,2,3,34,6,8]])


# In[10]:


x


# In[11]:


x[0]


# In[12]:


x[0][1:4]


# In[13]:


x[0:3,0:3]


# In[14]:


x[:,1:4]


# In[ ]:





# In[15]:


np.save('without_mask.npy',data)


# In[16]:


np.save('with_mask.npy',data)


# In[17]:


import matplotlib.pyplot as plt


# In[ ]:


plt.imshow(data[0])


# In[ ]:




