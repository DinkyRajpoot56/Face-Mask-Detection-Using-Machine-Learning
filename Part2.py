#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


img=cv2.imread('C:\\Users\\kush\\Desktop\\360_F_243123463_zTooub557xEWABDLk0jJklDyLSGl2jrr.jpg')


# In[3]:


img.shape


# In[4]:


img[0]


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


plt.imshow(img)


# In[7]:


cv2.imshow('result',img)


# In[ ]:


while True:
    cv2.imshow('result',img)
    if cv2.waitKey(2) == 27:
        break
cv2.destroyAllWindows()


# In[ ]:


haar_data = cv2.CascadeClassifier('data.xml')


# In[ ]:


haar_data.detectMultiScale(img)


# In[ ]:


capture=cv2.VideoCapture(0)
while True:
    flag, img = capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 4)
        cv2.imshow('result',img)
        if cv2.waitKey(2) == 27:
            break
capture.release()
cv2.destroyAllWindows()


# In[ ]:


import numpy as np


# In[ ]:


x = np.array([3,2,54,6])


# In[ ]:





# In[ ]:





# In[ ]:




