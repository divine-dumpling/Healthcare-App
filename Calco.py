#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pickle
import io
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


loaded_model_l = pickle.load(open('linear_model.sav', 'rb'))

landing = """<center><h2>Select your Choice</h2></center><center><h3>Warmup / Stats / Predict</h3></center>""" 

stats = """<center><h3><b><u>Statistics during Training the model</u></b></h3></center>"""

st.markdown('''<center><h1><b>Calco</b></center></h1>''', unsafe_allow_html=True)
st.markdown(landing, unsafe_allow_html=True)
ch = st.selectbox(" ",["Warmup","Stats","Predict"])
if ch == "Warmup":
    st.markdown("""<center><img src="https://user-images.githubusercontent.com/44704352/135709528-f3b677c3-593c-43ee-bf59-5a1ad1bfe6ac.gif", alt="workout-1", width=80%></center>""", unsafe_allow_html=True)

elif ch == "Stats":
    st.markdown(stats, unsafe_allow_html=True)
    st.markdown('''<br>''',unsafe_allow_html=True)
    st.markdown('''<ul><li><p style="font-size:18px">.</p></li></ul><center><img src="https://user-images.githubusercontent.com/44704352/135705584-045c5031-28a7-4be7-9b1c-048c57511bcd.png", width="90%"></center><br><p style="font-size:16px">The dataset above shown gives a glimpse of various features and their expected value ranges.</p><hr><ul><li><p style="font-size:18px">Correlation plot</p></li></ul><img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXQAAAEnCAYAAAC5ebgKAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAAf60lEQVR4nO3de5xcdX3/8dc7IRi5txBQCDGAIAaScAkRLKIgIuCvUCoqIGqpNfBTRG2lXGxFtI8WpTerKEalWrVCVSiIESI1GClgIMglCEiKRZYIBKiUYrm/+8c5k0wms7sTmN1z5vB+Ph77yJ7Lznx2s/ueM9/zvcg2EREx+CZUXUBERPRHAj0ioiES6BERDZFAj4hoiAR6RERDJNAjIhpivaqeeIsttvD06dOrevqIiIG0dOnSB21P6XasskCfPn06119/fVVPHxExkCTdPdyxNLlERDREAj0ioiES6BERDVFZG3o3Tz31FENDQzz++ONVl1J7kydPZurUqUyaNKnqUiKiJmoV6ENDQ2y88cZMnz4dSVWXU1u2eeihhxgaGmK77barupyIqIlRm1wknSfpAUnLhjkuSf8gabmkmyXt8VyLefzxx9l8880T5qOQxOabb553MhGxhl7a0L8CHDzC8UOAHcuPecDnn09BCfPe5OcUEZ1GDXTbi4GHRzjlcOCfXLgW2EzSS/tVYBXuu+8+jjrqKHbYYQdmzJjBoYceys9//vNhz99oo43W+Tle/epXP58SIyLW0o829G2Ae9q2h8p9v+o8UdI8iqt4pk2bNuoDn6kz+1Deamf4jFHPsc0RRxzBu971Ls4//3wAbrzxRu6//3522mmn513DM888w8SJE7n66quf92NFRDX6nU3QWz6Nph/dFru99++6DJLt+bbn2J4zZUrXkauVW7RoEZMmTeKEE05YtW+33XZj99135/Wvfz177LEHM2fO5OKLL17ra21z8skns+uuuzJz5kwuuOACAK688kr2339/jjnmGGbOnAmseVV/9tlns9deezFr1izOOKP4T33sscd405vexOzZs9l1111XPVZExHD6cYU+BGzbtj0VWNGHx63EsmXL2HPPPdfaP3nyZC666CI22WQTHnzwQfbee28OO+ywNdqyL7zwQm688UZuuukmHnzwQfbaay/2228/AJYsWcKyZcvW6pWycOFC7rzzTpYsWYJtDjvsMBYvXszKlSvZeuut+d73vgfAI488MobfdUQ0QT+u0C8B3ln2dtkbeMT2Ws0tg842p59+OrNmzeLAAw/k3nvv5f7771/jnKuuuoqjjz6aiRMnstVWW/Ha176W6667DoC5c+d27WK4cOFCFi5cyO67784ee+zB7bffzp133snMmTO54oorOOWUU/jxj3/MpptuOi7fZ0QMrlGv0CV9E3gdsIWkIeAMYBKA7XOBBcChwHLgN8BxY1XseNhll1349re/vdb+b3zjG6xcuZKlS5cyadIkpk+fvla3wZEW3N5www277rfNaaedxvHHH7/WsaVLl7JgwQJOO+00DjroID760Y+u43cTES8kvfRyOdr2S21Psj3V9pdtn1uGOWXvlvfZ3sH2TNsDPYXiAQccwBNPPMEXv/jFVfuuu+467r77brbccksmTZrEokWLuPvutSc822+//bjgggt45plnWLlyJYsXL2bu3LkjPt8b3/hGzjvvPP7nf/4HgHvvvZcHHniAFStWsMEGG3Dsscfy4Q9/mBtuuKG/32hENE6tRorWgSQuuugiPvjBD3LWWWcxefJkpk+fzsc+9jFOOukk5syZw2677cbOO++81tceccQRXHPNNcyePRtJfOpTn+IlL3kJt99++7DPd9BBB3Hbbbexzz77AMXN0q9//essX76ck08+mQkTJjBp0iQ+//nn1b0/Il4ANFIzwViaM2eOO+dDv+2223jlK19ZST2DKD+viGpU2W1R0lLbc7ody2yLERENkUCPiGiItKFHPA91HTEYL0y1u0Kvqk1/0OTnFBGdahXokydP5qGHHkpYjaI1H/rkyZOrLiUiaqRWTS5Tp05laGiIlStXVl1K7bVWLIqIaKlVoE+aNCkr8EREPEe1anKJiIjnLoEeEdEQCfSIiIZIoEdENEQCPSKiIRLoERENUatuixExNjJFwQtDrtAjIhoigR4R0RAJ9IiIhkigR0Q0RAI9IqIhEugREQ2RQI+IaIgEekREQyTQIyIaIoEeEdEQCfSIiIZIoEdENEQCPSKiIRLoERENkUCPiGiIzIceEbWRedufn56u0CUdLOkOScslndrl+KaSvivpJkm3Sjqu/6VGRMRIRg10SROBc4BDgBnA0ZJmdJz2PuBntmcDrwP+RtL6fa41IiJG0MsV+lxgue27bD8JnA8c3nGOgY0lCdgI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", height=50%, width=50%><p style="font-size:16px">Above plot shows how different independent variables are related to our dependent variable(calorie). On x-axis, different variables(features) are marked while on y-axis calorie is marked.<br>Higher the bar of a variable, stronger it is correlated with calorie. Value ranges between (0,1) where values closer to 0 shows weak correlation while values closer to 1 shows strong correlation.</p><hr><ul><li><p style="font-size:18px">Let's have a look on pairplot for 3 most relevant features i.e <u>duration, heart-rate, body-temperature</u> with <u>calories</u></p></li></ul>''',unsafe_allow_html=True)
    st.markdown('''<center><img src="https://user-images.githubusercontent.com/44704352/135705721-7c3b7f12-dce9-45b5-874e-324bf84c2806.png", width=90%, height=90%></center>''', unsafe_allow_html=True)
    st.markdown('''<ul><li><p style="font-size:18px">Let's see some visuals</p></li></ul><p style="font-size:16px">This plot below shows the linear relation between Duration of exercise with Calories burned. From this plot we can analyse that the relaitonship between these variables is linear as with an increase in duration, calorie burn also goes up.<br>These grey points are various data points.</p>''', unsafe_allow_html=True)
    st.markdown('''<p style="font-size:17px"><u>Linear Model</u></p>''', unsafe_allow_html=True)
    # st.subheader("Linear model")
    st.markdown('''<center><img src="https://user-images.githubusercontent.com/44704352/135705767-a745d2cc-081a-44a2-af31-fcb511a82c8e.png", width=85%, height=85%></center><br><hr>''',unsafe_allow_html=True)
    st.markdown('''<p style="font-size:17px"><u>Quadratic Model</u></p><p style="font-size:16px">This plot below shows the linear as well as quadratic relation between Duration of exercise with Calories burned. From these curves it is clear that for the given features, <u>Quadratic</u> curve fits better than linear one.</p>''', unsafe_allow_html=True)
    # st.subheader("Linear model")
    st.markdown('''<center><img src="https://user-images.githubusercontent.com/44704352/135705834-1137c51c-6616-4df0-9842-f341dc37254c.png", width=85%, height=85%></center><br><hr>''',unsafe_allow_html=True)
    st.markdown('''<center><a href ="https://github.com/divine-dumpling/Healthcare-App/blob/main/calco.ipynb", target="_blank"/><p>Github link</p></center></a><center>&copy; Ridhima Tigadi 2021</center>"''', unsafe_allow_html=True)
    # st.write('Link fot github')

elif ch == "Predict":
    st.header('Predict calories burnt during exercise')
    st.markdown("""<hr>""", unsafe_allow_html=True)
    # 'd','h','T','A','G','W','H'  features
    st.subheader('Please enter desired info')
    st.subheader("Duration (in minutes)")
    d = st.number_input(' ')
    # st.markdown("""<hr>""", unsafe_allow_html=True)
    st.subheader("Heart-rate (Beats-per-minute)")
    h = st.number_input(' ',key=1)
    # st.markdown("""<hr>""", unsafe_allow_html=True)
    st.subheader("Body-Temperature (in C)")
    T = st.number_input('',key=2)
    # st.markdown("""<hr>""", unsafe_allow_html=True)
    st.subheader("Age")
    A = st.number_input(' ',key=3)
    # st.markdown("""<hr>""", unsafe_allow_html=True)
    st.subheader("Gender (male:1, female:0)")
    G = st.number_input(' ',key=4)
    # st.markdown("""<hr>""", unsafe_allow_html=True)
    st.subheader("Weight (in kg)")
    W = st.number_input(' ',key=5)
    # st.markdown("""<hr>""", unsafe_allow_html=True)
    st.subheader("Height (in cm)")
    H = st.number_input(' ',key=6)

    ls = []
    ls.append(d)
    ls.append(h)
    ls.append(T)
    ls.append(A)
    ls.append(G)
    ls.append(W)
    ls.append(H)

    arr = np.array(ls)
    arr = arr.astype(np.float64)
    btn = st.button('Predict')
    st.markdown('''<hr>''', unsafe_allow_html=True)
    p = loaded_model_l.predict([arr])
    p = float(p)
    p = round(p,2)
    st.subheader("Calories burnt (Kcal)")
    if btn == True:
        st.info(p)

