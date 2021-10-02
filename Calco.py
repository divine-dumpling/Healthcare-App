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
    st.markdown('''<ul><li><p style="font-size:18px">.</p></li></ul><center><img src="https://user-images.githubusercontent.com/44704352/135705584-045c5031-28a7-4be7-9b1c-048c57511bcd.png", width="90%"></center><br><p style="font-size:16px">The dataset above shown gives a glimpse of various features and their expected value ranges.</p><hr><ul><li><p style="font-size:18px">Correlation plot</p></li></ul><img src="https://user-images.githubusercontent.com/44704352/135709987-90069c8c-498d-481c-be92-c55ba0a68070.png", height=90%, width=90%><p style="font-size:16px">Above plot shows how different independent variables are related to our dependent variable(calorie). On x-axis, different variables(features) are marked while on y-axis calorie is marked.<br>Higher the bar of a variable, stronger it is correlated with calorie. Value ranges between (0,1) where values closer to 0 shows weak correlation while values closer to 1 shows strong correlation.</p><hr><ul><li><p style="font-size:18px">Let's have a look on pairplot for 3 most relevant features i.e <u>duration, heart-rate, body-temperature</u> with <u>calories</u></p></li></ul>''',unsafe_allow_html=True)
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

