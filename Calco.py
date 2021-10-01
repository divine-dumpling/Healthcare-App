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

st.markdown('''<center><h1><b>Calexise</b></center></h1>''', unsafe_allow_html=True)
st.markdown('''<center><h2 style="font-family : Tahoma, sans-serif">ThyselfFit</h2></center>''', unsafe_allow_html=True)
st.markdown(landing, unsafe_allow_html=True)
ch = st.selectbox(" ",["Warmup","Stats","Predict"])
if ch == "Warmup":
    st.markdown("""<center><img src="https://cdn.dribbble.com/users/722563/screenshots/2928065/dribble_3.gif", alt="workout-1", width=80%></center>""", unsafe_allow_html=True)
    st.markdown("""<center><img src="https://cdn.dribbble.com/users/722563/screenshots/2925934/dribble_2.gif", alt="workout-2", width=80%></center>""", unsafe_allow_html=True)
    st.markdown("""<center><img src="https://cdn.dribbble.com/users/722563/screenshots/2923920/dribble_1.gif", alt="workout-3", width=80%></center>""", unsafe_allow_html=True)
    st.markdown("""<center>src <a href="https://dribbble.com/shots/2928065-Fitness-App-Animation-gif", target="_blank"/>1 <a href="https://dribbble.com/shots/2925934-Fitness-App-Animation-gif", target="_blank"/>2 <a href="https://dribbble.com/shots/2923920-Fitness-App-Animation", target="_blank"/>3 </center>""", unsafe_allow_html=True)

elif ch == "Stats":
    st.markdown(stats, unsafe_allow_html=True)
    st.markdown('''<br>''',unsafe_allow_html=True)
    st.markdown('''<ul><li><p style="font-size:18px">Let's first look at the dataset used in the project.</p></li></ul><center><img src="https://user-images.githubusercontent.com/51512071/117540013-c844d000-b02a-11eb-8e1b-43a3fc0a1c6e.png", width="90%"></center><br><p style="font-size:16px">The dataset above shown gives a glimpse of various features and their expected value ranges.</p><hr><ul><li><p style="font-size:18px">Correlation plot</p></li></ul><img src="https://user-images.githubusercontent.com/51512071/117549966-14a80400-b05b-11eb-95e1-5e6ca027b5fc.png", height=90%, width=90%><p style="font-size:16px">Above plot shows how different independent variables are related to our dependent variable(calorie). On x-axis, different variables(features) are marked while on y-axis calorie is marked.<br>Higher the bar of a variable, stronger it is correlated with calorie. Value ranges between (0,1) where values closer to 0 shows weak correlation while values closer to 1 shows strong correlation.</p><hr><ul><li><p style="font-size:18px">Let's have a look on pairplot for 3 most relevant features i.e <u>duration, heart-rate, body-temperature</u> with <u>calories</u></p></li></ul>''',unsafe_allow_html=True)
    st.markdown('''<center><img src="https://user-images.githubusercontent.com/51512071/117548193-3e5c2d80-b051-11eb-8519-ecf0e7c5ada7.png", width=90%, height=90%></center>''', unsafe_allow_html=True)
    st.markdown('''<ul><li><p style="font-size:18px">Let's see some visuals</p></li></ul><p style="font-size:16px">This plot below shows the linear relation between Duration of exercise with Calories burned. From this plot we can analyse that the relaitonship between these variables is linear as with an increase in duration, calorie burn also goes up.<br>These grey points are various data points.</p>''', unsafe_allow_html=True)
    st.markdown('''<p style="font-size:17px"><u>Linear Model</u></p>''', unsafe_allow_html=True)
    # st.subheader("Linear model")
    st.markdown('''<center><img src="https://user-images.githubusercontent.com/51512071/117539554-dbef3700-b028-11eb-9f12-1a2dede7cb07.png", width=85%, height=85%></center><br><hr>''',unsafe_allow_html=True)
    st.markdown('''<p style="font-size:17px"><u>Quadratic Model</u></p><p style="font-size:16px">This plot below shows the linear as well as quadratic relation between Duration of exercise with Calories burned. From these curves it is clear that for the given features, <u>Quadratic</u> curve fits better than linear one.</p>''', unsafe_allow_html=True)
    # st.subheader("Linear model")
    st.markdown('''<center><img src="https://user-images.githubusercontent.com/51512071/117547781-22578c80-b04f-11eb-8127-285b0cecab63.png", width=85%, height=85%></center><br><hr>''',unsafe_allow_html=True)
    st.markdown('''<center><a href ="https://github.com/anuanmol/Calexise/blob/main/calexise.ipynb", target="_blank"/><p>Github link</p></center></a><center>&copy; Anmol 2021</center>"''', unsafe_allow_html=True)
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

