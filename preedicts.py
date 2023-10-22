import requests
import pickle
import streamlit as st
from streamlit_lottie import st_lottie 
from PIL import Image
import numpy as np
import pandas as pd

with open(r'model.pkl','rb') as file:
    football_module = pickle.load(file)
with open(r'scaler.pkl','rb') as file:
    scaler_module = pickle.load(file)

st.set_page_config(
    page_title="FIFA BEST PLAYER PREDICTIONS",
    page_icon=":sports_medal:",
    layout="centered",
    initial_sidebar_state="expanded",
  
)

with st.sidebar:
    image = Image.open(r"Cristiano and Messi wallpaper 4k.jpeg")
    new_image = image.resize((400, 300))
    st.image(image, caption=None, width=None, use_column_width="always", 
    clamp=False, channels="RGB", output_format="auto")
    image1 = Image.open(r"Vini jr.jpeg")
    st.image(image1, caption=None, width=None, use_column_width="always", 
    clamp=False, channels="RGB", output_format="auto")
    image2 = Image.open(r"Pin by Gg on Kylian Mbapp‚ _ Football players images, Real madrid wallpapers, Kylian mbapp‚.jpeg")
    st.image(image2, caption=None, width=None, use_column_width="always", 
    clamp=False, channels="RGB", output_format="auto") 
    image4 = Image.open(r"Jude Bellingham.jpeg")
    st.image(image4, caption=None, width=None, use_column_width="always", 
    clamp=False, channels="RGB", output_format="auto")
    image5 = Image.open(r"66b56696-eae6-452b-a261-2d092a6d56e0.jpeg")
    st.image(image5, caption=None, width=None, use_column_width="always", 
    clamp=False, channels="RGB", output_format="auto")

def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ball = load_lottie("https://lottie.host/4b73c268-6131-4704-8906-ca575c28a29a/Vp3Hr8VmJa.json")
st.header("PlayerPredicts", divider='rainbow')
#Header of the website
with st.container():
    left_column,right_column = st.columns(2)
    with left_column:
        st.subheader("About Us")
        st.text("Welcome to the future of football analytics! At PlayerPredicts, we are passionate about harnessing the power of artificial intelligence to revolutionize the way we evaluate and understand player performance in football.")
        st.subheader("Our Mission")
        st.text("Our mission is to provide cutting-edge AI solutions to sports organizations, coaches, and enthusiasts, enabling them to make data-driven decisions and gain deeper insights into player performance. We aim to make player rating predictions more accurate, efficient, and accessible.")
        st.subheader("What we do")
        st.text("Using advanced machine learning algorithms, we analyze vast amounts of data, including player statistics, match results, and various performance metrics. Our AI models consider not only the obvious factors but also subtle nuances and hidden patterns that can impact a player's rating.")
    with right_column:
        st_lottie(lottie_ball,height = 300, key ="coding")
    #st.divider()
    st.subheader("Let's shape the future of football together!", divider='rainbow')


#sliders for user response
with st.container():
    #st.markdown("<small>This is even smaller text</small>", unsafe_allow_html=True)
    #st.write("Please, rate your football skills based on these categories.")
    st.text("Please, rate your football skills based on these categories.")
    Movement_reactions = st.slider('Rate your movement reactions skills',0,100,0)
    #st.write(Movement_reactions)
    Passing = st.slider('Rate your passing skills', 0,100,0)
    mentality_composure = st.slider('Rate your mentality composure',0,100,0)
    dribbling = st.slider('Rate your  dribbling skills',0,100,0)
    power_shot_power = st.slider('Rate your the power of your  shots',0,100,0)
    physic = st.slider('Rate your physic',0,100,0)
    mentality_vision = st.slider('Rate your  mentality vision',0,100,0)
    attacking_short_passing = st.slider('Rate your short passing attack skills',0,100,0)
    shooting = st.slider('Rate your shooting skills',0,100,0)
    skill_long_passing = st.slider('Rate your long passing skills',0,100,0)
    age = st.slider('What is your age',16,  60, 16)
    skill_ball_control= st.slider('Rate your ball control skills',0,100,0)
    skill_curve= st.slider('Rate your ball curving skills',0,100,0)
    attacking_crossing= st.slider('Rate your attack crossing skills',0,100,0)
    power_long_shots= st.slider('Rate your the power of your long shots',0,100,0)
    mentality_aggression= st.slider('Rate your mental aggression',0,100,0)
    users_response=[Movement_reactions,Passing,mentality_composure,dribbling,power_shot_power,
                 physic,mentality_vision,attacking_short_passing,shooting,skill_long_passing,age,
                skill_ball_control,skill_curve,attacking_crossing,
                power_long_shots,mentality_aggression]
    submit = st.button("Submit")
    cols = ['movement_reactions', 'passing', 'mentality_composure', 'dribbling', 
            'power_shot_power', 'physic', 'mentality_vision', 'attacking_short_passing', 
            'shooting', 'skill_long_passing', 'age', 'skill_ball_control',
              'skill_curve', 'attacking_crossing', 
            'power_long_shots', 'mentality_aggression']
    if submit:
        newData = pd.DataFrame([users_response],columns= cols)
        scaledData = scaler_module.transform(newData)
        predict = football_module.predict(scaledData)
        print(scaledData)
        st.subheader("Your football player rating is "+str( round(predict[0])),divider='rainbow')
        print(predict[0]+(0.8809915725973115*1.96))
        print(predict[0]+(0.8809915725973115-1.96))
        st.text("There's a 95% chance that the rating is between " + str(round(predict[0]+(0.8809915725973115-1.96))) +" and "+str(round(predict[0]+(0.8809915725973115*1.96))))

        

    #adding responses to list



    



    