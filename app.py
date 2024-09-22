import streamlit as st
import pickle
import os
import numpy as np
import warnings  


warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')


from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO


st.set_page_config(page_title="OnlineDoc", layout="wide", page_icon="Doc_icon.jpeg")

try:
    working_dir = os.path.dirname(os.path.abspath(__file__))
    diabetes_model = pickle.load(open(os.path.join(working_dir, 'DiabetesPredict.pkl'), 'rb'))
    heart_disease_model = pickle.load(open(os.path.join(working_dir, 'HeartPredictor.pkl'), 'rb'))
except FileNotFoundError as e:
    st.error("Error loading model files. Please check the file paths.")
    st.stop()


st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #6a0dad, #c5c5ff);
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: rgba(255, 255, 255, 0.9);
        color: black;
    }
    .header {
        font-size: 30px;
        color: #ffffff;
        font-weight: bold;
    }
    .result {
        font-size: 20px;
        color: #ffffff;
    }
    .stTextInput>div>input {
        background-color: #e3e3e3;
        color: black;
    }
    .stButton>button {
        background-color: #5a5a5a;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


with st.sidebar:
    selected = option_menu("Welcome To OnlineDoc",
                           ['Our Dashboard',
                            'Diabetes Predictor',
                            'Heart Disease Predictor',
                            'Contact Us',
                            'About Us'],
                           menu_icon='person',
                           icons=['grid', 'activity', 'heart', 'envelope', 'info'],
                           default_index=0)


if selected == 'Our Dashboard':
    st.markdown("<h1 style='text-align: center; color: #ffffff;'>OnlineDoc Analysis</h1>", unsafe_allow_html=True)

    st.markdown("### Prediction Statistics", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)

    metrics = [
        ("Total Predictions", "500", "#ff9999"),
        ("Prediction Accuracy", "92%", "#66b3ff"),  
        ("Diseases Detected", "2", "#ffcc99"),      
        ("Predictions Today", "23", "#99ff99"),     
        ("Weekly Predictions", "200", "#99ccff")     
    ]


    for i, (label, value, color) in enumerate(metrics):
        with [col1, col2, col3, col4, col5][i]:
            st.markdown(f"<h3 style='color:{color};'>{label}</h3>", unsafe_allow_html=True)
            st.metric("", value)

    st.markdown("---")


    st.markdown("### Prediction Updates", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3 style='color:#ff6666;'>Positive Predictions</h3>", unsafe_allow_html=True)
        st.metric("", "681", "+15%")
        st.markdown("<h3 style='color:#66cc66;'>Negative Predictions</h3>", unsafe_allow_html=True)
        st.metric("", "788")

      
        if st.button("Download Report"):
            st.write("Download functionality not yet implemented.")

    with col2:

        labels = ['Diabetes', 'Heart Disease']
        pos_predictions = [340, 341]
        neg_predictions = [400, 388]

        fig, ax = plt.subplots(figsize=(6, 4))
        bar_width = 0.35
        index = np.arange(len(labels))

        ax.bar(index, pos_predictions, bar_width, label='Positive', color='#5cb85c', edgecolor='black')
        ax.bar(index + bar_width, neg_predictions, bar_width, label='Negative', color='#d9534f', edgecolor='black')

        ax.set_xlabel('Diseases', fontsize=12, color='#ffffff')
        ax.set_ylabel('Predictions', fontsize=12, color='#ffffff')
        ax.set_title('Predictions by Disease', fontsize=14, color='#ffffff')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(labels, fontsize=10, color='#ffffff')
        ax.legend()

   
        ax.spines['bottom'].set_color('#ffffff')
        ax.spines['left'].set_color('#ffffff')
        ax.tick_params(axis='x', colors='#ffffff')
        ax.tick_params(axis='y', colors='#ffffff')

        st.pyplot(fig)

    st.markdown("---")


    st.markdown("### Top Diseases Predicted", unsafe_allow_html=True)

    disease_labels = ['Diabetes', 'Heart Disease']
    disease_counts = [340, 341]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(disease_counts, labels=disease_labels, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'], startangle=90, wedgeprops={'edgecolor': 'black'})
    ax.axis('equal')

    st.pyplot(fig)

    st.markdown("---")

 
    st.markdown("### Prediction Breakdown & Individual Overview", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        breakdown_labels = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5']
        breakdown_values = [10, 50, 30, 80, 60]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(breakdown_labels, breakdown_values, marker='o', color='#ffcc00', linewidth=2)

        ax.set_title('Predictions Over Time', fontsize=14, color='#ffffff')
        ax.set_xlabel('Days', fontsize=12, color='#ffffff')
        ax.set_ylabel('Predictions', fontsize=12, color='#ffffff')

        ax.spines['bottom'].set_color('#ffffff')
        ax.spines['left'].set_color('#ffffff')
        ax.tick_params(axis='x', colors='#ffffff')
        ax.tick_params(axis='y', colors='#ffffff')

        st.pyplot(fig)

    with col2:

        individual_labels = ['Person 1', 'Person 2', 'Person 3', 'Person 4', 'Person 5']
        individual_scores = [75, 85, 95, 65, 80]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(individual_labels, individual_scores, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#99ccff'], edgecolor='black')

        ax.set_title('Individual Prediction Scores', fontsize=14, color='#ffffff')
        ax.set_xlabel('Individuals', fontsize=12, color='#ffffff')
        ax.set_ylabel('Score', fontsize=12, color='#ffffff')

        ax.spines['bottom'].set_color('#ffffff')
        ax.spines['left'].set_color('#ffffff')
        ax.tick_params(axis='x', colors='#ffffff')
        ax.tick_params(axis='y', colors='#ffffff')

        st.pyplot(fig)

 
    if st.button("Download Results"):
         st.write("Download functionality not yet implemented.")


        


elif selected == 'Diabetes Predictor':
    st.title("Diabetes Predictor")
    st.markdown("### Please enter the following details:")

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
    with col2:
        Glucose = st.text_input("Glucose Level")
    with col3:
        BloodPressure = st.text_input("Blood Pressure Value")
    
    with col1:
        SkinThickness = st.text_input("Skin Thickness Value")
    with col2:
        Insulin = st.text_input("Insulin Value")
    with col3:
        BMI = st.text_input("BMI Value")
    
    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    with col2:
        Age = st.text_input("Age")

    diabetes_result = ""
    if st.button("Diabetes Test Result"):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        try:
            user_input = [float(x) for x in user_input]
            prediction = diabetes_model.predict([user_input])
            if prediction[0] == 1:
                diabetes_result = "The person has diabetes."
            else:
                diabetes_result = "The person does not have diabetes."
        except ValueError:
            diabetes_result = "Please enter valid numerical values for all inputs."

    st.markdown(f"<div class='result'>{diabetes_result}</div>", unsafe_allow_html=True)

elif selected == 'Heart Disease Predictor':
    st.title("Heart Disease Predictor")
    st.markdown("### Please enter the following details:")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input("Age")
    with col2:
        sex = st.text_input("Sex")
    with col3:
        cp = st.text_input("Chest Pain Types")

    with col1:
        trestbps = st.text_input("Resting Blood Pressure")
    with col2:
        chol = st.text_input("Serum Cholesterol in mg/dl")
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
    with col3:
        ca = st.text_input('Major vessels colored by fluoroscopy')
    
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversible defect')

    heart_disease_result = ""
    if st.button("Heart Disease Test Result"):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        try:
            user_input = [float(x) for x in user_input]
            prediction = heart_disease_model.predict([user_input])
            if prediction[0] == 1:
                heart_disease_result = "This person is having heart disease."
            else:
                heart_disease_result = "This person does not have any heart disease."
        except ValueError:
            heart_disease_result = "Please enter valid numerical values for all inputs."

    st.markdown(f"<div class='result'>{heart_disease_result}</div>", unsafe_allow_html=True)

elif selected == 'Contact Us':
    st.title("Contact Us")

    def is_valid_email(email):
        email_pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        return re.match(email_pattern, email) is not None

    with st.form("contact_form"):
        name = st.text_input("First Name")
        email = st.text_input("Email Address")
        message = st.text_area("Your Message")
        submit_button = st.form_submit_button("Submit")

    if submit_button:
        if not name:
            st.error("Please provide your name.")
        elif not email or not is_valid_email(email):
            st.error("Please provide a valid email address.")
        elif not message:
            st.error("Please provide a message.")
        else:
            st.success("Your message has been sent successfully!")


elif selected == 'About Us':
    st.title("About Us")
    st.markdown("<h2 style='color: black;'>Made with ❤️ by Yash, Suhani, and Vishakha.</h2>", unsafe_allow_html=True)



