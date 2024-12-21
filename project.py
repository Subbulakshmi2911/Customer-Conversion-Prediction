import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import pickle
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np

# Load the trained model
model = joblib.load('D:/Final Project/trained_classification_model.pkl')

# Function to map categorical features back to their original values
def map_back(column, mapping):
    if isinstance(mapping, dict):
        reverse_mapping = {v: k for k, v in mapping.items()}
        return column.map(reverse_mapping)
    else:
        return column

# Custom CSS styles for the app
st.markdown(
    """
    <style>
    .header {
        background-color: #33adff;
        color: white;
        padding: 0.4rem;
        text-align: center;
        font-size: 1.75rem;
        border-radius: 0.25rem;
        margin-bottom: .75rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .button {
        background-color: #33adff;
        color: white;
        padding: 1rem 1.5rem;
        font-size: 1.2rem;
        border: none;
        border-radius: 0.5rem;
        cursor: pointer;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .prediction {
        font-size: 1.8rem;
        font-weight: bold;
        margin-top: 1.5rem;
        text-align: center;
    }
    .branding {
        font-size: 1rem;
        font-weight: bold;
    }
    .sidebar-content {
        padding: 2rem;
    }
    .sidebar-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #33adff;
        margin-bottom: 1rem;
    }
    .sidebar-link {
        font-size: 1.2rem;
        color: #33adff;
        text-decoration: none;
    }
    .sidebar-link:hover {
        text-decoration: underline;
    }
    .cover-image {
        max-width: 75%;
        margin: 0 auto;
        display: block;
        border-radius: 1rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Sidebar menu
# Sidebar menu
with st.sidebar:
    select = option_menu("Main Menu", ["About", "Prediction Models"], 
                         icons=["house", "gear", "info-circle"], default_index=0,
                         styles={
                             "container": {"padding": "5px", "background-color": "#EFFBFF"},
                             "nav-link": {"font-size": "16px", "text-align": "left", "margin": "5px", "--hover-color": "#CCE5FF"},
                             "nav-link-selected": {"background-color": "#009999"},
                         })

    # Adding content for "About the App Developer"
    st.sidebar.markdown('### About the App Developer')
    st.sidebar.markdown('<p class="branding">SUBBULAKSHMI S</p>', unsafe_allow_html=True)
    st.sidebar.markdown('<p class="branding">BATCH: MDTE_04</p>', unsafe_allow_html=True)
    st.sidebar.markdown('<p class="branding">DOMAIN : DATA SCIENCE</p>', unsafe_allow_html=True)
    st.sidebar.markdown('<p class="branding">Contact Information:</p>', unsafe_allow_html=True)
    st.sidebar.markdown('[LinkedIn Profile](https://www.linkedin.com/in/subbulakshmi-s-15b489171.)')
    st.sidebar.markdown('GitHub Account: [Subbulakshmmi2911](https://github.com/Subbulakshmi2911)')
    st.sidebar.markdown('Email: subbusubha.005@gmail.com')
# Streamlit app

if select == "About":
    st.markdown("""
    <h1 style="text-align: center; color: blue;">Customer Insurance Subscription Prediction</h1>
            """, unsafe_allow_html=True)

    # Header and Introduction
    st.header("🛠️ Skills Takeaway From This Project")
    st.markdown(
        """
        <ul>
            <li><strong>Python Scripting</strong></li>
            <li><strong>Data Preprocessing</strong></li>
            <li><strong>Exploratory Data Analysis (EDA)</strong></li>
            <li><strong>Machine Learning</strong></li>
            <li><strong>Streamlit Application Development</strong></li>
        </ul>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h3 class='section-header'>📖 Introduction</h3>", unsafe_allow_html=True)
    

    st.markdown(
        """
        ### :red[Overview:] 
        This project aims to construct a machine learning model and implement it as a user-friendly online application 
        to provide accurate predictions for **Customer Insurance Subscription**. The prediction model is built using 
        past transactional data to ensure reliability and precision.
        """
    )

    # Project Summary or Visual
    with st.expander("🔍 Learn More About This Project"):
        st.write(
            """
            - **Technologies Used:** Python, Streamlit, Pandas, Scikit-learn
            - **Key Features:** User-friendly interface, Insurance prediction, interactive visualization
            - **Objective:** Empower users to estimate resale prices and make informed decisions.
            """
        )
    

    # Footer or Call-to-Action
    st.markdown(
        """
        ---
        ### ✨ Ready to explore? Navigate to the **Prediction Models** section in the sidebar!
        """,
        unsafe_allow_html=True
    )


if select == "Prediction Models": 
    #st.markdown("# :red[Predicting Results based on Trained Model]")  
    st.markdown("""
    <h1 style="text-align: center; color: blue;">Predicting Results based on Trained Model</h1>
            """, unsafe_allow_html=True)
    def main():
        
        # Read the dataset
        dataset = pd.read_csv('D:/Final Project/train.csv')
        dataset = dataset.dropna(axis=0)

        # Map categorical features to numeric values
        mapping = {
            'y': {'no': 0, 'yes': 1},
            'mon': {'jan': 2, 'feb': 6, 'mar': 11, 'apr': 7, 'may': 0, 'jun': 4, 'jul': 1, 'aug': 5, 'sep': 9, 'oct': 8, 'nov': 3, 'dec': 10},
            'education_qual': {'tertiary':3, 'secondary':1, 'unknown':2, 'primary':0},
            'marital': {'married':0, 'single':2, 'divorced':1},
            'call_type': {'unknown':0, 'cellular':2, 'telephone':1},
            'prev_outcome': {'unknown':0, 'failure':1, 'other':2, 'success':3},
            'job':{
    'management': 8,
    'technician': 4,
    'entrepreneur': 1,
    'blue-collar': 0,
    'unknown': 5,
    'retired': 10,
    'admin.': 7,  
    'services': 3,
    'self-employed': 6,
    'unemployed': 9,
    'housemaid': 2,
    'student': 11
}
        }
        dataset.replace(mapping, inplace=True)




        col1, col2, col3 = st.columns(3)
        with col1:
           
            job = st.selectbox('Job', dataset['job'].unique(), format_func=lambda x: map_back(pd.Series([x]), mapping['job'])[0])
            
        with col2:
            marital = st.selectbox('Marital Status', dataset['marital'].unique(), format_func=lambda x: map_back(pd.Series([x]), mapping['marital'])[0])
        with col3:
            education_qual = st.selectbox('Education Qualification', dataset['education_qual'].unique(), format_func=lambda x: map_back(pd.Series([x]), mapping['education_qual'])[0])

        col4, col5, col6 = st.columns(3)
        with col4:
            call_type = st.selectbox('Call Type', dataset['call_type'].unique(), format_func=lambda x: map_back(pd.Series([x]), mapping['call_type'])[0])
        with col5:
            prev_outcome = st.selectbox('Previous Outcome', dataset['prev_outcome'].unique(), format_func=lambda x: map_back(pd.Series([x]), mapping['prev_outcome'])[0])
        with col6:
            mon = st.selectbox('Month', dataset['mon'].unique(), format_func=lambda x: map_back(pd.Series([x]), mapping['mon'])[0])

        age = st.slider('Age', min_value=18, max_value=95, value=18)
        day = st.slider('Day', min_value=1, max_value=31, value=1)
        dur = st.slider('Duration (Seconds)', min_value=0, max_value=4918, value=0)
        num_calls = st.slider('Number of Calls', min_value=0, max_value=63, value=0)

        # Add a "Predict" button
        if st.button('Predict', key='predict_button'):
            # Prepare the input data for prediction
            data = pd.DataFrame({
                'age': [age],
                'job': [job],
                'marital': [marital],
                'education_qual': [education_qual],
                'call_type': [call_type],
                'day': [day],
                'mon': [mon],
                'dur': [dur], 
                'num_calls': [num_calls],
                'prev_outcome': [prev_outcome]
            })

            # Make prediction using the loaded model
            prediction = model.predict(data)[0]

            st.markdown('<p class="header">Prediction</p>', unsafe_allow_html=True)

            # Display the prediction
            if prediction == 0:
                st.markdown('<p class="prediction" style="color: red;">No, The customer is highly unlikely to subscribe to the insurance.</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="prediction" style="color: green;">Yes, The customer is highly likely to subscribe to the insurance.</p>', unsafe_allow_html=True)

    # Run the app
    if __name__ == '__main__':
        main()