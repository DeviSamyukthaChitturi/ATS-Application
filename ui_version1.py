import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten, Add
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Huber

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


# Streamlit App UI for file upload
st.set_page_config(page_title="ATS- Resume Score Prediction", page_icon="📈", layout="centered")
st.title("📊 ATS- Resume Score Prediction")

st.write(
    """
    ## Welcome to the ATS Resume Score Predictor!
    Upload your CSV file containing the Resume and Job Description (JD) data, 
    and we'll predict the match score based on the content and relevance.
    """
)

# File uploader widget
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    
    st.write("### Dataset Preview")
    st.write(df.head())

    # Combine text fields for prediction
    df['combined_text'] = (
        df['Cleaned_JD_Qualifications'].astype(str) + " " +
        df['Cleaned_JD_Preference'].astype(str) + " " +
        df['Cleaned_JD_Job_Title'].astype(str) + " " +
        df['Cleaned_JD_Role'].astype(str) + " " +
        df['Cleaned_JD_Job_Description'].astype(str) + " " +
        df['Cleaned_JD_skills'].astype(str) + " " +
        df['Cleaned_JD_Responsibilities'].astype(str) + " " +
        df['Cleaned_Resume_Category'].astype(str) + " " +
        df['Cleaned_Resume_information'].astype(str)
    )

    # Tokenizer setup
    tokenizer = Tokenizer(num_words=5000)  # Limit to top 5000 words
    tokenizer.fit_on_texts(df['combined_text'])

    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences(df['combined_text'])

    # Pad sequences to make them the same length
    max_sequence_length = 200  # Adjust as needed
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    # Load model
    model1 = load_model(r"C:\Users\ADMIN\Documents\bhavik\ui development\model_1.h5")

    # Predict the values
    predictions = model1.predict(np.array(padded_sequences))

    # Normalize the Resume_Score before feeding into the model
    score_df  = pd.read_csv(r'C:\Users\ADMIN\Documents\bhavik\ui development\resume_scores.csv')
    scaler = MinMaxScaler()
    score_df['Resume_Score'] = scaler.fit_transform(score_df[['Resume_Score']])

    # Inverse the scaled predictions back to the original range
    predictions_original = scaler.inverse_transform(predictions)

    st.write("### ATS Score Predictions:")
    
    # Display predictions with formatted output
    st.markdown(
        """
        Based on the provided data, here are the ATS scores for your resumes relative to the respective job descriptions (JD):
        """
    )
    
    # Display the predictions in a more attractive format (use a table or chart)
    prediction_df = pd.DataFrame(predictions_original, columns=["Predicted ATS Score"])
    st.write(prediction_df)

    # Provide a download button for the result
    result_csv = prediction_df.to_csv(index=False)
    st.download_button(
        label="Download ATS Score Predictions",
        data=result_csv,
        file_name="ATS_Score_Predictions.csv",
        mime="text/csv"
    )

else:
    st.warning("Please upload a CSV file to get started!")
