import streamlit as st
import google.generativeai as genai
import docx2txt
import PyPDF2 as pdf
import pandas as pd
from keras.models import load_model
import re
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import base64
import inflect

# Load the pre-trained model (replace with the model you downloaded)
nlp = spacy.load("en_core_web_sm")

# Initialize inflect engine
p = inflect.engine()

# Load model
model1 = load_model("./model_1.h5")

api_key = "QUl6YVN5Q1RHOUlPUlpaQzY4ZEdJQWF3cVNzOEJGNnZYUmp0VEZJ"
# Configure the generative AI model with the Google API key
genai.configure(api_key=base64.b64decode(api_key).decode("utf-8"))

# Set up the model configuration for text generation
generation_config = {
    "temperature": 0,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

# Define safety settings for content generation
safety_settings = [
    {"category": f"HARM_CATEGORY_{category}", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    for category in ["HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]
]

def generate_response_from_gemini(input_text):
     # Create a GenerativeModel instance with 'gemini-pro' as the model type
    llm = genai.GenerativeModel(
    model_name="gemini-pro",
    generation_config=generation_config,
    safety_settings=safety_settings,
    )
    # Generate content based on the input text
    output = llm.generate_content(input_text)
    # Return the generated text
    return output.text

def extract_text_from_pdf_file(uploaded_file):
    # Use PdfReader to read the text content from a PDF file
    pdf_reader = pdf.PdfReader(uploaded_file)
    text_content = ""
    for page in pdf_reader.pages:
        text_content += str(page.extract_text())
    return text_content

def extract_text_from_docx_file(uploaded_file):
    # Use docx2txt to extract text from a DOCX file
    return docx2txt.process(uploaded_file)

# Function to clean tokens
def clean_tokens(tokens):
    tokens = tokens.split()
    cleaned_tokens = []
    for token in tokens:
        if token.isdigit():
            token = p.number_to_words(token)  # Convert numbers to text
        if token.lower() not in STOP_WORDS and token.lower() not in {"&", "eg", "etc", "e.g.", "etc."}:
            token = re.sub(r'[^\w\s]', '', token)  # Remove punctuation
            if token:
                cleaned_tokens.append(token.lower())
    return " ".join(cleaned_tokens)

# Set of technical terms or keywords to preserve
preserve_keywords = {"c++", "c#", "python", "java", "javascript", "sql", "html", "css", "r", "node.js", "react.js",
                     "angular.js", "ux/ui", "ui", "ux", "ms office", "ms excel", "ms power point", "aws", "oracle", "ruby"}

# List of entity types to exclude
exclude_entities = {'ORG', 'GPE', 'LOC', 'DATE', 'TIME', 'PERSON', 'FAC', 'NORP', 'EVENT'}

# # Function to preserve keywords and clean unwanted entities
def preserve_and_clean_entities(text):
    # Step 1: Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F.]+', ' ', text)  # Remove non-ASCII characters

    # Step 2: Process the text with NLP
    doc = nlp(text)  
    preserved_text = []  # List to store the cleaned and processed tokens

    # Step 3: Loop through tokens
    for token in doc:
        if token.text.lower() in preserve_keywords:
            preserved_text.append(token.text)  # Preserve keywords as-is
        elif token.ent_type_ in exclude_entities:
            continue  # Skip unwanted entities
        elif token.ent_type_ in ['PRODUCT', 'LANGUAGE']:
            preserved_text.append(token.text)  # Preserve product and language entities
        elif token.is_alpha:
            preserved_text.append(token.text.lower())  # Preserve alphabetic tokens in lowercase
        else:
            # Clean non-alphabetic tokens (e.g., punctuation)
            clean_token = re.sub(r'[^\w\s]', '', token.text)
            if clean_token:
                preserved_text.append(clean_token.lower())

    # Step 4: Join the tokens into a continuous string and return
    return "".join(preserved_text)

# Prompt Template
jd_input_prompt_template = """
As an experienced HR, your task is to find the required Qaulification, Job title, required gender, Job role, Job description, required 
skills, responsibilities and minimum number of years experience required from the given JD. 
JD:{job_description}
Values of Gender should be "male" or "female" or "both". If the required gender is not specified in the JD then give "both". 
I want the response in one single string having the structure
{{"JD_Qualifications":"","Gender":"","Job_Title":"","Job_role":"","Job_Description":"","Required_skills":"","Responsibilities":"","Minimum_Experience":""}}
"""

resume_input_prompt_template = """
As an experienced HR, your task is to find the field of experities of the the person from the given resume
Resume:{Resume}
I want the response in one single string having the structure
{{"Resume_Category":""}}
"""

# Streamlit App UI for file upload
st.set_page_config(page_title="ATS- Resume Score Prediction", page_icon="📈", layout="centered")
st.title("📊 ATS- Resume Score Prediction")

st.write(
    """
    ## Welcome to the ATS Resume Score Predictor!
    Upload your Resume and Job Description (JD), 
    and we'll predict the match score based on the content and relevance.
    """
)
st.markdown('<style>h1{text-align: center;}</style>', unsafe_allow_html=True)
job_description = st.text_area("Paste the Job Description",height=300)
uploaded_file = st.file_uploader("Upload Your Resume", type=["pdf", "docx"], help="Please upload a PDF or DOCX file")

submit_button = st.button("Submit")

if submit_button:
    if uploaded_file is not None:
        st.subheader("ATS Evaluation Result:")
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf_file(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_text = extract_text_from_docx_file(uploaded_file)

        try:
            JD_fields = generate_response_from_gemini(jd_input_prompt_template.format(job_description=job_description))
            JD_fields_dict = eval(JD_fields)
        except Exception:
            JD_fields = generate_response_from_gemini(jd_input_prompt_template.format(job_description=job_description))
            JD_fields_dict = eval(JD_fields)

        Cleaned_JD_Qualifications = clean_tokens(preserve_and_clean_entities(JD_fields_dict["JD_Qualifications"]))
        Cleaned_JD_Preference = JD_fields_dict["Gender"]
        Cleaned_JD_Job_Title = JD_fields_dict["Job_Title"]
        Cleaned_JD_Role = JD_fields_dict["Job_role"]
        Cleaned_JD_Job_Description = clean_tokens(preserve_and_clean_entities(JD_fields_dict["Job_Description"]))
        Cleaned_JD_skills = clean_tokens(preserve_and_clean_entities(JD_fields_dict["Required_skills"]))
        Cleaned_JD_Responsibilities = clean_tokens(preserve_and_clean_entities(JD_fields_dict["Responsibilities"]))
        Cleaned_JD_Minimum_Experience = JD_fields_dict["Minimum_Experience"]

        try:
            Resume_category = generate_response_from_gemini(resume_input_prompt_template.format(Resume=resume_text))
            Resume_category_dict = eval(Resume_category)
        except Exception:
            Resume_category = generate_response_from_gemini(resume_input_prompt_template.format(Resume=resume_text))
            Resume_category_dict = eval(Resume_category)

        Cleaned_Resume_Category = Resume_category_dict["Resume_Category"]
        Cleaned_Resume_information = clean_tokens(preserve_and_clean_entities(resume_text))

        input_dict = {"Cleaned_JD_Qualifications":Cleaned_JD_Qualifications,
                      "Cleaned_JD_Preference":Cleaned_JD_Preference,
                      "Cleaned_JD_Job_Title":Cleaned_JD_Job_Title,
                      "Cleaned_JD_Role":Cleaned_JD_Role,
                      "Cleaned_JD_Job_Description":Cleaned_JD_Job_Description,
                      "Cleaned_JD_skills":Cleaned_JD_skills,
                      "Cleaned_JD_Responsibilities":Cleaned_JD_Responsibilities,
                      "Cleaned_Resume_Category":Cleaned_Resume_Category,
                      "Cleaned_Resume_information":Cleaned_Resume_information,
                      "Cleaned_JD_Minimum_Experience":Cleaned_JD_Minimum_Experience,
                      }
        
        input_df = pd.DataFrame([input_dict])
            # Combine text fields for prediction
        input_df['combined_text'] = (
            input_df['Cleaned_JD_Qualifications'].astype(str) + " " +
            input_df['Cleaned_JD_Preference'].astype(str) + " " +
            input_df['Cleaned_JD_Job_Title'].astype(str) + " " +
            input_df['Cleaned_JD_Role'].astype(str) + " " +
            input_df['Cleaned_JD_Job_Description'].astype(str) + " " +
            input_df['Cleaned_JD_skills'].astype(str) + " " +
            input_df['Cleaned_JD_Responsibilities'].astype(str) + " " +
            input_df['Cleaned_Resume_Category'].astype(str) + " " +
            input_df['Cleaned_Resume_information'].astype(str)
        )

        # Tokenizer setup
        tokenizer = Tokenizer(num_words=5000)  # Limit to top 5000 words
        tokenizer.fit_on_texts(input_df['combined_text'])

        # Convert text to sequences
        sequences = tokenizer.texts_to_sequences(input_df['combined_text'])

        # Pad sequences to make them the same length
        max_sequence_length = 200  # Adjust as needed
        padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

        # # Predict the values
        predictions = model1.predict(np.array(padded_sequences))
        print(predictions)

        # Display the predictions in a more attractive format (use a table or chart)
        prediction_df = pd.DataFrame(predictions*100, columns=["Predicted ATS Score"])
        st.write(prediction_df)

         # Provide a download button for the result
        result_csv = prediction_df.to_csv(index=False)
        st.download_button(
            label="Download ATS Score Predictions",
            data=result_csv,
            file_name="ATS_Score_Predictions.csv",
            mime="text/csv"
        )
            
