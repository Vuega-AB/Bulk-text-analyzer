import streamlit as st
import openai
from openai import ChatCompletion
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import re

# Function to preprocess text data
def preprocess_text(text):
    # Remove numbers and unwanted values
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove words with 1 or 2 characters
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Join tokens back into text
    preprocessed_text = ' '.join(stemmed_tokens)
    
    return preprocessed_text


# Function to extract text from XML file
def extract_text_from_xml(xml_file):
    data = pd.read_excel(xml_file)
    text = ""
    for column in data.columns:
        text += " ".join(data[column].astype(str)) + " "
    return text

def main():
    st.title("Complaints Analysis with OpenAI")

    # Input field for OpenAI API key
    openai_api_key = st.text_input("Enter your OpenAI API key:", "")

    if openai_api_key:
        # Set OpenAI API key
        openai.api_key = openai_api_key

        # Input field for uploading XML file
        xml_file = st.file_uploader("Upload XML file", type=["xlsx"])

        if xml_file:
            # Display XML file content
            st.subheader("Extracted Key Terms:")

            # Extract text from XML file
            extracted_text = extract_text_from_xml(xml_file)
            # st.subheader("Extracted Text:")
            # st.write(extracted_text)

            if st.button("Extract Key Terms"):
                # Preprocess the extracted text
                preprocessed_text = preprocess_text(extracted_text)

                # Prompt OpenAI's language model to extract key terms
                prompt = f"Extract critical keywords from patient complaint data to aid a healthcare committee in swiftly identifying trends and addressing concerns, from the following text:\n{preprocessed_text}\n\nAfter extraction, please provide the frequency count for each key term and return it as a map only."


                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
             
                
                key_terms = response.choices[0].message.content

                st.subheader("Extracted Key Terms:")

                st.write(key_terms)

                # Convert the string into a dictionary
                key_terms_dict = eval(key_terms)

                # Separate key terms and counts into two lists
                key_terms = list(key_terms_dict.keys())
                counts = list(key_terms_dict.values())

                # Create a dictionary of key terms and their counts
                data = {'Key Term': key_terms, 'Frequency Count': counts}

                # Create a DataFrame from the dictionary
                df = pd.DataFrame(data)

                # Draw the bar chart
                st.subheader("Key Term Frequency Counts:")

                st.subheader("Bar chart for key terms:")
                st.bar_chart(df.set_index('Key Term'))

if __name__ == "__main__":
    main()
