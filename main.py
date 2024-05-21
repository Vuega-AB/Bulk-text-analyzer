import streamlit as st
import openai
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import re
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from langdetect import detect
import string
from collections import Counter

LANG_MAPPING = {
    'af': 'afrikaans', 'ar': 'arabic', 'az': 'azerbaijani', 'bg': 'bulgarian', 'bn': 'bengali',
    'ca': 'catalan', 'cs': 'czech', 'da': 'danish', 'de': 'german', 'el': 'greek', 'en': 'english',
    'es': 'spanish', 'et': 'estonian', 'fa': 'persian', 'fi': 'finnish', 'fr': 'french', 'gu': 'gujarati',
    'he': 'hebrew', 'hi': 'hindi', 'hr': 'croatian', 'hu': 'hungarian', 'id': 'indonesian', 'it': 'italian',
    'ja': 'japanese', 'kn': 'kannada', 'ko': 'korean', 'lt': 'lithuanian', 'lv': 'latvian', 'mk': 'macedonian',
    'ml': 'malayalam', 'mr': 'marathi', 'ne': 'nepali', 'nl': 'dutch', 'no': 'norwegian', 'pa': 'punjabi',
    'pl': 'polish', 'pt': 'portuguese', 'ro': 'romanian', 'ru': 'russian', 'si': 'sinhala', 'sk': 'slovak',
    'sl': 'slovene', 'sq': 'albanian', 'sr': 'serbian', 'sv': 'swedish', 'sw': 'swahili', 'ta': 'tamil',
    'te': 'telugu', 'th': 'thai', 'tl': 'tagalog', 'tr': 'turkish', 'uk': 'ukrainian', 'ur': 'urdu',
    'vi': 'vietnamese', 'zh-cn': 'chinese', 'zh-tw': 'chinese', 'unknown': 'english'
}
# Function to count the occurrences of related words in the original text
def count_related_words_in_text(text, related_words):
    word_freq_map = {word: 0 for word in related_words}

    for word in related_words:
        word_freq_map[word] = text.lower().count(word)

    return word_freq_map

def plot_word_frequencies(word_freq_map, num_top_words=20):
    sorted_word_freq = sorted(word_freq_map.items(), key=lambda x: x[1], reverse=True)
    top_words = dict(sorted_word_freq[:num_top_words])
    st.bar_chart(top_words)

def preprocess_data_Map(chatgpt_words, text):
    # Convert text to lowercase and remove unwanted characters
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits

    # Initialize the frequency map for ChatGPT words
    filtered_word_freq_map = {word: 0 for word in chatgpt_words}

    # Count occurrences of each phrase in chatgpt_words in the text
    for chatgpt_word in chatgpt_words:
        count = text.count(chatgpt_word.lower())
        filtered_word_freq_map[chatgpt_word] = count

    return filtered_word_freq_map

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

def split_text_into_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks

def create_vector_store_from_text_chunks(text_chunks, openai_api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def get_conversation_chain(vector_store, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        vector_store=vector_store,
        similarity_threshold=0.8,
        max_memory_size=100,
        return_messages=True,
        input_key="question")
    conversation_chain = ConversationalRetrievalChain.from_llm(
        retriever=vector_store.as_retriever(),
        llm=llm,  
        memory=memory)
    result = conversation_chain({"question": "extract related words based on this data, return it in a list comma seprated in the same language"})
    return result["answer"]

def main():
    st.set_page_config(page_title="Bulk text analyzer", page_icon="⚡")

    st.title("Bulk text analyzer 🤖")

    # Input field for OpenAI API key
    openai_api_key = st.text_input("Enter your OpenAI API key:", "", type= 'password')

    if openai_api_key:
        # Set OpenAI API key
        openai.api_key = openai_api_key

        # Input field for uploading XML file
        xml_file = st.file_uploader("Upload XML file", type=["xlsx"])

        if xml_file:
            # Display XML file content
            st.subheader("Extracted Key Terms:")

            extracted_text = extract_text_from_xml(xml_file)
            extracted_text = preprocess_text(extracted_text)
            if st.button("Extract Key Terms"):
                # Split text into chunks
                text_chunks = split_text_into_chunks(extracted_text)

                #st.write(text_chunks)

                # Create vector store from text chunks
                vector_store = create_vector_store_from_text_chunks(text_chunks, openai_api_key)

                st.write(vector_store.index.ntotal)

                # Get conversation chain result from ChatGPT
                result = get_conversation_chain(vector_store, openai_api_key)

                st.subheader("Response:")
                st.write(result)

                # Convert the result from ChatGPT into a list of words
                result = result.replace(".", "")
                result_words = result.split(", ")


                # Preprocess the extracted text data with the words from ChatGPT
                filtered_word_freq_map = count_related_words_in_text(extracted_text, result_words)

                st.subheader("Map")
                st.write(filtered_word_freq_map)

                plot_word_frequencies(filtered_word_freq_map)

if __name__ == "__main__":
    main()
