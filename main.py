import streamlit as st
import openai
import pandas as pd
import re
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')
nltk.download('stopwords')

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

def get_conversation_chain(vector_store, openai_api_key, user_description):
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        vector_store=vector_store,
        similarity_threshold=0.8,
        max_memory_size=100,
        return_messages=True,
        input_key="question")
    
    prompt = f"Given the following data description: {user_description}. Extract related words based on this data, return it in a list, comma separated in the same language."
    conversation_chain = ConversationalRetrievalChain.from_llm(
        retriever=vector_store.as_retriever(),
        llm=llm,  
        memory=memory)
    result = conversation_chain({"question": prompt})
    return result["answer"]

def main():
    st.set_page_config(page_title="Bulk text analyzer", page_icon="⚡")

    st.title("Bulk text analyzer 🤖")

    # Input field for OpenAI API key
    openai_api_key = st.text_input("Enter your OpenAI API key:", "", type='password')

    if openai_api_key:
        # Set OpenAI API key
        openai.api_key = openai_api_key

        # Input field for uploading XML file
        xml_file = st.file_uploader("Upload XML file", type=["xlsx"])

        if xml_file:
            # Input field for user to describe the data
            user_description = st.text_area("Describe the data you have uploaded:")

            if user_description:
                # Display XML file content
                st.subheader("Extracted Key Terms:")

                extracted_text = extract_text_from_xml(xml_file)
                extracted_text = preprocess_text(extracted_text)
                if st.button("Extract Key Terms"):
                    # Split text into chunks
                    text_chunks = split_text_into_chunks(extracted_text)

                    # Create vector store from text chunks
                    vector_store = create_vector_store_from_text_chunks(text_chunks, openai_api_key)

                    st.write(vector_store.index.ntotal)

                    # Get conversation chain result from ChatGPT
                    result = get_conversation_chain(vector_store, openai_api_key, user_description)

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
