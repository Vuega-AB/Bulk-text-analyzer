import streamlit as st
import openai
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import re
import nltk
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Language mapping
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

# Function to plot word frequencies
def plot_word_frequencies(word_freq_map, num_top_words=20):
    sorted_word_freq = sorted(word_freq_map.items(), key=lambda x: x[1], reverse=True)
    top_words = dict(sorted_word_freq[:num_top_words])
    st.bar_chart(top_words)

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove words with 1 or 2 characters
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    preprocessed_text = ' '.join(stemmed_tokens)
    return preprocessed_text

# Function to extract text from XML file
def extract_text_from_xml(xml_file):
    data = pd.read_excel(xml_file)
    text = " ".join(data.astype(str).apply(lambda x: ' '.join(x), axis=1))
    return text

# Function to split text into chunks
def split_text_into_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks

# Function to create vector store from text chunks
def create_vector_store_from_text_chunks(text_chunks, openai_api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

# Function to get conversation chain
def get_conversation_chain(vector_store, openai_api_key):
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4-turbo")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        vector_store=vector_store,
        similarity_threshold=0.8,
        max_memory_size=100,
        return_messages=True,
        input_key="question"
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        retriever=vector_store.as_retriever(),
        llm=llm,
        memory=memory
    )
    # Step 1: Provide the data to the model for understanding
    conversation_chain({"question": "Please read and understand the following data: "})
    
    # Step 2: Request extraction of related words
    result = conversation_chain({"question": "Based on your understanding of the data, extract related words and return them in a list, comma-separated, in the same language."})
    return result["answer"]

# Main function
def main():
    st.set_page_config(page_title="Bulk Text Analyzer", page_icon="⚡")
    st.title("Bulk Text Analyzer 🤖")

    openai_api_key = st.text_input("Enter your OpenAI API key:", "", type='password')

    if openai_api_key:
        openai.api_key = openai_api_key

        xml_file = st.file_uploader("Upload XML file", type=["xlsx"])

        if xml_file:
            st.subheader("Extracted Key Terms:")

            extracted_text = extract_text_from_xml(xml_file)
            extracted_text = preprocess_text(extracted_text)

            if st.button("Extract Key Terms"):
                text_chunks = split_text_into_chunks(extracted_text)
                vector_store = create_vector_store_from_text_chunks(text_chunks, openai_api_key)
                st.write(vector_store.index.ntotal)

                result = get_conversation_chain(vector_store, openai_api_key)

                st.subheader("Response:")
                st.write(result)

                result = result.replace(".", "")
                result_words = result.split(", ")

                filtered_word_freq_map = count_related_words_in_text(extracted_text, result_words)

                st.subheader("Map")
                st.write(filtered_word_freq_map)

                plot_word_frequencies(filtered_word_freq_map)

if __name__ == "__main__":
    main()
