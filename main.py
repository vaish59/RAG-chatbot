import faiss
import numpy as np
import streamlit as st
import spacy
from sentence_transformers import SentenceTransformer

# Load models efficiently
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    return nlp, embedding_model

nlp, embedding_model = load_models()

# Knowledge base
documents = {
    "What is AI?": "AI stands for Artificial Intelligence.",
    "What is Machine Learning?": "Machine learning is a subset of AI.",
    "What is Deep Learning?": "Deep learning is a part of machine learning.",
    "What is NLP?": "Natural Language Processing helps computers understand human language."
}

# Preprocessing function
def preprocess(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

# Create FAISS index
@st.cache_resource
def create_faiss_index():
    processed_questions = list(documents.keys())
    processed_docs = [preprocess(q) for q in processed_questions]
    embeddings = np.array([embedding_model.encode(doc) for doc in processed_docs]).astype("float32")
    
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    
    return index, processed_questions

index, processed_questions = create_faiss_index()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit App
def main():
    st.title("AI Chatbot 🤖")

    # Sidebar with suggested questions
    st.sidebar.header("💡 Try Asking:")
    for question in documents.keys():
        if st.sidebar.button(question):  
            st.session_state.user_input = question

    # User Input
    user_input = st.text_input("Ask me anything:", key="user_input", value=st.session_state.get("user_input", ""))

    # Process Query
    if st.button("Send") and user_input.strip():
        with st.spinner("Thinking..."):
            processed_query = preprocess(user_input)
            query_embedding = embedding_model.encode(processed_query).astype("float32").reshape(1, -1)
            
            D, I = index.search(query_embedding, k=1)
            matched_index = I[0][0]
            
            if matched_index != -1:
                matched_question = processed_questions[matched_index]
                response = documents.get(matched_question, "Sorry, I couldn't find a relevant answer.")
            else:
                response = "Sorry, I couldn't find a relevant answer."

        # Store in history
        st.session_state.chat_history.append(f"You: {user_input}\nBot: {response}")

        # Show response
        st.success("Bot: " + response)

    # Display Chat History
    st.subheader("📜 Chat History")
    for message in st.session_state.chat_history:
        st.text(message)

    # Download Chat History Button
    if st.button("⬇️ Download Chat History"):
        chat_text = "\n\n".join(st.session_state.chat_history)
        st.download_button(label="Download", data=chat_text, file_name="chat_history.txt", mime="text/plain")

    # Clear Chat Button
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

if __name__ == "__main__":
    main()










