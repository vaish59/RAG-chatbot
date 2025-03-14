# RAG Chatbot 🤖  
A Retrieval-Augmented Generation (RAG) chatbot using **FAISS, Streamlit, and NLP**.

## 🔹 How I Created This Chatbot  

### **1️⃣ Installed Dependencies**
First, I installed the required libraries:
```bash
pip install faiss-cpu numpy streamlit spacy sentence-transformers
Then, I downloaded the spaCy model:

python -m spacy download en_core_web_sm


2️⃣ Created main.py
Built a knowledge base (predefined Q&A).
Used spaCy for text preprocessing.
Used SentenceTransformer to generate embeddings.
Created a FAISS index for fast retrieval.
Developed the Streamlit UI for the chatbot.
3️⃣ Ran the Chatbot
To start the chatbot, I ran:

streamlit run main.py


The chatbot interface launched in the browser.

4️⃣ Uploaded to GitHub
Initialized Git
Committed the files
Pushed to GitHub Repository


📜 How to Run This Chatbot?
Clone the repository:
bash
git clone https://github.com/vaish59/RAG-chatbot.git

Install dependencies:
bash
pip install -r requirements.txt

Run the chatbot:

streamlit run main.py


4️⃣ Save and Push to GitHub
Once you've added this content:

Save the file (Ctrl + S in Notepad).
Push the updated README.md to GitHub:
bash

git add README.md
git commit -m "Updated README with project steps"
git push origin main
