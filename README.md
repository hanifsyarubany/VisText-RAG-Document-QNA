# VisText-RAG-DocumentQNA

## 🎨 Streamlit App – Frontend Overview
<p align="center">
  <img src="[assets/frontend_ui.png](https://cdn-images-1.medium.com/v2/resize:fit:1200/1*pO6fpSO5ov-R4aluq59R-Q.png)" alt="Alt Text" width="700"/>
</p>

---

## ⚙️ Reproducing the Environment

```bash
# Create Conda Environment
conda create -n multimodal_rag python=3.11
conda activate multimodal_rag

# Install Libraries
pip install -r requirements.txt
```

---
## 🚀 Running the Application

Once you've set up the environment and downloaded the required models, you can launch both the backend and frontend with the following commands:

#### ✅ Run the Backend Server (FastAPI)
```bash
uvicorn main:app --port 8000 
```
#### ✅ Run the Frontend (Streamlit)
```bash
streamlit run frontend.py --port 8001
```

---
## 📚 Adding Knowledge Base
You can enhance the chatbot's responses by providing your own knowledge base (PDF documents).
Before indexing any documents, **ensure that the backend server is running**.                     
1. Put your PDF files into the following directory
```bash
document_sources/
```
2. Then, run the following command to index new documents:
```bash
python execute_indexing.py
```
3. If you want to refresh the entire indexing pipeline (i.e., delete old vectors and start fresh from `document_sources/`), run:
```bash
python execute_indexing.py --initialize
```
