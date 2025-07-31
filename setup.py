import os
import torch
from groq import Groq
from typing import cast
from ultralytics import YOLO
from pymilvus import MilvusClient
from transformers.utils.import_utils import is_flash_attn_2_available
from retriever_class import MilvusColbertRetriever, MilvusBasicRetriever
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from sentence_transformers import SentenceTransformer

# Define the device
device = "cuda:0"

# Define knowledge base sources and target image directory
document_source_dir = "document_sources"
img_dir = "image_database"
os.makedirs(img_dir, exist_ok=True) # Ensure the directory exists

# YOLO-12L-Doclaynet
yolo_model = YOLO("pretrained_models/yolo-doclaynet/yolov12l-doclaynet.pt")
yolo_model = yolo_model.to(device)

# ColQwen2.5-Colpali
colpali_model = ColQwen2_5.from_pretrained(
        "pretrained_models/colqwen2.5-v0.2",
        torch_dtype=torch.bfloat16,
        device_map=device,  # or "mps" if on Apple Silicon
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    ).eval()
colpali_processor = ColQwen2_5_Processor.from_pretrained("pretrained_models/colqwen2.5-v0.2")
processor = cast(
    ColPaliProcessor, 
    colpali_processor)

# Mxbai-embed-large-v1
embed_model = SentenceTransformer("pretrained_models/mxbai-embed-large-v1",device=device)

# Groq API-Llama4
os.environ["GROQ_API_KEY"] = "<your-api-key>"
client_groq = Groq()

# Milvus Client
client = MilvusClient("milvus_file.db")
colbert_retriever = MilvusColbertRetriever(collection_name="colbert", milvus_client=client,img_dir=img_dir)
basic_retriever = MilvusBasicRetriever(collection_name="basic", milvus_client=client)

# Define Entity Colors
ENTITIES_COLORS = {
    "Picture": (255, 72, 88),
    "Table": (128, 0, 128)
}

print("FINISH SETUP...")