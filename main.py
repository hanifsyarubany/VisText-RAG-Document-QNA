from setup import *
from function import *
from prompt_template import *
from fastapi import FastAPI
import numpy as np
import pytesseract
import torch
import argparse
import os
import fitz  
import cv2
import random
import json
import re
from pydantic import BaseModel

class ChatRequest(BaseModel):
    user_query: str

class IndexRequest(BaseModel):
    initialize: bool
    
app = FastAPI()

@app.post("/chat-inference")
def generate_chat_inference(req: ChatRequest):
    user_query = req.user_query
    # Retrieve the Context
    print("Retrieve")
    batch_query = colpali_processor.process_queries([user_query]).to(device)
    embeddings_query = torch.unbind(colpali_model(**batch_query).to("cpu"))[0].float().numpy()
    colbert_retriever_result = colbert_retriever.search(embeddings_query, topk=3)
    basic_retriever_result = basic_retriever.search(embed_model.encode(user_query), topk=3)
    # Create Payload
    print("Payload")
    payload_content = [{
                        "type": "text",
                        "text": f"User Query: {user_query}"
                        }]
    for i in range(len(colbert_retriever_result)):
        img_payload = {
            "type": "image_url",
            "image_url": {"url":url_conversion(colbert_retriever_result[i]["content"])}
        }
        payload_content.append(img_payload)
    for i in range(len(basic_retriever_result)):
        txt_payload = {
            "type": "text",
            "text": f"Text-based Context #{i+1}:\n{basic_retriever_result[i]['content']}"
        }
        payload_content.append(txt_payload)
    # Create Messages
    print("Create")
    messages = [
        {
            "role": "system",
            "content": system_instruction
        },
        {
            "role": "user",
            "content": payload_content
        }
    ]
    # Model Inference
    print("Inference")
    chat_result = llama4_inference(messages)
    chat_result = re.findall(r'\{[^{}]+\}', chat_result)
    chat_result = json.loads(chat_result[-1])
    # Image Content
    print("Content")
    if chat_result["need_image"]:
        img_content = colbert_retriever_result[chat_result['image_index']]["content"]
    else:
        img_content = ""
    # Output
    print("Output")
    return {
        "response":chat_result["response"],
        "need_image":chat_result["need_image"],
        "img_base64":img_content
    }

@app.post("/indexing")
def indexing_pipeline(req: IndexRequest):
    print("START...")
    if req.initialize:
        colbert_retriever.create_collection()
        colbert_retriever.create_index()
        basic_retriever.create_collection()
        basic_retriever.create_index()
        flush_directory(img_dir)
        print("Finish Initializing...")
    print("Start Indexing...\n")
    # Read folder document sources
    list_files = os.listdir(document_source_dir)
    if len(list_files)==0:
        return_message = "There is no Document in your document source directory!!!"
        print(return_message)
        return {"response":return_message}
    # Read all of the files
    no_files=True
    for filename in list_files:
        if filename not in os.listdir(img_dir):
            no_files = False
            print(f"============ INDEXING file \"{filename}\" ============\n")
            doc = fitz.open(os.path.join(document_source_dir,filename))
            for page_id in range(doc.page_count):
                print(f"=====> Indexing Page-{page_id}")
                # Get page image
                print("Get page image...")
                page = doc.load_page(page_id)
                pix = page.get_pixmap(dpi=300)
                img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
                img_page = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                # Yolo Layout Detection
                print("Perform YOLO DocLayNet...")
                layout_results = layout_detection(img_page)
                # Extract and Masking
                print("Extract and masking the image...")
                extracted_imgs, img_page = extract_and_masking_images(img_page, layout_results)
                # Saving images
                print("Save images...")
                save_img_files(extracted_imgs, filename, page_id)
                # Create Embedding Loader
                print("Encode images...")
                embedding_loader = create_image_embedding_loader(extracted_imgs)
                # Extract current text in the modified page image
                print("Extract the text using OCR...")
                text = pytesseract.image_to_string(img_page)
                # Indexing to Colbert collection
                print("Index to Colbert Collection...")
                for i in range(len(extracted_imgs)):
                    data = {
                        "colbert_vecs": embedding_loader[i].float().numpy(),
                        "doc_id": random.getrandbits(63),
                        "page_id": page_id,
                        "fig_id": i,
                        "filename": filename,
                    }
                    colbert_retriever.insert(data)
                # Indexing to Basic collection
                print("Index to Basic Collection...")
                data = {
                    "vector": embed_model.encode(text),
                    "content": text,
                    "page_id": page_id,
                    "filename": filename
                }
                basic_retriever.insert(data)
                # Finish
                print("Finish !!!\n")
    if no_files:
        return_message = "There are no additional files that can be indexed into the vector database !!!"
        print(return_message)
        return {"response":return_message}
    return {"response":"DONE"}