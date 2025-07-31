from colpali_engine.utils.torch_utils import ListDataset
from torch.utils.data import DataLoader
from typing import List
from tqdm import tqdm
from PIL import Image
from setup import *
import cv2
import os
import shutil

def layout_detection(img_doc):
    return yolo_model.predict(
        source=img_doc, 
        classes=[6,8],
        conf=0.25, 
        iou=0.45)

def extract_and_masking_images(img_doc, layout_results):
    height, width, _ = img_doc.shape
    extracted_imgs = []
    for box in layout_results[0].boxes:
        x, y, w, h = map(int,box.xywh[0])  # Box coordinates (center x, y, width, height)
        # Calculate top-left corner (x_min, y_min)
        x_min = x - w // 2
        y_min = y - h // 2
        x_max = x_min + w
        y_max = y_min + h
        # Clamp coordinates to image boundaries
        x_start = max(0, x_min)
        y_start = max(0, y_min)
        x_end = min(width, x_max)
        y_end = min(height, y_max)
        # Skip if region is invalid
        if x_start >= x_end or y_start >= y_end:
            continue
        # Extract the image to the array of extracted_imgs
        extracted_imgs.append(img_doc[y_start:y_end, x_start:x_end].copy())
        # Set region to white
        img_doc[y_start:y_end, x_start:x_end] = [255, 255, 255]
    return extracted_imgs, img_doc

def create_image_embedding_loader(extracted_imgs):
    images = [Image.fromarray(img_arr) for img_arr in extracted_imgs]
    dataloader_images = DataLoader(
        dataset=ListDataset[str](images),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x),
    )
    ds: List[torch.Tensor] = []
    for batch_doc in dataloader_images:
        with torch.no_grad():
            batch_doc = {k: v.to(colpali_model.device) for k, v in batch_doc.items()}
            embeddings_doc = colpali_model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
    return ds

def save_img_files(extracted_imgs, filename, page_id):
    # Target path
    save_path = os.path.join(img_dir, filename, f"page_{page_id}/")
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save the image
    for i in range(len(extracted_imgs)):
        cv2.imwrite(save_path+f"fig_{i}.jpg", extracted_imgs[i])

def flush_directory(dir_path):
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)  # remove file or symlink
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # remove directory and all contents
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        print(f"Directory {dir_path} does not exist.")

def url_conversion(img_base64):
    return f"data:image/jpeg;base64,{img_base64}"

def llama4_inference(messages, token=1024):
    completion = client_groq.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        messages=messages,
        temperature=0.1,
        max_completion_tokens=token,
        top_p=1,
        stream=True,
        stop=None,
    )
    inference_result = ""
    for chunk in completion:
        chunk_inference = chunk.choices[0].delta.content or ""
        inference_result += chunk_inference
    text = inference_result
    return text
