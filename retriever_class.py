
from pymilvus import MilvusClient, DataType
import numpy as np
import concurrent.futures
import os
import base64

class MilvusColbertRetriever:
    def __init__(self, milvus_client, collection_name, img_dir, dim=128):
        # Initialize the retriever with a Milvus client, collection name, and dimensionality of the vector embeddings.
        # If the collection exists, load it.
        self.collection_name = collection_name
        self.client = milvus_client
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.load_collection(collection_name)
        self.dim = dim
        self.img_dir = img_dir

    def create_collection(self):
        # Create a new collection in Milvus for storing embeddings.
        # Drop the existing collection if it already exists and define the schema for the collection.
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field(field_name="seq_id", datatype=DataType.INT16)
        schema.add_field(field_name="doc_id", datatype=DataType.INT64)
        schema.add_field(field_name="doc", datatype=DataType.VARCHAR, max_length=65535)
        self.client.create_collection(collection_name=self.collection_name, schema=schema)

    def create_index(self):
        # Create an index on the vector field to enable fast similarity search.
        # Releases and drops any existing index before creating a new one with specified parameters.
        self.client.release_collection(collection_name=self.collection_name)
        self.client.drop_index(collection_name=self.collection_name, index_name="vector")
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_name="vector_index",
            index_type="IVF_FLAT",  
            metric_type="IP",  
        )
        self.client.create_index(collection_name=self.collection_name, index_params=index_params, sync=True)

    def search(self, data, topk):
        # Perform a vector search on the collection to find the top-k most similar documents.
        search_params = {"metric_type": "IP", "params": {}}
        results = self.client.search(
            self.collection_name,
            data,
            limit=int(50),
            output_fields=["vector", "seq_id", "doc_id","$meta"],
            search_params=search_params,
        )
        doc_meta = {}
        for r_id in range(len(results)):
            for r in range(len(results[r_id])):
                entity = results[r_id][r]["entity"]
                doc_id = entity["doc_id"]
                if doc_id not in doc_meta:
                    doc_meta[doc_id] = {
                        "page_id": entity["page_id"],
                        "fig_id": entity["fig_id"],
                        "filename": entity["filename"],
                    }
        scores = []

        def rerank_single_doc(doc_id, data, client, collection_name):
            # Rerank a single document by retrieving its embeddings and calculating the similarity with the query.
            doc_colbert_vecs = client.query(
                collection_name=collection_name,
                filter=f"doc_id in [{doc_id}]",
                output_fields=["seq_id", "vector", "doc"],
                limit=1000,
            )
            doc_vecs = np.vstack(
                [doc_colbert_vecs[i]["vector"] for i in range(len(doc_colbert_vecs))]
            )
            score = np.dot(data, doc_vecs.T).max(1).sum()
            return (score, doc_id)

        with concurrent.futures.ThreadPoolExecutor(max_workers=300) as executor:
            futures = {
                executor.submit(
                    rerank_single_doc, doc_id, data, self.client, self.collection_name
                ): doc_id
                for doc_id in doc_meta.keys()
            }
            for future in concurrent.futures.as_completed(futures):
                score, doc_id = future.result()
                meta = doc_meta[doc_id]
                img_path = os.path.join(self.img_dir, meta["filename"], f"page_{meta['page_id']}", f"fig_{meta['fig_id']}.jpg")
                with open(img_path, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode('utf-8')
                scores.append({
                    "score":float(score), 
                    "page_id": meta["page_id"],
                    "fig_id": meta["fig_id"],
                    "filename": meta["filename"],
                    "content": img_base64})

        scores.sort(key=lambda x: x["score"], reverse=True)
        if len(scores) >= topk:
            return scores[:topk]
        else:
            return scores

    def insert(self, data):
        # Insert ColBERT embeddings and metadata for a document into the collection.
        # Insert the data as multiple vectors (one for each sequence) along with the corresponding metadata.
        colbert_vecs = [vec for vec in data["colbert_vecs"]]
        seq_length = len(colbert_vecs)
        self.client.insert(
            self.collection_name,
            [
                {
                    "vector": colbert_vecs[i],
                    "seq_id": i,
                    "doc_id": data["doc_id"] ,
                    "doc": "",
                    "page_id": data["page_id"],
                    "fig_id": data["fig_id"],
                    "filename": data["filename"],
                }
                for i in range(seq_length)
            ],
        )

class MilvusBasicRetriever:
    def __init__(self, milvus_client, collection_name, dim=1024):
        # Initialize the retriever with a Milvus client, collection name, and dimensionality of the vector embeddings.
        # If the collection exists, load it.
        self.collection_name = collection_name
        self.client = milvus_client
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.load_collection(collection_name)
        self.dim = dim

    def normalize(self, vec):
        # Normalize the vector
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def create_collection(self):
        # Create a new collection in Milvus for storing embeddings.
        # Drop the existing collection if it already exists and define the schema for the collection.
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        self.client.create_collection(collection_name=self.collection_name, schema=schema)

    def create_index(self):
        # Create an index on the vector field to enable fast similarity search.
        # Releases and drops any existing index before creating a new one with specified parameters.
        self.client.release_collection(collection_name=self.collection_name)
        self.client.drop_index(collection_name=self.collection_name, index_name="vector")
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_name="vector_index",
            index_type="IVF_FLAT",  # or any other index type you want
            metric_type="IP",  # or the appropriate metric type
        )
        self.client.create_index(collection_name=self.collection_name, index_params=index_params, sync=True)

    def search(self, data, topk):
        # Perform a vector search on the collection to find the top-k most similar documents.
        normalized_data = self.normalize(data)
        search_params = {"metric_type": "IP", "params": {}}
        results = self.client.search(
            self.collection_name,
            [normalized_data],
            limit=topk,
            output_fields=["vector", "content","$meta"],
            search_params=search_params,
        )
        return_arr = []
        for hit in results[0]:
            return_arr.append({
                "score":hit.distance,
                "page_id":hit["entity"]["page_id"],
                "filename":hit["entity"]["filename"],
                "content":hit["entity"]["content"]
            })
        return return_arr

    def insert(self, data):
        data["vector"] = self.normalize(np.array(data["vector"])).tolist()
        self.client.insert(
            self.collection_name,
            [data]
        )