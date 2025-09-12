from typing import List
from langchain_core.documents import Document
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import FieldSchema, DataType, CollectionSchema, Collection, utility, connections




def create_vectorstore(documents: List[Document]):
    
    bgem3_ef = BGEM3EmbeddingFunction(
            model_name= r".cache\model\BAAI\bge-m3",
            device="cpu",
            use_fp16=False         
            )
    texts = [doc.page_content for doc in documents]
    embeddings = bgem3_ef(texts)
    # 定义字段
    fields = [
        FieldSchema(name= "id", dtype= DataType.INT64, is_primary= True, auto_id= True),
        FieldSchema(name= "text", dtype= DataType.VARCHAR,  max_length= 65535),
        FieldSchema(name= "metadata", dtype= DataType.JSON),
        FieldSchema(name= "dense_vector", dtype= DataType.FLOAT_VECTOR, dim= bgem3_ef.dim["dense"]),
        FieldSchema(name= "sparse_vector", dtype= DataType.SPARSE_FLOAT_VECTOR)
    ]
    # 创建集合模式
    schema = CollectionSchema(fields= fields, description= "notebookRAG Collection hybrid")
    
    # 创建集合
    conection = connections.connect( # 连接milvus
        host= "localhost",
        port= 19530
    )
    collection_name = "notebook_1"
    collection = Collection(
        name= collection_name,
        schema= schema,
        consistency_level= "Strong" # 先将等级设置为Strong, 后续再改
    )

    # 密集索引
    dense_index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "IP", 
        "params": {"nlist": 100}
    }
    collection.create_index("dense_vector", dense_index_params)
    # 稀疏索引
    sparse_index_params = {
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "IP"
    }
    collection.create_index("sparse_vector", sparse_index_params)

    data = [
        texts, # text
        [doc.metadata for doc in documents], # metadata
        embeddings["dense"], # 密集
        embeddings["sparse"] # 稀疏
    ]
    collection.insert(data)
    collection.load() # 加载集合到内存
    return collection_name


# def get_embeddings(texts: List[str]):
#     """
#     输入文本列表，通过BGEM3EmbeddingFunction转为embeddings(并非纯embeddings格式)
#     :param texts: 需要转换的文本列表
#     :return: 包含embeddings的数据格式
#     """
#     try:
#         bgem3_ef = BGEM3EmbeddingFunction(
#             model_name="/data/bge-m3",
#             device="cuda",
#             use_fp16=False         
#             )
#     except Exception as e:
#         print("bge-m3加载出错")
#     texts_embeddings = bgem3_ef(texts)
#     return texts_embeddings