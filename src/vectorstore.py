from typing import List

from milvus_model.hybrid import BGEM3EmbeddingFunction


def get_embeddings(texts: List[str]):
    """
    输入文本列表，通过BGEM3EmbeddingFunction转为embeddings(并非纯embeddings格式)
    :param texts: 需要转换的文本列表
    :return: 包含embeddings的数据格式
    """
    try:
        bgem3_ef = BGEM3EmbeddingFunction(
            model_name="/data/bge-m3",
            device="cuda",
            use_fp16=False         
            )
    except Exception as e:
        print("bge-m3加载出错")
    texts_embeddings = bgem3_ef(texts)
    return texts_embeddings

def 