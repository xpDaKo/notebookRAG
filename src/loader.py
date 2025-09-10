import os
import re
from typing import List
from tqdm import tqdm
import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def pdf2documents(directory: str) -> List[Document]:
    """
    从目录directory解析所有pdf文件并提取文本内容和元数据
    :param directory: 存放pdf的notes目录
    :return: 返回包含文本+元数据的Document列表
    """
    pdf_path_list = get_pdf_path(directory)
    texts, metadatas = [], []
    for pdf_path in tqdm(pdf_path_list, total= len(pdf_path_list), desc= "解析notes中"):
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            page_text = page.get_text() # page为fitz.Page 对象，需要通过get_text获取文本内容
            page_text = remove_useless_content(page_text,pdf_path) # 去除无用字符
            metadata = {
                "file": "《" + os.path.basename(pdf_path).replace('.pdf', '') + "》",
                "page": page_num + 1 # 从1开始，更符合大部分人习惯
                }

            texts.append(page_text)
            metadatas.append(metadata)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        chunk_size=500,
        chunk_overlap=100
    )
    documents = text_splitter.create_documents(texts, metadatas=metadatas)
    return documents

def get_pdf_path(directory: str) -> List[str]:
    """
    查找目录directory下的所有pdf文件并返回所有文件的绝对路径的列表
    :param directory: 存放pdf的notes目录
    :return: 所有pdf文件绝对路径的列表
    """
    pdf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdf'):
                pdf_files.append(os.path.join(root,file))
    return pdf_files

def remove_useless_content(text: str,pdf_path: str):
    """
    通过规则来去除一些多余字符
    :param text: 文章的文本
    :param pdf_path: 文章绝对路径，用于处理一些包含文件的字符
    :return: 处理好的text
    """
    # 去除不必要的内容
    fixed_strings_to_remove = [
        "来自： AiGC面试宝典",
        "宁静致远",
        "知识星球",
    ]
    for item in fixed_strings_to_remove:
        text = text.replace(item, "")

    # 去除文章标题
    title_key = re.sub(r'^\d+[-_.]', '', os.path.basename(pdf_path).replace('.pdf', '')) # 获取不含序号的文章标题，如：大模型（LLMs）基础面
    text = re.sub(title_key, '',text)

    return text
