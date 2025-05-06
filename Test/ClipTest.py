from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import requests
from io import BytesIO
import multiprocessing

# 从网络加载图像
image_url = "https://images.unsplash.com/photo-1579353977828-2a4eab540b9a?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1000&q=80"  # 一张树的图片
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# 定义多个文本
texts = ["a tree", "a dog", "a house"]  # 可以只用一个文本，也可以用多个

def process_image_text_similarity():
    # 在函数内加载模型，避免多进程问题
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", local_files_only=False)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", local_files_only=False)
    
    # 图像和文本预处理
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

    # 获取模型输出
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # 形状为 [1, len(texts)]
    print("Logits before softmax:", logits_per_image)

    # 如果有多个文本，计算softmax概率
    probs = logits_per_image.softmax(dim=1)
    print("Logits after softmax:", probs)

    # 打印每个文本的相似度
    for i, text in enumerate(texts):
        similarity_score = probs[0][i].item()
        print(f"图片和文本'{text}'的相似度为: {similarity_score:.4f}")

    # 如果只关心单一文本的原始分数
    print("\n单一文本的原始相似度分数（无需softmax）：")
    for i, text in enumerate(texts):
        raw_score = logits_per_image[0][i].item()
        print(f"图片和文本'{text}'的原始分数为: {raw_score:.4f}")










if __name__ == '__main__':
    # 这个条件检查确保代码只在直接运行此脚本时执行
    # 而不是在被其他脚本导入时执行
    # 这对于多进程应用至关重要，防止子进程重复执行主程序
    
    # freeze_support() 解决Windows平台特有的多进程问题
    # 它允许在Windows上正确启动多进程应用
    # 如果没有它，transformers库内部使用的多进程将导致错误
    multiprocessing.freeze_support()
    
    # 开始处理图像与文本的相似度计算
    process_image_text_similarity()