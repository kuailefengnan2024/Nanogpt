from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# 加载模型和处理器
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# 加载图像
image_path = "G:/Nanogpt/Test/173589294072843_P16209623.jpg"  # 替换为图片路径
image = Image.open(image_path)

# 定义多个文本
texts = ["a tree", "a dog", "a house"]  # 可以只用一个文本，也可以用多个

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