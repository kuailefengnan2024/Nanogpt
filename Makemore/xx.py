import torch


# 定义一个简单的词汇表类
class SimpleVocab:
    def __init__(self, words, unk_token='<unk>'):
        self.unk_token = unk_token
        self.stoi = {s: i+1 for i, s in enumerate(words)} # 索引从 1 开始
        self.stoi[self.unk_token] = 0                     # 为 unknown token 分配索引 0
        self.itos = {i: s for s, i in self.stoi.items()}   # 创建反向映射

    def __getitem__(self, token):
        """获取 token 对应的 index (stoi)。默认为 unk index。"""
        return self.stoi.get(token, self.stoi[self.unk_token])

    def lookup_token(self, index):
        """获取 index 对应的 token (itos)。默认为 unk token。"""
        return self.itos.get(index, self.unk_token)

    def encode(self, tokens):
        """将 token 列表转换为 index 列表。"""
        return [self[token] for token in tokens]

    def decode(self, indices):
        """将 index 列表转换为 token 列表。"""
        # 处理 tensor 输入
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return [self.lookup_token(idx) for idx in indices]

# --- 主脚本逻辑 ---

# 定义词汇表单词
word_list = ['hi', 'torch', 'python', 'is', 'fun']

# 创建 vocab 实例
vocab = SimpleVocab(word_list)

# 使用 vocab 实例
i = vocab['hi']                       # 使用 __getitem__ 实现 stoi 功能
t = vocab.lookup_token(i)             # 使用 lookup_token 实现 itos 功能
x = torch.tensor(vocab.encode(['hi', 'torch'])) # 使用 encode 实现 list[str] -> Tensor[int]
y = vocab.decode(x)                   # 使用 decode 实现 Tensor[int] -> list[str]

print(vocab.stoi) # Print the string-to-index mapping
print(i, t, x, y)

# --- 更多调用示例 ---

# 1. 使用 encode 将 token 列表转换为 index 列表
tokens_to_encode = ['python', 'is', 'fun', 'unknown_word']
encoded_indices = vocab.encode(tokens_to_encode)
print(f"Encoded indices for {tokens_to_encode}: {encoded_indices}") # 应该输出类似 [3, 4, 5, 0] (unknown_word 映射到 <unk> 的索引 0)

# 2. 使用 decode 将 index 列表转换为 token 列表
indices_to_decode = [1, 2, 0, 5]
decoded_tokens = vocab.decode(indices_to_decode)
print(f"Decoded tokens for {indices_to_decode}: {decoded_tokens}") # 应该输出类似 ['hi', 'torch', '<unk>', 'fun']

# 3. 使用 decode 将 PyTorch Tensor 转换为 token 列表
tensor_to_decode = torch.tensor([3, 4, 5])
decoded_from_tensor = vocab.decode(tensor_to_decode)
print(f"Decoded tokens for tensor {tensor_to_decode}: {decoded_from_tensor}") # 应该输出类似 ['python', 'is', 'fun']

# 4. 使用 lookup_token 获取单个 token
token_at_index_4 = vocab.lookup_token(4)
print(f"Token at index 4: {token_at_index_4}") # 应该输出 'is'
token_at_index_99 = vocab.lookup_token(99) # 查找一个不存在的 index
print(f"Token at index 99: {token_at_index_99}") # 应该输出 '<unk>'
