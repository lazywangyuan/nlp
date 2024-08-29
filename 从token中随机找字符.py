from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('D:\models\paraphrase-multilingual-MiniLM-L12-v2_')
tokens = tokenizer.get_vocab().keys()
char_token_list = []
for char in tokens:
    if '\u4e00' <= char <= '\u9fff':
        char_token_list.append(char)
print(char_token_list)