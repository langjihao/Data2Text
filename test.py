from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config, GenerationConfig
from addtokenizer import get_keys
import json

def clean_data(json_data):
    clean_list = []
    # 将 JSON 字符串转换为字典
    data_dict = json.loads(json_data)
    del data_dict['斑块情况']
    del data_dict['狭窄程度']
    # 初始化一个空列表来存储每个键值对的文本
    text_list = []
    
    # 遍历字典，将每个键值对转换为文本
    for key, value in data_dict.items():
        text_list.append(f"{key}: {value}")
    
    # 将列表中的所有文本拼接成一个字符串，每个键值对之间用逗号和空格分隔
    text = ", ".join(text_list)
    text = "生成一份报告：" + text 
    return text

model_directory = "/hy-tmp/results/checkpoint-3500"
# 加载配置
config = T5Config.from_pretrained(model_directory)
generation_config = GenerationConfig.from_pretrained(model_directory)

# 加载模型
model = T5ForConditionalGeneration.from_pretrained(model_directory, config=config)


new_keys = get_keys()
# 加载数据集
tokenizer = T5Tokenizer.from_pretrained("lemon234071/t5-base-Chinese")
tokenizer.add_tokens(new_keys)
model.resize_token_embeddings(len(tokenizer))
# 输入文本
eval = []
with open ('dataset.json', 'r') as f:
    data = json.load(f)
for item in data[0:10]:
    input_text = clean_data(item['input_text'])
    target_text = item['target_text']
    # 预处理输入
    input_ids = tokenizer(input_text, return_tensors="pt",max_length=128, padding='max_length').input_ids
    # 生成输出
    outputs = model.generate(input_ids, max_length=128)
    # 解码输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(" ", "")
    print("Target Text: ", target_text)
    print("Pred: ",generated_text)
    eval.append({"gth":target_text, "pred":generated_text})
with open('eval.json', 'w') as f:
    json.dump(eval, f)


