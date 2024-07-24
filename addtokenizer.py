import jieba
from collections import Counter
import json
import re
def clean_data(json_data):
    regex = re.compile(r'[^\w\s]|_|\d', re.UNICODE)

    data_dict = json.loads(json_data)
    
    # 初始化一个空列表来存储每个键值对的文本
    text_list = []
    
    # 遍历字典，将每个键值对转换为文本
    for key, value in data_dict.items():
        text_list.append(f"{key}: {value}")
    
    # 将列表中的所有文本拼接成一个字符串，每个键值对之间用逗号和空格分隔
    text = ", ".join(text_list)
    cleaned_value = regex.sub('', text)
    return cleaned_value
def get_keys():
    with open('dataset.json', 'r') as f:
        data = json.load(f)
        dicts = []
        for item in data:
            dict1 = jieba.lcut(clean_data(item['input_text']))
            regex = re.compile(r'[^\w\s]|_|\d', re.UNICODE)
            cleaned_value = regex.sub('', item['target_text'])
            dict2 = jieba.lcut(cleaned_value)
            dicts.extend(dict1)
            dicts.extend(dict2)
        counter = Counter(dicts)
    keys_list = list(counter.keys())
    for i in range(10):
        keys_list.append(str(i))
    return keys_list