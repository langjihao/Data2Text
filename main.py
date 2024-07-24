from datasets import load_dataset
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments,DataCollatorWithPadding
from addtokenizer import get_keys
import random
# 加载数据集
dataset = load_dataset('json',data_files="dataset.json")

# 分割数据集为训练集s、验证集和测试集
train_test_split = dataset['train'].train_test_split(test_size=0.2)
train_val_split = train_test_split['train'].train_test_split(test_size=0.25)

train_dataset = train_val_split["train"]
val_dataset = train_val_split["test"]
test_dataset = train_test_split["test"]
def clean_data(json_data):
    clean_list = []
    # 将 JSON 字符串转换为字典
    for item in json_data:
        data_dict = json.loads(item)
        del data_dict['斑块情况']
        del data_dict['狭窄程度']
        # 初始化一个空列表来存储每个键值对的文本
        text_list = []
        
        # 遍历字典，将每个键值对转换为文本
        for key, value in data_dict.items():
            text_list.append(f"{key}: {value}")
        random.shuffle(text_list)
        # 将列表中的所有文本拼接成一个字符串，每个键值对之间用逗号和空格分隔
        text = ", ".join(text_list)
        text = "生成一份报告：" + text 
        clean_list.append(text)
    return clean_list
def shuffle_text(text):
    shuffle_list = []
    for item in text:
        text_list = item.split('，')
        random.shuffle(text_list)
        # 使用join(',')将列表合并成新的字符串
        targets = ','.join(text_list)
        shuffle_list.append(targets)
    return shuffle_list
# 准备数据的函数
def preprocess_function(examples):
    inputs = clean_data(examples['input_text'])
    targets = shuffle_text(examples['target_text'])
    model_inputs = tokenizer(inputs, max_length=128, padding='max_length', truncation=True)
    
    # 设置 decoder_input_ids
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, padding='max_length', truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    print(type(model_inputs))
    return model_inputs
new_keys = get_keys()
# 加载数据集
tokenizer = T5Tokenizer.from_pretrained("lemon234071/t5-base-Chinese")
tokenizer.add_tokens(new_keys)
# 使用tokenizer处理数据
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)

# 数据收集器
data_collator = DataCollatorWithPadding(tokenizer)
# 初始化tokenizer和模型

model = T5ForConditionalGeneration.from_pretrained("lemon234071/t5-base-Chinese")
model.resize_token_embeddings(len(tokenizer))
# 定义训练参数
class MyTrainer(Trainer):
    def on_epoch_end(self, args, state, control, **kwargs):
        # 调用父类的on_epoch_end方法，确保其他默认行为不受影响
        super().on_epoch_end(args, state, control, **kwargs)
        
        # 在每个epoch结束时评估测试集
        test_results = self.evaluate(eval_dataset=test_dataset)
        print(f"Test results at epoch {state.epoch}: {test_results}")

# 确保你已经定义了model, tokenized_train_dataset, tokenized_val_dataset, data_collator, 和 test_dataset

training_args = TrainingArguments(
    output_dir="/hy-tmp/results",
    num_train_epochs=50,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
)

# 使用自定义的Trainer
trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator
)

# 训练模型
trainer.train()

# 保存模型
model.save_pretrained("/hy-tmp/checkpoint")

# # 待编码的字符串
# text = "MLA为2.52mm²，MLD为0.9mm，狭窄长度为5.69mm，狭窄率为70%，属于重度狭窄，可见存在红血栓，钙化斑块处于高危水平，未发现斑块破裂现象。"

# # 使用tokenizer编码然后解码
# encoded_input = tokenizer.encode(text, add_special_tokens=True)
# decoded_output = tokenizer.decode(encoded_input)

# print("原始字符串:", text)
# print("编码后的token IDs:", encoded_input)
# print("解码后的字符串:", decoded_output)