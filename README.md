
## 微调 llama3、qwen 等模型。

### 微调：llama3

使用 unsloth 实现 peft  来微调  llama3 lora 模型


#### 第一步安装requirement


执行下面命令
```shell
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.26" trl peft accelerate bitsandbytes
```


#### 第二步下载：模型文件

https://huggingface.co/unsloth/llama-3-8b-bnb-4bit/tree/main

```shell
tree  models/
models/
├── config.json
├── generation_config.json
├── model.safetensors
├── special_tokens_map.json
├── tokenizer_config.json
└── tokenizer.json
```


#### 第三步下载训练数据：
https://huggingface.co/datasets/yahma/alpaca-cleaned/tree/main



下载后，新增自己的中文数据：

```shell
[
    {
        "instruction": "Give three tips for staying healthy.",
        "input": "",
        "output": "1. Eat a balanced and nutritious diet: Make sure your meals are inclusive of a variety of fruits and vegetables, lean protein, whole grains, and healthy fats. This helps to provide your body with the essential nutrients to function at its best and can help prevent chronic diseases.\n\n2. Engage in regular physical activity: Exercise is crucial for maintaining strong bones, muscles, and cardiovascular health. Aim for at least 150 minutes of moderate aerobic exercise or 75 minutes of vigorous exercise each week.\n\n3. Get enough sleep: Getting enough quality sleep is crucial for physical and mental well-being. It helps to regulate mood, improve cognitive function, and supports healthy growth and immune function. Aim for 7-9 hours of sleep each night."
    },
 ]
```


模型文件 放置： models
训练数据 放置：traning_datasets
lora 模型：   lm3_lora_model


#### 运行
```python
运行： python traning.py 进行训练
运行： python inference.py 进行推理
```



整体目录：

```
.
├── inference.py
├── llama_llm.py
├── lm3_lora_model
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── README.md
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── tokenizer.json
├── models
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── tokenizer.json
├── outputs
│   └── runs
│       ├── May04_08-35-29_adminws-System-Product-Name
│       │   └── events.out.tfevents.1714782935.adminws-System-Product-Name.3233304.0
│       ├── May04_08-42-40_adminws-System-Product-Name
│       │   └── events.out.tfevents.1714783360.adminws-System-Product-Name.3249804.0
├── prompt_setting.py
├── __pycache__
│   ├── llama_llm.cpython-310.pyc
│   └── prompt_setting.cpython-310.pyc
├── traning_datasets
│   ├── alpaca_data_cleaned.json
│   └── gitattributes
└── traning.py
```


中文数据集：https://modelscope.cn/datasets/baicai003/Llama3-Chinese-dataset/summary


### 期待您的issue
