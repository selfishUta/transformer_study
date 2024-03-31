# 人工智能 *transformers* 库学习笔记

## 简介
transformers是一个先进的机器学习工具。
transformers支持pytorch, tensorflow和JAX。
transformers可以下载并训练预训练模型。
transformers可以处理NLP，CV，音频，以及多模态任务。
### NLP:
    - 文本分类
    - 命名实体识别
    - 问答
    - 语言建模
    - 摘要
    - 翻译
    - 多项选择
    - 文本生成
### CV：
    - 图像分类
    - 目标检测
    - 语义分割
### 音频
    - 自动语音识别
    - 音频分类
### 多模态
    - 表格问答
    - OCR
    - 从扫描文档提取信息
    - 视频分类
    - 视觉问答

## 生产环境
模型可以导出为 *ONNX* 和 *TorchScript* 格式，用于在生产环境下部署

## 学习方法
    1. 安装和使用transformers
    2. transformers基础
    3. 预训练模型微调，创建和分享
    4. 任务，模型，基础概念和思想
    5. API解读：配置，模型，分词器，流水线

### 开始使用
    transformers的快速使用和安装，便于配置和运行

### 教程
    适合初学者开始学习，帮助初学者学到使用transformers的基本技能
### 操作指南
    如何实现一个特定目标，比如为语言建模微调一个预训练模型，或者创建并分享个性化模型
### 概念指南
    对transformers模型，任务和设计理念背后的基本概念和思想的解释
### API介绍：
    所有的类和函数
    
    1. 主要类别：配置，模型，分词器，流水线
    2. 模型 每个模型实现有关的类和函数
    3. 内部帮助 详述了内部使用的工具类和函数

## transformers中的模型

|模型|pytorch|tensorflow|flax|
|----|----|----|----|
|ALBERT	|✅|	✅|	✅|
|ALIGN	|✅|	❌|	❌|
|AltCLIP	|✅|	❌|	❌|
|Audio Spectrogram Transformer	|✅	|❌	|❌|
|Autoformer|	✅|	❌|	❌|
|Bark|	✅|	❌|	❌|
|BART|	✅|	✅|	✅|
|BARThez|	✅|	✅|	✅|
|BARTpho|	✅|	✅|	✅|
|BEiT|	✅|	❌|	✅|
|BERT|	✅|	✅|	✅|
|Bert Generation|	✅|	❌|	❌|
|BertJapanese|	✅|	✅|	✅|
|BERTweet|	✅|	✅|	✅|
|BigBird|	✅|	❌|	✅|
|BigBird-Pegasus|	✅|	❌|	❌|
|BioGpt|	✅|	❌|	❌|
|BiT|	✅|	❌|	❌|
|Blenderbot|	✅|	✅|	✅|
|BlenderbotSmall|	✅|	✅|	✅|
|BLIP|	✅|	✅|	❌|
|BLIP-2|	✅|	❌|	❌|
|BLOOM|	✅|	❌|	✅|
|BORT|	✅|	✅|	✅|
|BridgeTower|	✅|	❌|	❌|
|BROS|	✅|	❌|	❌|
|ByT5|	✅|	✅|	✅|
|CamemBERT|	✅|	✅|	❌|
|CANINE|	✅|	❌|	❌|
|Chinese-CLIP|	✅|	❌|	❌|
|CLAP|	✅|	❌|	❌|
|CLIP|	✅|	✅|	✅|
|CLIPSeg|	✅|	❌|	❌|
|CLVP|	✅|	❌|	❌|
|CodeGen|	✅|	❌|	❌|
|CodeLlama|	✅|	❌|	✅|
|Conditional DETR|	✅|	❌|	❌|
|ConvBERT|	✅|	✅|	❌|
|ConvNeXT|	✅|	✅|	❌|
|ConvNeXTV2|	✅|	✅|	❌|
|CPM|	✅|	✅|	✅|
|CTRL|	✅|	✅|	❌|
|CvT|	✅|	✅|	❌|
|Data2VecAudio|	✅|	❌|	❌|
|Data2VecText|	✅|	❌|	❌|
|Data2VecVision|	✅|	✅|	❌|
|DeBERTa|	✅|	✅|	❌|
|DeiT|	✅|	✅|	❌|
|DePlot|	✅|	❌|	❌|
|DETA|	✅|	❌|	❌|
|DETR|	✅|	❌|	❌|
|DialoGPT|	✅|	✅|	✅|
|DiNAT|	✅|	❌|	❌|
|DINOv2|	✅|	❌|	❌|
|DistilBERT|	✅|	✅|	✅|
|DiT|	✅|	❌|	✅|
|DonutSwin|	✅|	❌|	❌|
|DPR|	✅|	✅|	❌|
|DPT|	✅|	❌|	❌|
|EfficientFormer|	✅|	✅|	❌|
|EfficientNet|	✅|	❌|	❌|
|ELECTRA|	✅|	✅|	✅|
|EnCodec|	✅|	❌|	❌|
|ERNIE|	✅|	❌|	❌|
|ErnieM|	✅|	❌|	❌|
|ESM|	✅|	✅|	❌|
|Falcon|	✅|	❌|	❌|
|FastSpeech2Conformer|	✅|	❌|	❌|
|FlauBERT|	✅|	✅|	❌|
|FLAVA|	✅|	❌|	❌|
|FNet|	✅|	❌|	❌|
|FocalNet|	✅|	❌|	❌|
|Fuyu|	✅|	❌|	❌|
|Gemma|	✅|	❌|	✅|
|GIT|	✅|	❌|	❌|
|GLPN|	✅|	❌|	❌|
|GPTBigCode|	✅|	❌|	❌|
|Graphormer|	✅|	❌|	❌|
|GroupViT|	✅|	✅|	❌|
|HerBERT|	✅|	✅|	✅|
|Hubert|	✅|	✅|	❌|
|IDEFICS|	✅|	❌|	❌|
|ImageGPT|	✅|	❌|	❌|
|Informer|	✅|	❌|	❌|
|InstructBLIP|	✅|	❌|	❌|
|Jukebox|	✅|	❌|	❌|
|LayoutLM|	✅|	✅|	❌|
|LayoutLMv2|	✅|	❌|	❌|
|LayoutLMv3|	✅|	✅|	❌|
|LayoutXLM|	✅|	❌|	❌|
|LED|	✅|	✅|	❌|
|LeViT|	✅|	❌|	❌|
|LiLT|	✅|	❌|	❌|
|LLaMA|	✅|	❌|	✅|
|Llama2|	✅|	❌|	✅|
|LLaVa|	✅|	❌|	❌|
|Longformer|	✅|	✅|	❌|
|LongT5|	✅|	❌|	✅|
|LUKE|	✅|	❌|	❌|
|LXMERT|	✅|	✅|	❌|
|M2M100|	✅|	❌|	❌|
|Marian|	✅|	✅|	✅|
|MarkupLM|	✅|	❌|	❌|
|Mask2Former|	✅|	❌|	❌|
|MaskFormer|	✅|	❌|	❌|
|MatCha|	✅|	❌|	❌|
|mBART|	✅|	✅|	✅|
|MEGA|	✅|	❌|	❌|
|Mistral|	✅|	❌|	✅|
|Mixtral|	✅|	❌|	❌|
|mLUKE|	✅|	❌|	❌|
|MMS|	✅|	✅|	✅|
|MobileBERT|	✅|	✅|	❌|
|MobileNetV1|	✅|	❌|	❌|
|MobileNetV2|	✅|	❌|	❌|
|MobileViT|	✅|	✅|	❌|
|MobileViTV2|	✅|	❌|	❌|
|MPNet|	✅|	✅|	❌|
|MPT|	✅|	❌|	❌|
|MRA|	✅|	❌|	❌|
|MT5|	✅|	✅|	✅|
|MusicGen|	✅|	❌|	❌|
|MVP|	✅|	❌|	❌|
|NAT|	✅|	❌|	❌|
|Nezha|	✅|	❌|	❌|
|NLLB|	✅|	❌|	❌|
|Nougat|	✅|	✅|	✅|
|Nyströmformer|	✅|	❌|	❌|
|OneFormer|	✅|	❌|	❌|
|OpenLlama|	✅|	❌|	❌|
|OPT|	✅|	✅|	✅|
|OWLv2|	✅|	❌|	❌|
|PatchTSMixer|	✅|	❌|	❌|
|PatchTST|	✅|	❌|	❌|
|Pegasus|	✅|	✅|	✅|
|Perceiver|	✅|	❌|	❌|
|Persimmon|	✅|	❌|	❌|
|Phi|	✅|	❌|	❌|
|PhoBERT|	✅|	✅|	✅|
|Pix2Struct|	✅|	❌|	❌|
|PLBart|	✅|	❌|	❌|
|PoolFormer|	✅|	❌|	❌|
|Pop2Piano|	✅|	❌|	❌|
|ProphetNet|	✅|	❌|	❌|
|PVT|	✅|	❌|	❌|
|QDQBert|	✅|	❌|	❌|
|Qwen2|	✅|	❌|	❌|
|RAG|	✅|	✅|	❌|
|REALM|	✅|	❌|	❌|
|Reformer|	✅|	❌|	❌|
|RegNet|	✅|	✅|	✅|
|RemBERT|	✅|	✅|	❌|
|ResNet|	✅|	✅|	✅|
|RetriBERT|	✅|	❌|	❌|
|RoBERTa|	✅|	✅|	✅|
|RoCBert|	✅|	❌|	❌|
|RoFormer|	✅|	✅|	✅|
|RWKV|	✅|	❌|	❌|
|SAM|	✅|	✅|	❌|
|SeamlessM4T|	✅|	❌|	❌|
|SeamlessM4Tv2|	✅|	❌|	❌|
|SegFormer|	✅|	✅|	❌|
|SegGPT|	✅|	❌|	❌|
|SEW|	✅|	❌|	❌|
|SigLIP|	✅|	❌|	❌|
|Speech2Text|	✅|	✅|	❌|
|SpeechT5|	✅|	❌|	❌|
|Splinter|	✅|	❌|	❌|
|SqueezeBERT|	✅|	❌|	❌|
|StableLm|	✅|	❌|	❌|
|Starcoder2|	✅|	❌|	❌|
|SwiftFormer|	✅|	❌|	❌|
|Swin2SR|	✅|	❌|	❌|
|SwitchTransformers|	✅|	❌|	❌|
|T5|	✅|	✅|	✅|
|TAPAS|	✅|	✅|	❌|
|TAPEX|	✅|	✅|	✅|
|TimeSformer|	✅|	❌|	❌|
|TrOCR|	✅|	❌|	❌|
|TVLT|	✅|	❌|	❌|
|TVP|	✅|	❌|	❌|
|UL2|	✅|	✅|	✅|
|UMT5|	✅|	❌|	❌|
|UniSpeech|	✅|	❌|	❌|
|UniSpeechSat|	✅|	❌|	❌|
|UnivNet|	✅|	❌|	❌|
|UPerNet|	✅|	❌|	❌|
|VAN|	✅|	❌|	❌|
|VideoMAE|	✅|	❌|	❌|
|ViLT|	✅|	❌|	❌|
|VipLlava|	✅|	❌|	❌|
|VisionTextDualEncoder|	✅|	✅|	✅|
|VisualBERT|	✅|	❌|	❌|
|ViT|	✅|	✅|	✅|
|VitDet|	✅|	❌|	❌|
|ViTMAE|	✅|	✅|	❌|
|ViTMatte|	✅|	❌|	❌|
|ViTMSN|	✅|	❌|	❌|
|VITS|	✅|	❌|	❌|
|ViViT|	✅|	❌|	❌|
|Wav2Vec2|	✅|	✅|	✅|
|Wav2Vec2Phoneme|	✅|	✅|	✅|
|WavLM|	✅|	❌|	❌|
|Whisper|	✅|	✅|	✅|
|XGLM|	✅|	✅|	✅|
|XLM|	✅|	✅|	❌|
|XLNet|	✅|	✅|	❌|
|YOLOS|	✅|	❌|	❌|
|YOSO|	✅|	❌|	❌|
|PEGASUS-X	|✅	|❌	|❌|
|RoBERTa-PreLayerNorm	|✅	|✅	|✅|
|SEW-D	|✅	|❌	|❌|
|Speech Encoder decoder	|✅	|❌	|✅|
|Swin Transformer	|✅	|✅	|❌|
|Swin Transformer V2	|✅	|❌	|❌|
|DeBERTa-v2	|✅	|✅	|❌|
|XLSR-Wav2Vec2	|✅	|✅	|✅|
|CPM-Ant	|✅	|❌	|❌|
|XLM-V	|✅	|✅	|✅|
|XLM-RoBERTa-XL	|✅	|❌	|❌|
|XLM-RoBERTa	|✅	|✅	|✅|
|XLM-ProphetNet	|✅	|❌	|❌|
|XLS-R	|✅	|✅	|✅|
|X-MOD	|✅	|❌	|❌|
|Decision Transformer	|✅	|❌	|❌|
|X-CLIP	|✅	|❌	|❌|
|ViT Hybrid	|✅	|❌	|❌|
|Vision Encoder decoder	|✅	|✅	|✅|
|Transformer-XL	|✅	|✅	|❌|
|Trajectory Transformer	|✅	|❌	|❌|
|Wav2Vec2-Conformer	|✅	|❌	|❌|
|Wav2Vec2-BERT	|✅	|❌	|❌|
|Time Series Transformer	|✅	|❌	|❌|
|Table Transformer	|✅	|❌	|❌|
|Depth Anything	|✅	|❌	|❌|
|Deformable DETR	|✅	|❌	|❌|
|Encoder decoder	|✅	|✅	|✅|
|FairSeq Machine-Translation	|✅	|❌	|❌|
|FLAN-UL2	|✅	|✅	|✅|
|FLAN-T5	|✅	|✅	|✅|
|Funnel Transformer	|✅	|✅	|❌|
|GPT Neo	|✅	|❌	|✅|
|GPT NeoX	|✅	|❌	|❌|
|GPT NeoX Japanese	|✅	|❌	|❌|
|GPT-J	|✅	|✅	|✅|
|GPT-Sw3	|✅	|✅	|✅|
|GPTSAN-japanese	|✅	|❌	|❌|
|I-BERT	|✅	|❌	|❌|
|KOSMOS-2	|✅	|❌	|❌|
|M-CTC-T	|✅	|❌	|❌|
|MADLAD-400	|✅	|✅	|✅|
|mBART-50	|✅	|✅	|✅|
|Megatron-BERT	|✅	|❌	|❌|
|Megatron-GPT2	|✅	|✅	|✅|
|MGP-STR	|✅	|❌	|❌|
|NLLB-MOE	|✅	|❌	|❌|
|OpenAI GPT	|✅	|✅	|❌|
|OpenAI GPT-2	|✅	|✅	|✅|
|OWL-ViT	|✅	|❌	|❌|
|T5v1.1	|✅	|✅	|✅|
|以上共包含260个模型|

## 快速上手
pipeline:   
如何使用pipeline进行推理？

AutoClass   
如何使用AutoClass加载PLM

Pytorch/Tensorflow  
如何使用 Pytorch/Tensorflow 训练模型？

### 安装
安装必要的库和ML框架    
pip install transformers datasets   
pip install torch   
pip install tensorflow

### pipeline
pipeline是使用预训练模型最简单的方式    
以下是使用pipeline可以处理的任务    
|任务|描述|模态|PipeLine|
|-|-|-|-|
|文本分类|	为给定的文本序列分配一个标签|	NLP|	pipeline(task=“sentiment-analysis”)|
|文本生成|	根据给定的提示生成文本|	NLP|	pipeline(task=“text-generation”)|
|命名实体识别|	为序列里的每个 token 分配一个标签（人, 组织, 地址等等）|NLP|pipeline(task=“ner”)|
|问答系统|	通过给定的上下文和问题,在文本中提取答案|NLP|pipeline(task=“question-answering”)|
|掩盖填充|	预测出正确的在序列中被掩盖的token|	NLP|	pipeline(task=“fill-mask”)|
|文本摘要|	为文本序列或文档生成总结|	NLP|	pipeline(task=“summarization”)|
|文本翻译|	将文本从一种语言翻译为另一种语言|	NLP|	pipeline(task=“translation”)|
|图像分类|	为图像分配一个标签| Computer vision|	pipeline(task=“image-classification”)|
|图像分割|	为图像中每个独立的像素分配标签（支持语义、全景和实例分割）|	Computer vision|	pipeline(task=“image-segmentation”)|
|目标检测|	预测图像中目标对象的边界框和类别|	Computer vision|	pipeline(task=“object-detection”)|
|音频分类|	给音频文件分配一个标签|	Audio|	pipeline(task=“audio-classification”)|
|自动语音识别|	将音频文件中的语音提取为文本|	Audio|	pipeline(task=“automatic-speech-recognition”)|
|视觉问答|	给定一个图像和一个问题，正确地回答有关图像的问题|	Multimodal|	pipeline(task=“vqa”)|

## 使用pipeline进行情感分析

    from transformers import pipeline
    
    classifier = pipeline("sentiment-analysis")
    result_1 = classifier("我喜欢打游戏。")
    result_2 = classifier([
        "我爱吃巧克力。",
        "我讨厌吃苦瓜。"
    ])

## 使用pipeline进行自动语音识别
    import torch
    from transformers import pipeline
    from datasets import load_dataset, Audio

    speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

    dataset = load_dataset("PolyAI/minds14", name="zh-CN", split="train")

    dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))

    result = speech_recognizer(dataset[:4]["audio"])
    print([d["text"] for d in result])

pipeline 生成器

## AutoClass
AutoClass是一个通过名称和路径查找模型的工具

## AutoTokenizer
### 分词器
分词器可以将文本转换为可以输入模型的数字形式    
首先，分词器将文本分割成单词或子单词token   
其次，通过查询字典将token转换为索引id   
transformers中主要有三种分词器：
- BPE
- WordPiece
- SentencePiece

### 预分词器
spaCy和Moses是两个基于规则的分词器。    
Transformer XL使用了空格和标点符号分词，结果产生了大小为267735的词典。  
尺寸太大的词典会增加内存使用量，提高时间复杂度。    
transformers模型几乎没有词典容量大于50000的。   
transformers使用了子词分词法。  
token粒度太大对内存资源消耗大，词典太大。
token粒度太小，不容易学习到东西。

### <font color='orange'>BPE</font>

AKA：Byte-Pair Encoding。   
来源：Neural Machine Translation of Rare Words with Subword Units (Sennrich et al., 2015)   

首先，BPE通过预分词器将文本数据分割成单词。     
【预分词器的结果】是【单词token】以及其【词频】 

例，GPT-2, RoBERTa通过空格来预分词。    
例，XLM，FlauBERT基于规则来预分词。FlauBERT的预分词使用了Moses。    
例，GPT通过Spacy和ftfy统计了训练语料库中每个单词的词频。

其次，BPE产生一个基础词典，通过学习融合规则，组合基础字典中的token来形成一个新的token。     
BPE需要人为设定一个期望字典大小的超参数。   
BPE会一直学习，直到词典大小满足期望。   

### 例：    
假设已获取预分词之后的单词词频集合如下：    
    
    [
        ("hug", 10),
        ("pug", 5),
        ("pun", 12),
        ("bun", 4),
        ("hugs", 5)
    ]

通过单词词频得到基础词典：

    ["b", "g", "h", "n", "p", "s", "u"]

将单词分割成基础词典中的符号：

    [
        ("h" "u" "g", 10), 
        ("p" "u" "g", 5), 
        ("p" "u" "n", 12), 
        ("b" "u" "n", 4), 
        ("h" "u" "g" "s", 5)
    ]

BPE统计每个可能的符号对的频次   
然后，挑选出词频最高的符号对    
例，"hu"在词频中出现了10+5 = 15次   
在上述例子中，最频繁的组合是"ug"。  
因此，BPE学到的第一个融合规则是将基础字典中所有相邻的'u'和'g'合并。     
第一次融合后的字典如下：  

    [
        ("h" "ug", 10), 
        ("p" "ug", 5), 
        ("p" "u" "n", 12), 
        ("b" "u" "n", 4), 
        ("h" "ug" "s", 5)
    ]
第一次融合后的基础字典：    
    
    ["b", "g", "h", "n", "p", "s", "u", "ug"]
次频繁的组合是"un"。    
第二次融合后的字典如下：    

    [
        ("h" "ug", 10), 
        ("p" "ug", 5), 
        ("p" "un", 12), 
        ("b" "un", 4), 
        ("h" "ug" "s", 5)
    ]
第二词融合后的基础字典：    

    ["b", "g", "h", "n", "p", "s", "u", "ug","un"]

如果基础字典不包含某个新词，那么下游分词任务中，新词将会被替换成 '\<unk\>'  
在实际操作中，表情符号通常会被识别为'\<unk\>'   

通过不断迭代，直到基础字典的大小达到设定的阈值，BPE融合规则停止学习。

GPT的基础词典大小：40478    
GPT有478个基础词典内的字符，在40000次融合后停止训练。

### Byte-level BPE
包含所有可能基础字符的基础字典太大了。  
Unicode字符有一百多万个字符。显示生成包含一百多万字符的字典是不切实际的。   

GPT-2使用了字节作为基础词典，以解决基础字典太大的问题。     
一个字节有八比特位，也就是256个字符。可以包含所有的字符。   
同时，GPT-2使用了其它规则来处理标点符号，以使GPT-2分词器可以对每个文本进行分词，而不需要使用'\<unk>'符号。  

GPT-2词典大小：50257    
256个基础token
1个文本结束token
通过50000次融合学习

### <font color="orange">WordPiece</font>

WordPiece是子词分词算法。   
paper: Japanese and Korean Voice Search (Schuster et al., 2012)
使用了WordPiece的模型： 
- BERT
- DistilBERT
- Electra

首先，初始化一个词典，该词典包含语料库中的每个字符。    
其次，迭代学习一个给定阈值的融合规则。  
与BPE选择词频最大的符号不同，WordPiece的融合目标是选择能够最大化训练数据似然值的符号对。

### 最大化训练数据似然值
1. 找到符号对
2. 计算符号对的概率 p_3
3. 计算符号对中两个符号的概率p_1, p_2
4. 该符号对的似然值 P = p_3/p_1/p_2
5. 迭代计算出似然值P最大的符号对，进行融合。

### <font color="orange">Unigram</font>
Unigram是一个子词分词器算法。    
PAPER: Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates (Kudo, 2018)

首先，使用大量基础符号初始化基础字典。  
然后，精简每个符号来获得更小的字典。    
基础字典中包含 所有的预分词单词 和 最常见的子字符串。   

Unigram没有直接使用在transformers模型中。   
Unigram通过和SentencePiece一起联合使用。    

首先，定义Unigram的损失函数：log似然函数    
其次，定义unigram模型   
然后，对基础词典中每个字符，计算移除该字符后，损失值的升高量。
最后，unigram移除10%或者20%损失升高值最小的字符。   
重复计算损失值的升高量，移除字符，直到词典达到预期的大小。  

为了能够对任何单词进行分词，Unigram通常保留基础的字符。

在分词出现多种分词结果的情况下，unigram通常选择概率最大的分词结果。

定义unigram损失：   
语料库单词数量：N   
每个单词所有可能的分词结果：S(x)    

$$\mathcal{L} = - \sum_{i=1}^Nlog\left(\sum_{x\in S(x_i)}p(x)\right)$$

### <font color="orange">SentencePiece</font>

BPE和WordPiece都使用了预分词。  
预分词的方式都是通过空格来分词。    
而中文没有空格，可以使用中文预分词器来分词。    

XLM的预分词器可以对中文，日语和泰语进行分词。   

PAPER: SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing (Kudo et al., 2018)    
首先，将输入文本看作原始输入流。  
使用的符号集合中也包含空格。    
然后使用BPE或者Unigram来产生合适的词典。    

XLNet使用了SentencePiece。  
SentencePiece解码非常容易。 
所有的tokens可以被串联起来，然后将'_'替换为空格。

transformers库中所有使用了SentencePiece的模型，都会和unigram组合起来使用。

使用了SentencePiece的模型：
- ALBERT
- XLNet
- Marian
- T5

## 保存模型

使用PreTrainedModel.save_pretrained()对微调后的模型和分词器进行保存：   

    pt_save_dir = ".pt_save_pretrained"
    tokenizer.save_pretrained(pt_save_dir)
    pt_model.save_pretrained(pt_save_dir)

再次使用该模型，PretrainedModel.from_pretrained():

    pt_model = AutoModelForSequenceClassification.from_pretrained("./pt_save_pretrained")

### pytorch使用tensorflow模型

    from transformers import AutoModel
    
    tokenizer = AutoTokenizer.from_pretrained(tf_save_dir)
    pt_model = AutoModelForSequenceClassification.from_pretrained(tf_save_dir, from_tf=True)

### tensorflow使用pytorch模型

    from transformers import TFAutoModel
    
    tokenizer = AutoTokenizer.from_pretrained(pt_save_dir)
    tf_model = TFAutoModelForSequenceClassification.from_pretrained(pt_save_dir, from_pt=True)

## 自定义模型（预训练，一般情况下不靠考虑，除非公司领导是冤大头）

通过修改模型配置类来修改模型    
配置指明了模型属性，例如隐藏层数量，注意力头数量。  

修改模型注意力头数量：
    
    from transformers import AutoConfig

    my_config = AutoConfig.from_pretrained("distilbert/distilbert-base-uncased", n_heads=12)

使用AutoModle.from_config()来创建模型

    from transformers import AutoModel
    
    my_model = AutoModel.from_config(my_config)

tensorflow创建模型

    from transformers import TFAutoModel
    
    my_model = TFAutoModel.from_config(my_config)

## LLM在企业的应用
大模型可以自动化的生成各种类型和形式的营销内容。    
文本，图片，视频，音频  
LLM可以减少人工的创作成本和时间。   
LLM可以提高内容的质量和吸引力。 









