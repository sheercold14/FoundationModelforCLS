import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class TextEncoder(nn.Module):
    def __init__(self, bert_name='bert-base-uncased', out_channels=256) -> None:
        super(TextEncoder, self).__init__()
        # 加载预训练BERT模型
        self.bert = BertModel.from_pretrained(bert_name,resume_download=True).cuda()
        #self.tokenizer = BertTokenizer.from_pretrained(bert_name)

        # 适配器（Adapter）模块
        self.adapter = Adapter(c_in=768).cuda()  # BERT 的输出维度是 768
        self.adapter_alpha = nn.parameter.Parameter(torch.tensor(0.2)).cuda()  # 可训练的适配器权重

        # 最后的投影层，将BERT的输出投影到目标空间
        self.adapter_projection = nn.Sequential(
            nn.Linear(768, out_channels),  # 768是BERT的输出维度
            nn.ReLU(inplace=True)
        ).cuda()
        
        # 冻结BERT的参数，只有适配器会进行训练
        for name, param in self.named_parameters():
            if "adapter" not in name:
                param.requires_grad_(False)

    def forward(self, texts):
        # 使用BERT Tokenizer进行文本的Tokenize
        #inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512).to(next(self.parameters()).device)
        inputs = texts
        # 获取BERT的输出 (最后一层的隐层状态)
        with torch.no_grad():  # 在这里我们不进行BERT的训练
            outputs = self.bert(**inputs)
        
        # 取BERT的[CLS]标记的输出 (通常用于文本分类)
        text_feat = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, 768)

        # 通过适配器调整文本特征
        text_adapt = self.adapter(text_feat)
        
        # 混合原始特征和适配器输出，使用适配器权重（alpha）
        text_feat = self.adapter_alpha * text_adapt + (1 - self.adapter_alpha) * text_feat
        
        # 将特征投影到目标输出空间
        text_feat = self.adapter_projection(text_feat)
        
        return text_feat