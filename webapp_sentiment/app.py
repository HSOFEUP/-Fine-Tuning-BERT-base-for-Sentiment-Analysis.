from flask import Flask , request , render_template ,redirect , send_file
import torch
import torch.nn.functional as F
import torchvision
import torch.nn as nn
import transformers
from transformers import BertModel
from torch.utils.data import Dataset, DataLoader
import numpy as np
from io import BytesIO
import base64

import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns



import json
import pandas as pd
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")
plt.ioff()

app = Flask(__name__)

num_classes = 3
bert_model_name = 'bert-base-cased'
MAX_LEN = 128


class SentimentClassifier(nn.Module):
    def __init__(self,num_classes):
        super(SentimentClassifier,self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(p = 0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size,num_classes)
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self,input_ids , attention_mask):
        temp = self.bert(input_ids,attention_mask)
        pooled_output = temp[1]
        out = self.dropout(pooled_output)
        out = self.linear(out)
        return out

model = SentimentClassifier(num_classes)
model.load_state_dict(torch.load('best_model_state.bin',map_location=torch.device('cpu')))
tokenizer = transformers.BertTokenizer.from_pretrained(bert_model_name)
pred_class_list = ['Negative','Neutral','Positive']

@app.after_request
def add_header(response):
    response.cache_control.max_age = 0
    return response

@app.route('/',methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        sample_review = request.form['review']
        encode_sample_review = tokenizer.encode_plus(
                            sample_review,
                            add_special_tokens=True,
                            max_length=MAX_LEN,
                            truncation=True,
                            return_token_type_ids=False,
                            pad_to_max_length=True,
                            return_attention_mask=True,
                            return_tensors='pt')

        input_ids = encode_sample_review['input_ids']
        attention_mask = encode_sample_review['attention_mask']

        output = model(input_ids, attention_mask)
        _, pred_id = torch.max(output, dim=1)
        prediction = pred_class_list[pred_id]
        pred_probs = F.softmax(output,dim=1).detach().numpy().round(6).reshape(3)

        data = {'class_names':pred_class_list, 'value':pred_probs} 
        # Create DataFrame 
        df = pd.DataFrame(data)
        sns.set_style("darkgrid")
        fig = plt.figure(figsize = (6,6))
        sns.barplot(x='value', y='class_names', data=df, orient='h')
        
        plt.xlabel('probability')
        plt.xlim([0, 1])
        url = 'static/Images/img.png'
        plt.savefig(url)
        plt.close(fig)

        return render_template('index.html', 
                                review = sample_review,
                                preds = prediction,
                                Neg_prob = str(100*pred_probs[0]),
                                Neu_prob = str(100*pred_probs[1]),
                                Pos_prob = str(100*pred_probs[2]),
                                image_loc = url) 
                                
    
    else:
        return render_template('index.html',
                                review = '',
                                preds = '',
                                Neg_prob = '',
                                Neu_prob = '',
                                Pos_prob = '',
                                image_loc = None)


if __name__ == '__main__':
    app.run(debug=True)