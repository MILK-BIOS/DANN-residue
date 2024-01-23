import os
import torch
from models.FC_model import FCModel
import pandas as pd
import numpy as np
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_root = os.path.join('..', 'models')
My_model=FCModel()
My_model=torch.load(os.path.join(
    model_root, 'mnist_mnistm_model_epoch_9.pth'
))
My_model = My_model.eval()
My_model=My_model.to(device)

df=pd.read_csv('D:/学习/代码/DANN/models/20_11.CSV')

col = [i for i in df.columns if i in ['VHresidue', 'VVresidue', 'product', 'ratio', 'span']]
X = df[col]
#X=X.iloc[8]
label_mapping = {'TC': 0, 'RC': 1, 'NC': 2}
if 'residue' in df:
    labels = df['residue'].map(label_mapping)
else:
    labels = pd.Series(np.zeros(len(df)))
label=labels.iloc[8]

X=np.array(X)
X = torch.tensor(X, dtype=torch.float32).to(device)
#X=X.unsqueeze(0)
class_label, domain_label = My_model(input_data=X, alpha=0)
pred = class_label.data.max(1, keepdim=True)[1]
print(pred)
print(label)
