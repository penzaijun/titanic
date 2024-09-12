from model import MLP
from data import load_data,write_csv,TitanicDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import datetime

device=0
epoch=100
run_name=datetime.datetime.now().strftime("%m-%d-%H-%M")

train,test=load_data()
dataset=TitanicDataset(train)
dataloader=DataLoader(dataset,batch_size=16,shuffle=True)
dataset_test=TitanicDataset(test)
dataloader_test=DataLoader(dataset_test,batch_size=16,shuffle=False)

model=MLP([dataset.feature_dim,50,50,50,2]).to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=5e-3)
criterion=torch.nn.CrossEntropyLoss()

best_acc=0
for e in range(epoch):
    model.train()
    total_loss=0
    bar=tqdm(dataloader)
    for b_idx,d in enumerate(bar):
        feature,label=d
        feature=feature.to(device)
        label=label.to(device)

        optimizer.zero_grad()
        logit=model(feature)
        loss=criterion(logit,label)
        total_loss+=loss.item()
        loss.backward()
        optimizer.step()
        bar.set_postfix({'loss':total_loss/(b_idx+1)})
    
    #validation on train set
    model.eval()
    total=0
    cor=0
    bar=tqdm(dataloader)
    for b in bar:
        feature,label=b
        feature=feature.to(device)
        label=label.to(device)
        logit=model(feature)
        pred=torch.argmax(logit,dim=1)
        cor+=torch.sum(label==pred).item()
        total+=len(label)
        bar.set_postfix({'acc':cor/total})
    
    if cor/total>best_acc:
        print(f'NEW BEST ACC:{cor/total}')
        torch.save(model.state_dict(),f'./checkpoints/{run_name}')
        best_acc=cor/total

model.load_state_dict(torch.load(f'./checkpoints/{run_name}'))
model.eval()
idx=0
for b in dataloader_test:
    feature,_=b
    feature=feature.to(device)
    logit=model(feature)
    label_val=torch.argmax(logit,dim=1)
    for l in label_val:
        test[idx].Survived=l.item()
        idx+=1
write_csv(test)