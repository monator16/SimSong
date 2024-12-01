#%%
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

import numpy as np
import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------
# dataloaders
# ------------
from data import create_contrastive_datasets, ContrastiveDataset


# 데이터 경로 설정
dataset_dir = "/home/elicer/project/collect_data/mp3"

# 오디오 파일 경로 및 데이터셋 준비
train_dataset, val_dataset = create_contrastive_datasets(dataset_dir, train_ratio=0.8)

#Stage 1 학습 파라미터
num_epochs = 3
batch_size = 64

target_column = 'Set Index'
sim_set_dir = '/home/elicer/project/data/final_final.csv'

# ContrastiveDataset으로 변환
sample_rate = 44100 # [1, sample_rate*30]: 30초로 구간 설정
train_contrastive_dataset = ContrastiveDataset(train_dataset, sim_set_dir, target_column, input_shape=[1, sample_rate*30])
val_contrastive_dataset = ContrastiveDataset(val_dataset, sim_set_dir, target_column, input_shape=[1, sample_rate*30])

# DataLoader로 배치 생성
train_loader = DataLoader(train_contrastive_dataset, batch_size=batch_size, shuffle=True, drop_last =True)
val_loader = DataLoader(val_contrastive_dataset, batch_size=batch_size, shuffle=False, drop_last= True)
# -> 한 배치의 구성 : clip_a, clip_b, file_id

#%%
# ------------
# model
# ------------
from models import ContrastiveModel
from ast_encoder import ASTEncoder
from loss import soft_info_nce_loss, info_nce_loss
from loss_weight import generate_lyrics_embeddings, compute_similarity




# 1. 모델과 옵티마이저 초기화

ast_encoder = ASTEncoder()
ast_encoder.set_train_mode()


model = ContrastiveModel(ast_encoder)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#%%
# 2. BERT 모델 로드 (가사 임베딩용)
from transformers import BertTokenizer, BertModel

bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


#3. 학습 모델 로드
def train_weighted_negative_sampling(train_loader, model, optimizer, bert_model,tokenizer,device,num_epochs,batch_size):
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            clip_a, clip_b, file_ids, target_value = batch  # 오디오와 file_ids 로드
            clip_a, clip_b = clip_a.to(device), clip_b.to(device)

            # 1) 오디오 임베딩 생성 (오디오는 로스 계산에만 사용)
            projected_a, projected_b = model(clip_a, clip_b)

            audio_embeddings = torch.cat([projected_a, projected_b], dim=0)  # Combine along the batch dimension
            print(audio_embeddings.shape)

            # 2) 가사 임베딩 생성
            lyrics_embeddings = generate_lyrics_embeddings(file_ids, bert_model, tokenizer, device)

            # 3) 가사 임베딩들 간의 유사도 계산
            sim_ij = compute_similarity(lyrics_embeddings.repeat(2, 1))        

            # 4) 손실 계산
            loss = soft_info_nce_loss(
                features=audio_embeddings,
                sim_ij=sim_ij,
                batch_size=batch_size,
                n_views=2,
                temperature=0.5,
                device=device
            )
            loss = loss.requires_grad_()

            # 5) 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
        
    #모델 저장
    torch.save(model, 'WNS_model_1203.pth')
        

def train_pure_negative_sampling(train_loader,model,optimizer, bert_model,tokenizer, device, num_epochs,batch_size):
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            clip_a, clip_b, file_ids, target_value = batch  # 오디오와 file_ids 로드
            clip_a, clip_b = clip_a.to(device), clip_b.to(device)

            # 1) 오디오 임베딩 생성 (오디오는 로스 계산에만 사용)
            projected_a, projected_b = model(clip_a, clip_b)

            audio_embeddings = torch.cat([projected_a, projected_b], dim=0)  # Combine along the batch dimension
            print(audio_embeddings.shape)

            # 2) 가사 임베딩 생성
            lyrics_embeddings = generate_lyrics_embeddings(file_ids, bert_model, tokenizer, device)

            # 3) 가사 임베딩들 간의 유사도 계산
            sim_ij = compute_similarity(lyrics_embeddings.repeat(2, 1))        

            # 4) 손실 계산
            loss = info_nce_loss(
                features=audio_embeddings,
                batch_size=batch_size,
                n_views=2,
                temperature=0.5,
                device=device
            )
            loss = loss.requires_grad_()

            # 5) 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    #모델 저장
    torch.save(model, 'NS_model_1203.pth')


#%%
# ------------
# 학습 - stage 1
# ------------



#둘 중 하나는 빼고 돌리기 
weighting = True
if weighting:
    train_weighted_negative_sampling(train_loader, model, optimizer, bert_model,tokenizer,device,num_epochs,batch_size)
    #staage 2함수
else:
    train_pure_negative_sampling(train_loader,model,optimizer, bert_model,tokenizer, device, num_epochs,batch_size)

#%%
###############################
import torch
import torch.nn as nn

class Mlp_classifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Mlp_classifier, self).__init__()
        layers = []
        prev_size = input_size

        # Hidden Layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        # Output Layer
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())  # Sigmoid for multi-label classification

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

#pure로 돌릴 땐 바꾸기
stage1_model = torch.load("/home/elicer/project/src/WNS_model_1203.pth")
stage1_model.eval()  # Stage 1은 학습하지 않음


# 모델 초기화
from models import ContrastiveModel

projection_dim =128
arg = 'sep'
# 'sep', 'mean', 'concat'

if arg == "concat":
    input_size = 2*projection_dim
else:
    input_size = projection_dim


hidden_sizes = [128, 64]    # Hidden layer 크기
output_size = 3672          # 클래스 수

model = Mlp_classifier(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)




# Loss 및 Optimizer 설정
import torch.optim as optim
criterion = nn.BCELoss()  # BCE 손실 함수

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ------------
# 학습 - stage 2
# ------------
num_epochs = 10  # 학습 에포크 수
model.to(device)
stage1_model.to(device)

from sklearn.metrics import hamming_loss  # 해밍 손실 계산


for epoch in range(num_epochs):
    total_loss = 0  # 에포크 동안 손실 누적

    for batch in train_loader:
        clip_a, clip_b, file_ids, target_value = batch  # 오디오와 file_ids 로드
        clip_a, clip_b = clip_a.to(device), clip_b.to(device)
        import re

        # 쉼표와 공백으로 분리 후 숫자로 변환
        target_value = [
            float(num) for item in target_value for num in re.split(r'[,\s]+', item.strip()) if num
        ]
        target_value = torch.tensor(target_value, dtype=torch.float32).to(device)

       

        # Stage 1 Embedding
        with torch.no_grad():  # Stage 1은 학습하지 않음
            stage1_model = stage1_model.to(clip_a.device)
            emb1,emb2 = stage1_model(clip_a, clip_b)

        # 원핫인코딩 y 집합 벡터
        batch_size = target_value.size(0) # (batch_size는 train_loader에서 정의한 것과 같음)
        y_train = torch.zeros((batch_size, output_size), device=device)  # 배치 크기에 맞는 텐서 
        y_train.scatter_(1, target_value.unsqueeze(1), 1)  # target_value를 multi-hot 벡터로 변환
        
        if arg == 'sep' : 
            # Forward Pass
            loss = 0
            for emb in [emb1, emb2]:
                pred = model(emb)  # MLP로 예측
                part_loss = criterion(pred, y_train)  # 손실 계산
                loss += part_loss
                
            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
        elif arg == 'mean':
            emb = (emb1 + emb2) / 2
            pred = model(emb)
            loss = criterion(pred, y_train)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            
        elif arg == 'concat':
            emb = torch.cat((emb1, emb2), dim=1)
            pred = model(emb)
            loss = criterion(pred, y_train)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    
            
        else:
            raise ValueError("Invalid arg value. Choose from 'sep', 'mean', 'concat'.")

    total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)  # 평균 손실 계산
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Test (Validation) Loop
    val_loss = 0
    y_true = []
    y_pred = []
    y_pred1 = []
    y_pred2 = []

    for batch in val_loader:
        clip_a, clip_b, file_ids, target_value = batch
        clip_a, clip_b= clip_a.to(device), clip_b.to(device)
        # 쉼표와 공백으로 분리 후 숫자로 변환
        target_value = [
            float(num) for item in target_value for num in re.split(r'[,\s]+', item.strip()) if num
        ]
        target_value = torch.tensor(target_value, dtype=torch.float32).to(device)


        stage1_model.eval()

        # Stage 1 Embedding
        with torch.no_grad():  # Stage 1은 학습하지 않음
            stage1_model = stage1_model.to(clip_a.device)
            emb1,emb2 = stage1_model(clip_a, clip_b)

        # 원핫인코딩 y 집합 벡터
        batch_size = target_value.size(0) # (batch_size는 train_loader에서 정의한 것과 같음)
        y_val = torch.zeros((batch_size, output_size), device=device)  # 배치 크기에 맞는 텐서 
        y_val.scatter_(1, target_value.unsqueeze(1), 1)  # target_value를 multi-hot 벡터로 변환
        y_true.append(y_val.cpu().numpy())  # 리스트에 추가

        model.eval()
        with torch.no_grad():
            if arg == 'sep' : 
                # Forward Pass
                pred1 = model(emb1)
                y_pred1.append(pred1.cpu().numpy())  # 예측 결과 저장
                pred2 = model(emb2)
                y_pred2.append(pred2.cpu().numpy())  # 예측 결과 저장

                    
            elif arg == 'mean':
                emb = (emb1 + emb2) / 2
                pred = model(emb)
                y_pred.append(pred.cpu().numpy())  # 예측 결과 저장

                
                
            elif arg == 'concat':
                emb = torch.cat((emb1, emb2), dim=1)
                pred = model(emb)
                y_pred.append(pred.cpu().numpy())  # 예측 결과 저장
    
    # 리스트를 NumPy 배열로 변환하여 해밍 손실 계산
    y_true = np.concatenate(y_true, axis=0)  # 실제 값
    y_pred = np.concatenate(y_pred, axis=0)  # 예측 값

    # 'sep'의 경우, pred1과 pred2에 대해 각각 해밍 손실을 계산한 후 평균을 구함
    if arg == 'sep':
        y_pred1 = np.concatenate(y_pred1, axis=0)
        y_pred2 = np.concatenate(y_pred2, axis=0)
        score = (hamming_loss(y_true, y_pred1) + hamming_loss(y_true, y_pred2)) / 2

    else:
        score = hamming_loss(y_true, y_pred)  # 해밍 손실 계산

    print(f'Hamming Loss: {score:.4f}')
#################################


# %%
