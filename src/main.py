import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------
# dataloaders
# ------------
from data import create_contrastive_datasets, ContrastiveDataset

import torch
from torch.utils.data import DataLoader
from torchaudio_augmentations import Compose

# 예시로 사용할 변환 함수
transform = Compose([]) 

# 데이터 경로 설정
dataset_dir = "./data/mp3"
sim_set_dir = './data/filtered_lyrics_with_sets.csv'

target_column = 'Set Index'

# 오디오 파일 경로 및 데이터셋 준비
train_dataset, val_dataset = create_contrastive_datasets(dataset_dir, train_ratio=0.8)

# ContrastiveDataset으로 변환
sample_rate = 44100 # [1, sample_rate*30]: 30초로 구간 설정
train_contrastive_dataset = ContrastiveDataset(train_dataset, sim_set_dir, target_column, input_shape=[1, sample_rate*30], transform=transform)
val_contrastive_dataset = ContrastiveDataset(val_dataset, sim_set_dir, target_column, input_shape=[1, sample_rate*30], transform=transform)

# DataLoader로 배치 생성
train_loader = DataLoader(train_contrastive_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_contrastive_dataset, batch_size=16, shuffle=False)
# -> 한 배치의 구성 : clip_a, clip_b, file_id, target_value


# ------------
# model
# ------------
from models import ContrastiveModel
from ast_encoder import ASTEncoder
from loss import soft_info_nce_loss
from loss_weight import generate_lyrics_embeddings, compute_similarity

# 1. 모델과 옵티마이저 초기화
######## 여기 만들어야 함!!!!!!!!!!!!
ast_encoder = ASTEncoder()
model = ContrastiveModel(ast_encoder)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 2. BERT 모델 로드 (가사 임베딩용)
from transformers import BertTokenizer, BertModel

bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 3. 학습 루프
num_epochs = #설정해야 함
batch_size = #설정해야 함
for epoch in range(num_epochs):
    for batch in train_loader:
        clip_a, clip_b, file_ids = batch  # 오디오와 file_ids 로드

        # 1) 오디오 임베딩 생성 (오디오는 로스 계산에만 사용)
        audio_embeddings = ast_encoder.preprocess(clip_a, clip_b)

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
            temperature=0.07,
            device=device
        )

        # 5) 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
