'''
train 루프 관련 코드
'''

# ------------
# AudioLyricsModel
# : Pytorch lightning의 trainer 사용해서 다시 코드 작성함함
# ------------

import pytorch_lightning as pl
from torch.utils.data import DataLoader

class AudioLyricsModel(pl.LightningModule):
    def __init__(self, model, lyrics, bert_model, tokenizer, batch_size):
        super().__init__()
        self.model = model
        self.lyrics = lyrics # 가사 사용 유무 : True/False
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def training_step(self, batch, batch_idx):
        clip_a, clip_b, file_ids = batch
        clip_a, clip_b = clip_a.to(self.device), clip_b.to(self.device)

        # 1) 오디오 임베딩 생성 (오디오는 로스 계산에만 사용)
        projected_a, projected_b = self.model(clip_a, clip_b, self.device)
        audio_embeddings = torch.cat([projected_a, projected_b], dim=0)

        # 2) 가사 임베딩 생성과 유사도 계산산
        lyrics_embeddings = generate_lyrics_embeddings(file_ids, self.bert_model, self.tokenizer, self.device)
        sim_ij = compute_similarity(lyrics_embeddings.repeat(2, 1))        

        # 3) 손실 계산
        if lyrics == True:
            loss = soft_info_nce_loss(
                features=audio_embeddings,
                sim_ij=sim_ij,
                batch_size=self.batch_size,
                n_views=2,
                temperature=0.5,
                device=self.device
            )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

# DataLoader 설정
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 모델 및 Trainer 설정
model = AudioLyricsModel(model, bert_model, tokenizer, batch_size=32)
trainer = pl.Trainer(max_epochs=10, log_every_n_steps=10)

# 학습 시작
trainer.fit(model, train_loader)





import torch
import os

from loss_weight import generate_lyrics_embeddings, compute_similarity
from loss import soft_info_nce_loss, info_nce_loss

def train_weighted_negative_sampling(train_loader, model, optimizer, bert_model, tokenizer, device, num_epochs, batch_size, checkpoint_path='WNS_checkpoint.pth'):
    start_epoch = 0

    # 체크포인트가 존재하면 불러오기
    if checkpoint_path:
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")
        else : 
            print(f"Initially start")

    model.train()
    for epoch in range(start_epoch, num_epochs): 
        for batch in train_loader:
            clip_a, clip_b, file_ids = batch  # 오디오와 file_ids 로드
            clip_a, clip_b = clip_a.to(device), clip_b.to(device)

            # 1) 오디오 임베딩 생성 (오디오는 로스 계산에만 사용)
            projected_a, projected_b = model(clip_a, clip_b, device)
            audio_embeddings = torch.cat([projected_a, projected_b], dim=0)  # Combine along the batch dimension

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

            # 5) 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # # 에포크가 끝날 때마다 체크포인트 저장
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': loss.item()
        # }, checkpoint_path)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # 최종 모델 저장
    final_model_path = 'WNS_model_1214.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")

        

def train_pure_negative_sampling(train_loader,model,optimizer, bert_model,tokenizer, device, num_epochs,batch_size):
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            clip_a, clip_b, file_ids = batch  # 오디오와 file_ids 로드
            clip_a, clip_b = clip_a.to(device), clip_b.to(device)

            # 1) 오디오 임베딩 생성 (오디오는 로스 계산에만 사용)
            projected_a, projected_b = model(clip_a, clip_b, device)

            audio_embeddings = torch.cat([projected_a, projected_b], dim=0)  # Combine along the batch dimension
            # print(audio_embeddings.shape)

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
            
            # print('one batch is completed')

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    #모델 저장
    torch.save(model, 'NS_model_1214.pth')