import torch
import torch.nn as nn
import numpy as np

class ContrastiveModel(nn.Module):
    def __init__(self, encoder):#dim 일단 랜덤
        super().__init__()
        self.audio_encoder = encoder
        self.projection_head = nn.Sequential(
            nn.Linear(932352, 256), #Ast 출력 차원에 맞게 조정
            nn.ReLU(),
            nn.Linear(256, 128) #최종 projection 차원

        )

    
    def forward(self,clip_a, clip_b ):
        device = clip_a.device  # clip_a의 디바이스를 확인하여 일관되게 사용

        if self.training:
            self.audio_encoder.set_train_mode()
        else:
            self.audio_encoder.set_eval_mode()

        clip_a_embeddings, clip_b_embeddings = self.audio_encoder.preprocess(clip_a, clip_b)
        
         # NumPy 배열을 PyTorch Tensor로 변환 (필요할 경우)
        if isinstance(clip_a_embeddings, np.ndarray):
            clip_a_embeddings = torch.from_numpy(clip_a_embeddings).float()  # NumPy -> Tensor 변환
            clip_a_embeddings.to(device)
        if isinstance(clip_b_embeddings, np.ndarray):
            clip_b_embeddings = torch.from_numpy(clip_b_embeddings).float()  # NumPy -> Tensor 변환
            clip_b_embeddings.to(device)

        # Projection
        projected_a = self.projection_head(clip_a_embeddings)
        projected_b = self.projection_head(clip_b_embeddings)

        #projection 합치기

        

        return projected_a, projected_b
    