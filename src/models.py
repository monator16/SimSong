import torch
import torch.nn as nn

class CotrastiveModel(nn.Module):
    def __init(self, encoder, projection_dim =128):#dim 일단 랜덤
        super().__init__()
        self.audio_encoder = encoder
        self.proection_head = nn.Sequential(
            nn.Linear(), #Ast 출력 차원에 맞게 조정
            nn.ReLU(),
            nn.Linear(,projection_dim) #최종 projection 차원

        )
    
    def forward(self,clip_a, clip_b, file_id ):
        embedding_a = self.audio_encoder(clip_a)
        embedding_b = self.audio_encoder(clip_b)
        
        # Projection
        projected_a = self.projection_head(embedding_a)
        projected_b = self.projection_head(embedding_b)

        #file_id는 가사 데이터를 불러오는 데 활용
        lyrics_a = self.load_lyrics(file_id)

        return projected_a, projected_b, lyrics_a
    
    def 