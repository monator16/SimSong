import torch
import torch.nn as nn

class CotrastiveModel(nn.Module):
    def __init(self, encoder, projection_dim =128):#dim 일단 랜덤
        super().__init__()
        self.audio_encoder = encoder
        self.proection_head = nn.Sequential(
            nn.Linear(self.audio_encoder.output_dim, 256), #Ast 출력 차원에 맞게 조정
            nn.ReLU(),
            nn.Linear(256,projection_dim) #최종 projection 차원

        )
    
    def forward(self,clip_a, clip_b, file_id ):
        clip_a_embeddings, clip_b_embeddings = self.audio_encoder.preprocess(clip_a, clip_b)
        
        # Projection
        projected_a = self.projection_head(clip_a_embeddings)
        projected_b = self.projection_head(clip_b_embeddings)

        #projection 합치기

        #file_id는 가사 데이터를 불러오는 데 활용
        lyrics_a = self.load_lyrics(file_id)

        return projected_a, projected_b, lyrics_a
    
    