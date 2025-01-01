#%%

# ------------
# AudioDataset
# ------------
import os
import torchaudio
import torch
from torch.utils.data import Dataset
import re
import numpy as np
from torch import Tensor
import pandas as pd
from torch.utils.data import DataLoader
from typing import Tuple, List
import random

class AudioDataset(Dataset):
    def __init__(self, file_paths):
        """
        Contrastive Learning용 오디오 데이터셋.
        Args:
            file_paths (list): 오디오 파일 경로 리스트.
            transform (callable, optional): 데이터 증강을 위한 함수.
        """
        self.file_paths = file_paths
       

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 오디오 파일 로드
        file_path = self.file_paths[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        
       

        return waveform, file_path #file_path가 track_id



def create_contrastive_datasets(dataset_dir):
    file_paths = [
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
    ]

    test_dataset=AudioDataset(file_paths)

    return test_dataset



class ContrastiveDataset(Dataset):
    def __init__(self, dataset: Dataset,  input_shape: List[int]):
        # dir : sim_set이 들어있는 최종 csv 파일 경로
        self.dataset = dataset

        self.input_shape = input_shape
        self.ignore_idx = []
    
    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
    
        sample_rate = 44100
        range = sample_rate * 15

        if idx in self.ignore_idx:
            return self[idx + 1]

        try:
            audio, file_path = self.dataset[idx]

            if audio is None or audio.shape[1] - (range * 2) < self.input_shape[1]:  # 오디오가 None이거나 길이가 짧으면 제외
                self.ignore_idx.append(idx)
                return self[idx + 1]

            # 전반부와 후반부로 나누고 각각 연속된 클립 선택
            mid_point = audio.shape[1] // 2
            first_half = audio[:, range:mid_point]
            second_half = audio[:, mid_point:audio.shape[1] - range]

            # 연속된 클립 추출
            clip_a = self._get_continuous_clip(first_half)
            clip_b = self._get_continuous_clip(second_half)

            # 파일 경로에서 파일 이름(ID) 추출
            file_name = os.path.basename(file_path)
            file_id = os.path.splitext(file_name)[0]

            # # file_id를 기준으로 CSV에서 target_column 값 가져오기
            # target_value = None
            # if file_id in self.metadata.iloc["Track ID"].values:  # 첫 번째 열에 file_id가 있는지 확인
            #     row = self.metadata[self.metadata.iloc[:, 0] == file_id]
            #     target_value = row[self.target_column].values[0]  # 특정 열 값 가져오기

            # # target_value가 None인 경우, 데이터 건너뛰기
            # if target_value is None:
            #     self.ignore_idx.append(idx)
            #     return self[idx + 1]

            #---------------------------
            #원핫벡터로 target value 설정
            
            

            # None 값 확인
            if clip_a is None or clip_b is None or file_id is None :
                self.ignore_idx.append(idx)
                print('넘어감')
                return self[idx + 1]

            # if target_value is None:
            #     self.ignore_idx.append(idx)
            #     return self[idx + 1]
            # 클립 A, 클립 B, 곡 ID, 타겟 값을 반환
            return clip_a, clip_b, file_id

        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            self.ignore_idx.append(idx)
            return self[idx + 1]



    def __len__(self) -> int:
        return len(self.dataset)

    def _get_continuous_clip(self, audio: Tensor) -> Tensor:
        """오디오에서 연속된 길이의 클립을 추출하는 메서드"""
        clip_length = self.input_shape[1]  # 원하는 클립 길이
        max_start_idx = audio.shape[1] - clip_length
        start_idx = random.randint(0, max_start_idx)  # 연속된 구간을 선택하기 위해 랜덤 인덱스 선택
        clip = audio[:, start_idx:start_idx + clip_length]
        
       
        
        return clip

dataset_dir = "/home/elicer/project/collect_data/mp3"
test_dataset = create_contrastive_datasets(dataset_dir)

sample_rate = 44100 # [1, sample_rate*30]: 30초로 구간 설정
test_contrastive_dataset = ContrastiveDataset(test_dataset,  input_shape=[1, sample_rate*30])

print(len(test_contrastive_dataset))
batch_size = 68 # 모든 노래 사용하기 위해서 batch_size = 68로 변경
test_loader = DataLoader(test_contrastive_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

import sys
import os

sys.path.append(os.path.abspath('/home/elicer/project/src'))

from models import ContrastiveModel
from ast_encoder import ASTEncoder
from loss import soft_info_nce_loss, info_nce_loss
from loss_weight import generate_lyrics_embeddings, compute_similarity

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
import re
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

# %% WNS_model_1203 불러와서 테스트

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


#optimal에서 Anchor Id만 추출하기
optimal_df = pd.read_csv('/home/elicer/project/data/optimal_optimal_data.csv') 
filtered_optimal_df = optimal_df[optimal_df['IsInAnchor'] == 1] 
track_ids = filtered_optimal_df['Track ID'].tolist() #anchor의 track_id list

anchor_df = df[df['id'].isin(track_ids)] #anchor의 df (track_id + embedding)
anchor_embeddings = np.stack(anchor_df['embedding'].values)  # anchor의 임베딩 배열 

ohters_df = df[~df['id'].isin(track_ids)]
others_embeddings = np.stack(ohters_df['embedding'].values)

all_embeddings = np.vstack([anchor_embeddings, others_embeddings])  # anchor와 다른 임베딩을 합침

normalized_embeddings = normalize(all_embeddings, norm='l2')

# 정규화된 앵커 임베딩과 다른 임베딩 분리하기
normalized_anchor_embeddings = normalized_embeddings[:len(anchor_embeddings)]  # 정규화된 앵커 임베딩
normalized_others_embeddings = normalized_embeddings[len(anchor_embeddings):]  # 정규화된 다른 임베딩

# Cosine Similarity 계산 및 Top 5 & 10 유사 음악 ID 추가
top5_sets = []
top10_sets = []

for i in range(len(anchor_embeddings)): #56개

    # 현재 음악과 다른 모든 음악과의 유사도를 계산
    normalized_embeddings = normalized_anchor_embeddings[i] # 현재 음악(anchor 중 한개)

    similarities = [
        cosine_similarity(normalized_embeddings.reshape(1, -1), normalized_others_embeddings[j].reshape(1, -1)).flatten()
        for j in range(len(normalized_others_embeddings))
    ] # 한 앵커에 대해서 다른 노래들과 similarity 계산

    similarities = [item[0] for item in similarities] # value만 추출
    similarities = np.array(similarities)
    sorted_indices = similarities.argsort()[::-1]  # 유사도를 내림차순 정렬

    top5_indices = [idx for idx in sorted_indices if idx != i][:5]  # 자기 자신 제외하고 상위 5개
    top5_ids = df.iloc[top5_indices]['id'].tolist()  # 해당 인덱스의 ID 가져오기
    top5_sets.append(top5_ids)

    top10_indices = [idx for idx in sorted_indices if idx != i][:10]  # 자기 자신 제외하고 상위 10개
    top10_ids = df.iloc[top10_indices]['id'].tolist()  # 해당 인덱스의 ID 가져오기
    top10_sets.append(top10_ids)

# SET 칼럼 추가
anchor_df['SET_5'] = top5_sets
anchor_df['SET_10'] = top10_sets
# %% 
