'''Contrast learning을 위해서 오디오 pair를 만듦
-> 오디오의 전반과 후반 각각에서 원하는 길이의 연속된 오디오 구간 추출'''


# ------------
# AudioDataset
# ------------
import os
import torchaudio
from torch.utils.data import Dataset

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




# ------------
# create_datsets
# ------------
def create_datsets(dataset_dir, state):
    '''
    train/test data 로드
    Args:
        dataset_dir (str): 오디오 파일이 포함된 디렉토리.
        state (str): 'train' or 'test'
    '''
    path = os.path.join(dataset_dir, state)
    data_files = os.listdir(path)
    
    dataset = AudioDataset(data_files)
    
    return dataset


# ------------
# create_contrastive_datasets
# ------------
from sklearn.model_selection import train_test_split

def create_contrastive_datasets(dataset_dir, train_ratio=0.8, random_seed=42):
    """
    오디오 파일을 train/val로 나누고, 데이터셋 생성.
    Args:
        dataset_dir (str): 오디오 파일이 포함된 디렉토리.
        train_ratio (float): train 데이터의 비율.
        random_seed (int): 랜덤 시드 값.
        transform (callable, optional): 데이터 증강 함수.
    Returns:
        train_dataset (AudioDataset): 학습 데이터셋.
        val_dataset (AudioDataset): 검증 데이터셋.
    """
    # 파일 경로 읽기
    file_paths = [
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
    ]

    # Train/Validation Split
    train_files, val_files = train_test_split(
        file_paths, train_size=train_ratio, random_state=random_seed
    )

    # Dataset 생성
    train_dataset = AudioDataset(train_files)
    val_dataset = AudioDataset(val_files)

    return train_dataset, val_dataset


# ------------
# ContrastiveDataset
# ------------
import torch
from torch import Tensor
from torch.utils.data import Dataset
#from torchaudio_augmentations import Compose
from typing import Tuple, List
import random
import pandas as pd
import os

class ContrastiveDataset(Dataset):
    def __init__(self, dataset: Dataset, model_parameters):
        # dir : sim_set이 들어있는 최종 csv 파일 경로
        self.dataset = dataset
        # self.transform = transform
        time = model_parameters['time']
        sample_rate = model_parameters['sampe_rate']
        self.input_shape = [1, sample_rate * time]
        self.ignore_idx = []
        # self.metadata = pd.read_csv(dir)  # CSV 파일 읽기
        # self.target_column = target_column

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

            # file_id를 기준으로 CSV에서 target_column 값 가져오기
            # target_value = None
            # if file_id in self.metadata.iloc[:, 0].values:  # 첫 번째 열에 file_id가 있는지 확인
            #     row = self.metadata[self.metadata.iloc[:, 0] == file_id]
            #     target_value = row[self.target_column].values[0]  # 특정 열 값 가져오기

            # # target_value가 None인 경우, 데이터 건너뛰기
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