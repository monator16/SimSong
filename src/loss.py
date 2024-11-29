'''InfoNCE Loss와 Soft InfoNCE Loss 정의한 코드'''

# ------------
# info_nce_loss : simCLR 코드 참고함
# ------------
import torch
import torch.nn.functional as F

def info_nce_loss(features, batch_size, n_views, temperature, device):
    '''
    Args:
        features(z_ij): 모델의 출력 특징 벡터 (배치 크기 * 뷰 수)
        batch_size: 배치 크기
        n_views: 각 샘플에 대해 생성된 뷰의 수
        temperature: 온도 파라미터 (temperature scaling)
        device: 디바이스 (예: 'cuda', 'cpu')
    Returns:
        loss: InfoNCE 손실 값
    '''
    # 레이블 생성 (각 샘플의 모든 뷰는 동일한 레이블을 가짐)
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # 같은 샘플끼리는 1, 다른 샘플은 0
    labels = labels.to(device)

    # 특성 벡터 정규화 & 유사도 행렬 계산 (코사인 유사도)
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    # 주대각선 항목(자기 자신과의 유사도)을 마스크 처리
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)  # 대각선 항목 제거
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)  # 대각선 항목 제거

    # 양성 샘플(positive samples)과 음성 샘플(negative samples) 선택
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # 동일한 샘플들 간의 유사도
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # 다른 샘플들 간의 유사도

    # Temperature scaling
    logits = torch.cat([positives, negatives], dim=1)  # 양성 및 음성 샘플 합치기
    logits = logits / temperature

    # 손실 계산
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)  # 첫 번째 항목(positive)을 정답으로 설정
    loss = F.cross_entropy(logits, labels)

    return loss


# ------------
# soft_info_nce_loss
# ------------

import torch
import torch.nn.functional as F

def soft_info_nce_loss(features, sim_ij, batch_size, n_views, temperature, device, alpha=0.1, beta=1.0):
    """
    Soft-InfoNCE Loss 정의:
    유사도 행렬 기반으로 양성 샘플과 음성 샘플 간의 가중치를 계산하고 손실을 생성.

    Args:
        features (torch.Tensor): 모델 출력 특징 벡터 (크기: batch_size * n_views x feature_dim).
        sim_ij (torch.Tensor): 샘플 간 유사도 행렬 (크기: batch_size * n_views x batch_size * n_views).
        batch_size (int): 배치 크기.
        n_views (int): 각 샘플에 대해 생성된 뷰의 수.
        temperature (float): 온도 파라미터 (temperature scaling).
        device (torch.device): 학습 디바이스 (예: 'cuda', 'cpu').
        alpha (float): 음성 샘플의 가중치 조절 파라미터.
        beta (float): 음성 샘플 가중치 계산에 사용되는 상수.

    Returns:
        loss (torch.Tensor): Soft-InfoNCE 손실 값 (스칼라).
    """
    # 레이블 생성: 동일 샘플의 모든 뷰는 동일한 레이블
    labels = torch.cat([torch.arange(batch_size) for _ in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)

    # 특성 정규화
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    # 대각선(자기 자신과의 유사도)을 마스크 처리
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)  # 대각선 제거
    sim_ij = sim_ij[~mask].view(sim_ij.shape[0], -1)  # 대각선 제거

    # 양성 샘플과 음성 샘플 분리
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # 동일 샘플 간 유사도
    negatives = similarity_matrix[~labels.bool()].view(sim_ij.shape[0], -1)  # 다른 샘플 간 유사도

    # 음성 샘플에 대한 가중치(w_ij) 계산
    neg_sim_sum = torch.sum(sim_ij, dim=1, keepdim=True) - positives.sum(dim=1, keepdim=True)  # 양성 제외한 합
    wij = (beta - alpha * negatives) / (beta - alpha * neg_sim_sum / (batch_size * n_views - 1))

    # 음성 샘플 가중치 적용 및 스케일링
    negatives = torch.exp(negatives / temperature) * wij

    # 양성 샘플은 exp로 변환 후 온도 적용
    positives = torch.exp(positives / temperature)

    # 분자: 양성 샘플 (numerator)
    numerator = positives.sum(dim=1)  # 양성 샘플 합

    # 분모: 양성 + 음성 샘플 (denominator)
    denominator = numerator + negatives.sum(dim=1)

    # 손실 계산
    loss = -torch.log(numerator / denominator).mean()

    return loss
