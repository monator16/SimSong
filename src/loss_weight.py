'''Soft InfoNCE Loss에 사용되는 similarity score를 계산
'''
import torch
import torch.nn.functional as F
import pandas as pd

def load_lyrics(file_id):
    # CSV 파일 로드 (파일은 최종 파일로 함!!!!!!1)
    sim_set_dir = './data/filtered_lyrics_with_sets.csv'
    df = pd.read_csv(sim_set_dir)
    
    # file_id에 해당하는 행을 찾기
    row = df[df['Track ID'] == file_id]

    # 해당 file_id가 있는 경우 가사 반환
    if not row.empty:
        return row['Lyrics'].values[0]
    else:
        return "Lyrics not found for file_id: " + str(file_id)

def generate_lyrics_embeddings(file_ids, bert_model, tokenizer, device):
    """
    Args:
        file_ids: 가사 파일 ID 리스트.
        bert_model: BERT 기반 가사 임베딩 모델.
        tokenizer: BERT 토크나이저.
        device: 학습 장치 (예: 'cuda', 'cpu').

    Returns:
        lyrics_embeddings: 가사 임베딩 [batch_size, embed_dim].
    """
    lyrics_texts = [load_lyrics(file_id) for file_id in file_ids]  # file_id로 가사 텍스트 로드
    inputs = tokenizer(lyrics_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    lyrics_embeddings = outputs.last_hidden_state.mean(dim=1)  # [CLS] 토큰이 아닌 전체 평균 사용
    return lyrics_embeddings


def compute_similarity(lyrics_embeddings):
    """
    Args:
        lyrics_embeddings: 가사 임베딩 [batch_size, embed_dim].

    Returns:
        sim_ij: 가사 간 유사도 행렬 [batch_size, batch_size].
    """
    # 가사 임베딩들 간의 유사도 계산 (코사인 유사도)
    sim_ij = F.cosine_similarity(lyrics_embeddings.unsqueeze(1), lyrics_embeddings.unsqueeze(0), dim=-1)
    return sim_ij