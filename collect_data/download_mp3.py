#%%
import pandas as pd
import os
import yt_dlp
from yt_dlp import YoutubeDL
import time
import json 
   
    
#%%
# Define file paths
base_file = 'youtubeUrl_241128.csv'
sim = 'data\sim_except_HC.csv'
base_url = "https://www.youtube.com/watch?v="

# Read CSV files
df_base = pd.read_csv(base_file) 
df_lyrics = pd.read_csv(sim)


#%%
checkpoint_file = 'download_checkpoint.json'
# MP3 파일을 저장할 디렉토리
directory = 'mp3'

# 체크포인트 파일이 존재하면 읽고, 존재하지 않으면 0부터 시작
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        try:
            checkpoint = json.load(f)
        except json.JSONDecodeError:
            # JSONDecodeError 발생 시 기본값으로 초기화
            checkpoint = {'last_processed_index': 0}
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)
    start_index = checkpoint.get('last_processed_index', 0)
else:
    start_index = 0

# base 파일에서 각 행을 순차적으로 처리, 체크포인트 인덱스 이후부터 시작
for index, row in df_base.iloc[start_index:].iterrows():
    # title과 artist를 합쳐서 검색할 문자열 생성
    title_artist = f"{row['title']} {row['artist']}"  # 'a' 문자열 생성
    
    # filtered_lyrics_file에서 일치하는 trackid 찾기
    matching_row = df_lyrics[df_lyrics['Title'] + ' ' + df_lyrics['Artist'] == title_artist]
    
    if not matching_row.empty:
        # 일치하는 trackid 추출
        trackid = matching_row['Track ID'].values[0]
        
        # yt-dlp의 출력 템플릿과 옵션 설정
        ydl_opts = {
            "format": 'bestaudio/best',
            'outtmpl': os.path.join(directory, f'{trackid}.%(ext)s'),  # trackid를 파일명으로 저장
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '320'
            }],
            'noplaylist': True  # 재생목록을 다운로드하지 않도록 설정

        }

        # yt-dlp를 이용해 오디오 다운로드
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([base_url + row["video_id"]])

            # 다운로드가 성공적으로 완료되면 체크포인트 업데이트
            checkpoint = {'last_processed_index': index + 1}  # 다음 곡의 인덱스를 저장
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)

            print(f"{row['title']} by {row['artist']} 다운로드 완료")
        
        except Exception as e:
            print(f"{row['title']} by {row['artist']} 다운로드 중 오류 발생: {e}")
            # 오류 발생 시 계속해서 다음 곡을 처리하려면 continue 사용
            continue

    else:
        print(f"{title_artist}에 대한 일치 항목을 찾을 수 없음")

# 모든 다운로드가 완료되면 체크포인트 파일을 삭제할 수 있음 (선택 사항)
# os.remove(checkpoint_file)

# %%
