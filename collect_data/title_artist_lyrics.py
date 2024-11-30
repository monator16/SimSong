#%% Descriptions
# <title/artist>
# file : unique_tracks.txt
# format : track id<SEP>song id<SEP>artist name<SEP>song title

# <lyrics>
# file : mxm_dataset_train.txt & mxm_dataset_test.txt
# format
# - comment, ignore
# %word1,word2,... - list of top words, in popularity order
# TID,MXMID,idx:cnt,idx:cnt,... - track ID from MSD, track ID from musiXmatch,
# then word index : word count (word index starts at 1!)

#%%
import csv

# 파일 경로 설정
unique_tracks_file = "./unique_tracks.txt"
mxm_train_file = "./mxm_dataset_train.txt"
mxm_test_file = "./mxm_dataset_test.txt"
output_file = "./matched_tracks.csv"

# 1. unique_tracks.txt 읽기
tracks_dict = {}
with open(unique_tracks_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("<SEP>")
        if len(parts) == 4:
            track_id, song_id, artist_name, song_title = parts
            tracks_dict[track_id] = {"title": song_title, "artist": artist_name}

# 2. mxm_dataset_train.txt와 mxm_dataset_test.txt 읽기
lyrics_dict = {}
for lyrics_file in [mxm_train_file, mxm_test_file]:
    with open(lyrics_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("%") or line.startswith("#"):
                continue  # 메타 데이터 또는 주석 무시
            parts = line.strip().split(",")
            if len(parts) > 2:
                track_id = parts[0]
                lyrics = ",".join(parts[2:])  # idx:cnt 포맷의 가사 데이터
                lyrics_dict[track_id] = lyrics

# 3. 매칭된 데이터 추출
matched_data = []
for track_id, track_info in tracks_dict.items():
    if track_id in lyrics_dict:
        matched_data.append([
            track_id,
            track_info["title"],
            track_info["artist"],
            lyrics_dict[track_id]
        ])

# 4. 결과를 파일로 저장
with open(output_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Track ID", "Title", "Artist", "Lyrics"])  # 헤더 추가
    writer.writerows(matched_data)

print(f"매칭된 {len(matched_data)}개의 데이터를 {output_file}에 저장했습니다.")

