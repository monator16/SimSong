

# SimSong: Content-Based Music Recommendation System

## 📖 Project Overview
SimSong is a music recommendation system that employs contrastive learning techniques to generate music embeddings based on audio and lyrical content. The project aims to enhance the accuracy of music recommendations by considering both melody and lyrics in the recommendation process.

## 🚀 Features
- **Contrastive Learning**: Trains a model to create music embeddings that highlight contextual similarities among songs by comparing segments of the same song and contrasting them with others.
- **Lyrical Similarity**: Incorporates a BERT model to calculate lyrical similarities, aligning them with audio representations to improve recommendations.
  
## 📊 Performance Metrics
- **Recall@10**: 0.02
- **Precision@10**: 0.0214

These metrics are used to evaluate the model's effectiveness in retrieving relevant recommendations based on user preferences.

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/simsong.git
cd simsong
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Model
```bash
python main.py
```

## 📚 Dataset
This project utilizes the [Million Song Dataset (MSD)](https://millionsongdataset.com/) which consists of metadata and audio features for over one million contemporary popular music tracks. Additionally, we used the last.fm dataset for similarity scores and scraped raw audio and lyrics from YouTube and Genius.

## 🧪 Experiments
The study was divided into two main models:
1. **Weighted Negative Sampling (WNS)**: Uses soft InfoNCE loss to calculate a weighted negative loss.
2. **Standard Negative Sampling (NS)**: Employs pure negative loss using InfoNCE loss.

### Data Preprocessing
- **Sampling Rate**: Audio samples were processed at 16kHz.
- **Segmentation**: Randomly selected segments to enhance context in recommendations.

## 🚧 Limitations
- The model's performance may depend heavily on the quality of the datasets used. 
- Future improvements could include better alignment of genres and incorporating more comprehensive datasets.

## 👨‍💻 Authors
- Youngseo Lee
- Hyunji Lee
