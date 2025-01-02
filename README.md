
# SimSong: Content-Based Similar Music Recommendation Using Contrastive Learning  

SimSong is a research project that develops a music recommendation system based on audio content and lyrics. By leveraging state-of-the-art transformer-based models and contrastive learning techniques, this system generates embeddings to enhance the user experience with accurate music recommendations.  

---

## 📝 **Project Overview**  

### **Background**  
- Music, as an art form, is complex, involving elements like melody, ambiance, tempo, and lyrics. Capturing its essence with simple algorithms is challenging.  
- Existing studies often focused on either audio content or lyrics, without effectively integrating both.  
- This project bridges the gap by combining an Audio Spectrogram Transformer (AST) and contrastive learning to improve music recommendation systems.  

### **Objective**  
- Bring embeddings of different segments of the same song closer while adjusting the loss function based on lyrical similarity to capture contextual similarities in music.  
- Enhance recommendation quality by considering the synergy of melody and lyrics.  

---

## 🛠 **Technologies and Methodology**  

### **Model Architecture**  
1. **Audio Processing**  
   - Creates 30-second audio segments, processed through AST to generate spectrograms.  
   - AST is a transformer-based model capable of capturing long-range dependencies in audio data.  
2. **Contrastive Learning**  
   - Utilizes InfoNCE and Soft InfoNCE loss functions for embedding learning.  
   - Soft InfoNCE incorporates lyric-based weighting for negative samples to refine the learning process.  
3. **Lyric Processing**  
   - Lyrics embeddings are computed using a BERT model to assess semantic similarity.  

### **Dataset**  
- **Million Song Dataset (MSD)**: Contains metadata and audio features for over a million songs.  
- **Last.fm Dataset**: Provides song similarity scores.  
- **YouTube & Genius**: Used for raw audio and lyrics.  

### **Evaluation Metrics**  
- **Cosine Similarity**: Measures alignment between audio embeddings.  
- **Recall@k**: Evaluates the ability of the system to meet user preferences through top-k recommendations.  

---

## 📊 **Key Results**  

- The model using Soft InfoNCE loss (WNS model) outperformed the one using InfoNCE loss (NS model).  
- However, the overall performance (Recall@10 = 0.02) was limited due to dataset size and preprocessing constraints.  

---

## 📌 **Challenges and Improvements**  

### **Identified Issues**  
1. **Dataset Constraints**  
   - Training involved only 400 songs, lacking diversity and scale.  
2. **Preprocessing Impact**  
   - Removing parts of audio (e.g., intros/outros) may have excluded critical musical elements.  

### **Future Directions**  
1. **Embedding Validation**  
   - Test embeddings through tasks like genre classification or clustering to ensure meaningful representations.  
2. **Data Expansion and Refinement**  
   - Include more diverse music data and integrate artist and genre information for a comprehensive similarity metric.  

---

## 🚀 **Usage Instructions**  

### **Setup**  
-- need to update

---
