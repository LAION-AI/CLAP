"""
Contrastive Language-Audio Pretraining Model from LAION
--------------------------------------------------------
Paper: https://arxiv.org/abs/2211.06687
Authors (equal contributions): Ke Chen, Yusong Wu, Tianyu Zhang, Yuchen Hui
Support: LAION
"""

import librosa
import laion_clap

model = laion_clap.CLAP_Module(enable_fusion=True)
model.load_ckpt()

# Directly get audio embeddings from audio files
audio_file = [
    '/home/la/kechen/Research/KE_CLAP/ckpt/test_clap_short.wav',
    '/home/la/kechen/Research/KE_CLAP/ckpt/test_clap_long.wav'
]
audio_embed = model.get_audio_embedding_from_filelist(x = audio_file)
print(audio_embed)
print(audio_embed.shape)

# Get audio embeddings from audio data
audio_data, _ = librosa.load('/home/la/kechen/Research/KE_CLAP/ckpt/test_clap_short.wav', sr=48000) # sample rate should be 48000
audio_data = audio_data.reshape(1, -1) # Make it (1,T) or (N,T)

audio_embed = model.get_audio_embedding_from_data(x = audio_data)
print(audio_embed)
print(audio_embed.shape)

# Get text embedings from texts:
text_data = ["I love the contrastive learning", "I love the pretrain model"]
text_embed = model.get_text_embedding(text_data)
print(text_embed)
print(text_embed.shape)