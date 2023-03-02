import torch
import librosa
from clap_module import create_model
from training.data import get_audio_features
from training.data import int16_to_float32, float32_to_int16
from transformers import RobertaTokenizer

tokenize = RobertaTokenizer.from_pretrained('roberta-base')
def tokenizer(text):
    result = tokenize(
        text,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    return {k: v.squeeze(0) for k, v in result.items()}

def infer_text():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    precision = 'fp32'
    amodel = 'HTSAT-tiny' # or 'PANN-14'
    tmodel = 'roberta' # the best text encoder in our training
    enable_fusion = True # False if you do not want to use the fusion model
    fusion_type = 'aff_2d'
    pretrained = "/home/la/kechen/Research/KE_CLAP/ckpt/fusion_best.pt" # the checkpoint name, the unfusion model can also be loaded.

    model, model_cfg = create_model(
        amodel,
        tmodel,
        pretrained,
        precision=precision,
        device=device,
        enable_fusion=enable_fusion,
        fusion_type=fusion_type
    )
    # load the text, can be a list (i.e. batch size)
    text_data = ["I love the contrastive learning", "I love the pretrain model"] 
    # tokenize for roberta, if you want to tokenize for another text encoder, please refer to data.py#L43-90 
    text_data = tokenizer(text_data)
    model.eval()
    text_embed = model.get_text_embedding(text_data)
    text_embed = text_embed.detach().cpu().numpy()
    print(text_embed)
    print(text_embed.shape)

def infer_audio():
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    precision = 'fp32'
    amodel = 'HTSAT-tiny' # or 'PANN-14'
    tmodel = 'roberta' # the best text encoder in our training
    enable_fusion = True # False if you do not want to use the fusion model
    fusion_type = 'aff_2d'
    pretrained = "/home/la/kechen/Research/KE_CLAP/ckpt/fusion_best.pt" # the checkpoint name, the unfusion model can also be loaded.

    model, model_cfg = create_model(
        amodel,
        tmodel,
        pretrained,
        precision=precision,
        device=device,
        enable_fusion=enable_fusion,
        fusion_type=fusion_type
    )

    # load the waveform of the shape (T,), should resample to 48000
    audio_waveform, sr = librosa.load('/home/la/kechen/Research/KE_CLAP/ckpt/test_clap_short.wav', sr=48000) 
    # quantize
    audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
    audio_waveform = torch.from_numpy(audio_waveform).float()
    audio_dict = {}

    # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
    audio_dict = get_audio_features(
        audio_dict, audio_waveform, 480000, 
        data_truncating='fusion', 
        data_filling='repeatpad',
        audio_cfg=model_cfg['audio_cfg']
    )
    model.eval()
    # can send a list to the model, to process many audio tracks in one time (i.e. batch size)
    audio_embed = model.get_audio_embedding([audio_dict])
    audio_embed = audio_embed.detach().cpu().numpy()
    print(audio_embed)
    print(audio_embed.shape)
    


if __name__ == "__main__":
    infer_text()
    # infer_audio()
