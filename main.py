from parallel_wavegan.utils import load_model
from espnet2.bin.tts_inference import Text2Speech
import soundfile as sf
from utils import normalization
import torch
import numpy as np
import sys

def load_parallel_wavegan_model(model_path):
    """Loads Parallel WaveGAN model for sound conversion"""
    try:
        model = load_model(model_path).to("cpu").eval()
        model.remove_weight_norm()
        return model
    except ImportError as error:
        print(f"Failed to load voice model: {error}", file=sys.stderr)
        sys.exit(1)

def create_espnet_model(config_file, model_path, device="cpu"):
    """Creates ESPNet model for text-to-speech conversion"""
    return Text2Speech(
        config_file,
        model_path,
        device=device,
        threshold=0.3,
        minlenratio=0.8,
        maxlenratio=8.0,
        use_att_constraint=True,
        backward_window=1,
        forward_window=3,
        speed_control_alpha=1.0,
    )

def convert_text_to_audio(text, language, speech_model, voice_model):
    """Converts text to audio data"""
    normalized_text = normalization(text, language)
    with torch.no_grad():
        melody = speech_model(normalized_text)['feat_gen']
        audio = voice_model.inference(melody)
        if isinstance(audio, torch.Tensor):
            audio = audio.view(-1).cpu().numpy()
        elif not isinstance(audio, np.ndarray):
            print("Unexpected audio data type:", type(audio), file=sys.stderr)
            sys.exit(1)
    return audio

def save_audio_file(audio_data, sample_rate, file_path):
    """Saves audio data to WAV file"""
    sf.write(file_path, audio_data, sample_rate)

def main():
    # Settings
    sample_rate = 22050
    voice_model_path = "parallelwavegan_male2_checkpoint/checkpoint-400000steps.pkl"
    config_file = "exp/tts_train_raw_char/config.yaml"
    model_path = "exp/tts_train_raw_char/train.loss.ave_5best.pth"
    
    # Text to vocalize
    text = (
        "Сәлем! Мен Айгүлмін—болашақтан келген жасанды интеллект. Қазақстандық ғалымдардың туындысы бола отырып, "
        "менің миссиям—заманауи технологиялар, жасанды интеллект және біздің планетамыздың болашағы туралы "
        "біліммен бөлісу."
    )
    language = "kazakh"

    # Loading models
    voice_model = load_parallel_wavegan_model(voice_model_path)
    speech_model = create_espnet_model(config_file, model_path)
    speech_model.spc2wav = None  # disable spectrogram to wave conversion

    # Converting text to audio
    audio = convert_text_to_audio(text, language, speech_model, voice_model)
    
    # Saving result
    save_audio_file(audio, sample_rate, "result.wav")

if __name__ == "__main__":
    main()
