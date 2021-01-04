import random
import librosa

def cut_audio(audio, seq_len):
    if audio.size >= seq_len:
        max_audio_start = audio.size - seq_len
        audio_start = random.randint(0, max_audio_start)
        audio = audio[audio_start : audio_start + seq_len]
    return audio

def rescale_audio(audio):
    mini, maxi = min(audio), max(audio)
    return ((audio - mini) / (maxi - mini) - 0.5) * 2

def read_wav_file(file_path, sample_rate=22050, seq_len=8192 * 12):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    audio = cut_audio(audio, seq_len)
    audio = librosa.util.normalize(audio)
    return audio, sr