import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import librosa
import soundfile as sf

class DataAugmentation:
    def __init__(
            self,
            audio_root="dataset/wavs/",
            sr=16000,
            min_intensity=0.2,
            max_intensity=1.0,
            max_silence_duration = 1,
            max_total_silence_intervals = 1
        ):
        
        self.sr = sr
        # Background noise parameters
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.max_silence_duration = max_silence_duration
        self.max_total_silence_intervals = max_total_silence_intervals
        self.audio_paths = []
        self.audio_root = audio_root

        # Load noise audio paths
        self.audio_paths.extend(self.searchAudioFiles(self.audio_root))
        if self.audio_paths:  # If paths were found
            print(f"Found {len(self.audio_paths)} audio files.")
        else:
            print("No audio files were found.")

    def searchAudioFiles(self, audio_root):
        # Search and load noise audio paths recursively
        path_list = list(Path(audio_root).rglob("*.wav"))  # Use rglob for recursive globbing
        return path_list

    def define_intervals(self, audio, total_intervals):           
        legth_interval = int(len(audio) / total_intervals)
        intervals = []
        for i in range(total_intervals):
            start = i * legth_interval
            end = start + legth_interval
            intervals.append((start, end))
        return intervals
    
    def generate_random_intervals(self, y):
        """
        Generate two random intervals within the duration of an audio file.

        Parameters:
        y (ndarray): Audio waveform.

        Returns:
        list of tuples: List of two random intervals (start, end).
        """
        total_intervals = random.randint(1, self.max_total_silence_intervals)
        intervals = self.define_intervals(y, total_intervals)

        # Generate random intervals
        silence_intervals_list = []
        for start, end in intervals:
            max_len = min( int(self.sr * self.max_silence_duration), int(end - start) )
            len = random.randint(0, max_len)
            s = random.randint(start, end-len)                      
            silence_intervals_list.append((s, s+len))
        return silence_intervals_list


    def silence_intervals(self, y):
        """
        Silence specific intervals in the audio waveform.

        Parameters:
        y (ndarray): Audio waveform.

        Returns:
        ndarray: Audio waveform with silenced intervals.
        """
        silent_intervals_list = self.generate_random_intervals(y)
        # Silence the specified intervals
        for start, end in silent_intervals_list:
            y[start:end] = 0
        return y

    def add_noise(self, audio, noise, noise_level=0.1):
        """
        Apply noise to an audio signal.

        Parameters:
        audio (np.array): The original audio signal.
        noise (np.array): The audio noise signal.
        noise_level (float): The intensity of the noise to be added. This is a factor which determines the amplitude of the noise in relation to the maximum         of the audio signal.

        Returns:
        ndarray: The audio signal with noise added.
        """
      
        # normalize audio noise
        noise = noise / np.max(np.abs(noise))
        
        # Add noise to audio with specified noise level
        noisy_audio = audio + noise_level * noise * np.max(np.abs(audio))
        
        return noisy_audio

    def applyBackgroundNoise(self, waveform):
        """
        Apply background noise to the audio waveform.

        Parameters:
        waveform (ndarray): Audio waveform.

        Returns:
        ndarray: Audio waveform with added background noise.
        """
        if len(self.audio_paths) == 0:
            print("Warning: No noise audio paths found. Please check the configuration and path loading.")
            return False

        noise_path = random.choice(self.audio_paths)
        noise, _ = librosa.load(noise_path, sr=self.sr)

        length_audio = len(waveform)
        length_noise = len(noise)

        if length_noise > length_audio:
            start_idx = random.randint(0, length_noise - length_audio)
            end_idx = start_idx + length_audio
            noise = noise[start_idx:end_idx]
        else:
            p = random.uniform(0, 1)
            if p > 0.5:
                # Repeat noise to match the length of the audio
                repeat_count = length_audio // length_noise + 1
                noise = np.tile(noise, repeat_count)[:length_audio]
            else:
                # Pad noise with zeros to match the length of the audio
                pad_length = length_audio - length_noise
                noise = np.pad(noise, (0, pad_length), 'constant')                

        noise = self.silence_intervals(noise)
        noise_level = random.uniform(self.min_intensity, self.max_intensity)
        aug_waveform = self.add_noise(waveform, noise, noise_level)
        return aug_waveform


    def augment(self, waveform):
        """
        Augment the audio waveform with background noise.

        Parameters:
        waveform (ndarray): Input audio waveform.

        Returns:
        ndarray: Augmented audio waveform.
        """
        wav_length = len(waveform)
        waveform = self.applyBackgroundNoise(waveform)
        waveform = waveform[:wav_length]
        return waveform.squeeze()


if __name__ == "__main__":
    from glob import glob
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  '-i', default="dataset/wavs")
    parser.add_argument("--output_dir", '-o', default="dataset/overlapping_wavs")
    parser.add_argument("--min_intensity", default=0.2)
    parser.add_argument("--max_intensity", default=1.0)
    parser.add_argument("--max_silence_duration", type=float, default=5.0)
    parser.add_argument("--max_total_silence_intervals", type=int, default=3)
    parser.add_argument("--sr", default=16000)

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    augmenter = DataAugmentation(
        audio_root=args.input_dir, 
        sr=args.sr,
        min_intensity=args.min_intensity,
        max_intensity=args.max_intensity,
        max_silence_duration=args.max_silence_duration,
        max_total_silence_intervals=args.max_total_silence_intervals
    )

    for audio_filepath in tqdm(glob(os.path.join(args.input_dir, "*.wav"))):
        audio, sr = librosa.load(audio_filepath, sr=args.sr)
        aug_audio = augmenter.augment(audio)
        filename = os.path.basename(audio_filepath)
        out_filepath = os.path.join(args.output_dir, filename)
        sf.write(out_filepath, aug_audio, args.sr, format='wav', subtype='PCM_16')
