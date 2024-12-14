# %% md
# # 1. Import Libraries
# %%
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
from IPython.display import Audio
import torch
from demucs import pretrained
from demucs.apply import apply_model
import noisereduce as nr  # Import noisereduce library


# %% md
# # 2. Noise Reduction Function
# This function processes multiple noise samples by averaging their spectrograms and applying spectral subtraction to the noisy audio.
# %%
def noise_reduction(noisy_audio, noise_audios, sr, n_fft=2048, hop_length=512, win_length=2048):
    # Compute STFT for noisy audio
    D_noisy = librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    # Compute and average STFT for multiple noise samples
    noise_mags = []
    for noise_audio in noise_audios:
        D_noise = librosa.stft(noise_audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        noise_mags.append(np.abs(D_noise))
    noise_mag_mean = np.mean(np.stack(noise_mags, axis=2), axis=2)  # Average along the noise sample axis

    # Perform spectral subtraction
    noisy_mag = np.abs(D_noisy)
    phase = np.angle(D_noisy)
    mag_clean = noisy_mag - noise_mag_mean
    mag_clean = np.maximum(mag_clean, 0)  # Ensure no negative values
    D_clean = mag_clean * np.exp(1j * phase)

    # Inverse STFT to reconstruct the clean audio
    clean_audio = librosa.istft(D_clean, hop_length=hop_length, win_length=win_length)
    return clean_audio


# %% md
# # 3. Helper Functions for Visualization
# ### 3.1 Plot Waveform
# %%
def plot_waveform(audio, sr, title):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title(f"Waveform of {title}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()


# %% md
# ### 3.2 Plot Spectrogram
# %%
def plot_spectrogram(audio, sr, title):
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogram of {title}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.show()


# %% md
# ### 3.3 Plot Frequency Domain
# %%
def plot_frequency_domain(audio, sr, title):
    plt.figure(figsize=(10, 4))
    Y = np.fft.fft(audio)
    freq = np.fft.fftfreq(len(audio), 1 / sr)
    plt.plot(freq[:len(freq) // 2], np.abs(Y)[:len(Y) // 2])
    plt.title(f"Frequency Domain of {title}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()


# %% md
# # 4. Load Audio Files
# Here we load the noisy audio, clean audio, and multiple noise samples.
# %%
# File paths
noisy_audio_path = 'audio_noise.wav'
noise_audio_paths = ['noise/noise_white.wav', 'noise/noise_thermal.wav', 'noise/noise_air-conditioner.wav']
audio_audio_path = 'audio.wav'

# Load audio files
noisy_audio, sr = librosa.load(noisy_audio_path, sr=None)
audio_audio, _ = librosa.load(audio_audio_path, sr=sr)
noise_audios = [librosa.load(path, sr=sr)[0] for path in noise_audio_paths]

# %% md
# # 5. Perform Noise Reduction
# Call the noise_reduction function to process the noisy audio with multiple noise samples.
# %%
# Perform noise reduction
clean_audio = noise_reduction(noisy_audio, noise_audios, sr)

# Save the result
output_path = 'cleaned_audio.wav'
sf.write(output_path, clean_audio, sr)


# %% md
# # 6. Load and Apply Demucs Model
# We'll load the Demucs model and apply it to the already cleaned audio.
# %%
def load_demucs_model(model_name='htdemucs'):
    # Load the pretrained model from pretrained.py
    model = pretrained.get_model(model_name)  # This function is directly from pretrained.py
    return model


def apply_demucs_to_audio(audio, model):
    try:
        # Convert the cleaned audio into a torch tensor (Add batch and channel dimensions)
        audio_tensor = torch.tensor(audio).unsqueeze(0).unsqueeze(0)  # Shape: (batch, channel, time)

        # Move to the appropriate device (GPU if available, otherwise CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        audio_tensor = audio_tensor.to(device)

        # Apply Demucs using `apply_model` instead of directly calling the model
        with torch.no_grad():
            # Apply the model and get separated sources
            separated_sources = apply_model(model, audio_tensor)

        # The output will contain a batch of sources (like vocals, drums, etc.), we assume we're interested in the first source.
        refined_audio = separated_sources[0][0].cpu().numpy()  # [0][0] to get the first channel of the first source

        return refined_audio

    except Exception as e:
        # Print the error message without halting the program
        print(f"Error while applying Demucs: {e}")
        # Return the input audio as fallback
        return audio  # You can also decide to return a default value or log this error instead


# Load Demucs model and apply it to the cleaned audio
try:
    demucs_model = load_demucs_model(model_name='htdemucs')  # Use your model name here
    refined_audio = apply_demucs_to_audio(clean_audio, demucs_model)

    # Save the refined audio
    output_path_refined = 'refined_audio_demucs.wav'
    sf.write(output_path_refined, refined_audio, sr)

except Exception as e:
    # Catch any errors in the entire Demucs application process
    print(f"An error occurred while applying Demucs: {e}")
    refined_audio = clean_audio  # In case of error, use the cleaned audio as fallback

# %% md
# # 7. Apply Additional Noise Reduction with noisereduce
# %%
try:
    refined_audio_reduced = nr.reduce_noise(y=refined_audio, sr=sr, prop_decrease=0.7)

    # Save the further refined audio
    sf.write(output_path_refined, refined_audio_reduced, sr)

except Exception as e:
    print(f"An error occurred during additional noise reduction: {e}")

# %% md
# # 8. Visualize Results
# ### 8.1 Plot Waveforms
# %%
plot_waveform(noisy_audio, sr, "Noisy Audio")
plot_waveform(audio_audio, sr, "Original Clean Audio")
plot_waveform(clean_audio, sr, "Cleaned Audio")
plot_waveform(refined_audio, sr, "Refined Audio (Demucs)")
plot_waveform(refined_audio_reduced, sr, "Refined Audio with Noisereduce")

# %% md
# ### 8.2 Plot Spectrograms
# %%
plot_spectrogram(noisy_audio, sr, "Noisy Audio Spectrogram")
plot_spectrogram(audio_audio, sr, "Original Clean Audio Spectrogram")
plot_spectrogram(clean_audio, sr, "Cleaned Audio Spectrogram")
plot_spectrogram(refined_audio, sr, "Refined Audio (Demucs) Spectrogram")
plot_spectrogram(refined_audio_reduced, sr, "Refined Audio with Noisereduce Spectrogram")
