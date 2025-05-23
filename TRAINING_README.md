# Training a Japanese Text-to-Speech Model

This document provides instructions on how to use the `train_japanese_tts.py` script to train a Japanese Text-to-Speech (TTS) model based on the DiaModel architecture.

**Note:** This training script is experimental and assumes you have a compatible environment (PyTorch with CUDA, necessary libraries) and a prepared Japanese speech dataset. The quality of the trained model will heavily depend on the quality and quantity of your data, as well as hyperparameter tuning.

## 1. Prerequisites

### a. Dependencies
Ensure you have the following Python libraries installed, in addition to the main project dependencies:
- `torch` (with CUDA support for GPU training)
- `torchaudio`
- `sentencepiece`
- `pyopenjtalk` (for Japanese phonemization)
- `pandas` (for reading transcript files)
- `PyYAML` (for configuration files)
- `matplotlib` (for logging sample spectrograms)
- `tensorboard` (for logging training progress)

You can typically install these using pip:
```bash
pip install torch torchaudio pandas PyYAML pyopenjtalk sentencepiece matplotlib tensorboard
```
Ensure `torch` is installed with the correct CUDA version for your GPU. Refer to the official PyTorch installation guide.

### b. Japanese Speech Dataset
You need a dataset consisting of:
- **Audio files:** High-quality Japanese audio recordings (e.g., WAV files). It's recommended to have a consistent sample rate (e.g., 22050 Hz or 44100 Hz).
- **Transcripts:** A metadata file (e.g., `transcripts.csv`) mapping audio filenames to their corresponding Japanese transcriptions. The script expects columns like `wav_filename` and `transcript`.

**Dataset Structure Example:**
Assume your dataset is in `/path/to/your/japanese_speech_dataset/`:
```
/path/to/your/japanese_speech_dataset/
|-- wavs/                            # Directory containing your .wav files
|   |-- audio_0001.wav
|   |-- audio_0002.wav
|   |-- ...
|-- transcripts.csv                  # Metadata file
```

### c. Japanese SentencePiece Model
You need a SentencePiece model (`.model` file) trained on a large corpus of Japanese text. This model is used for text tokenization if phonemization is disabled, and potentially for text cleaning. You must provide the path to this model in the training configuration.

### d. Base DiaModel Configuration (Optional but Recommended)
While the training script can define model parameters, it's cleaner to reference a base `DiaConfig` JSON file (similar to the one used for inference) for the core `DiaModel` architecture (encoder/decoder layers, dimensions, etc.). The training script will then override necessary parts like `src_vocab_size` based on the phoneme vocabulary.

## 2. Configuration (`config_train_ja.yaml`)

Create a YAML configuration file (e.g., `config_train_ja.yaml`) to specify all training parameters. Below is an example structure with explanations:

```yaml
# --- Data Configuration ---
data:
  dataset_path: "/path/to/your/japanese_speech_dataset/" # REQUIRED: Root directory of the dataset
  transcript_filename: "transcripts.csv" # Name of the transcript CSV file
  audio_directory: "wavs/"             # Subdirectory for audio files, relative to dataset_path
  
  sentencepiece_model_path: "/path/to/your/japanese.model" # REQUIRED: Path to your SentencePiece model

  use_phonemizer: true      # Set to true to use pyopenjtalk for phonemization (recommended)
                            # If false, uses SentencePiece IDs directly.
  
  # Audio processing parameters
  sample_rate: 44100        # Target sample rate for audio
  n_fft: 2048               # FFT size for STFT
  hop_length: 512           # Hop length for STFT (determines mel frame rate)
  win_length: 2048          # Window length for STFT
  n_mels: 128               # Number of mel bins
  fmin: 0                   # Minimum frequency for mel filterbank
  fmax: 8000                # Maximum frequency for mel filterbank (adjust based on sample_rate)

# --- Model Configuration ---
model:
  # Parameters for the DiaModel architecture (acoustic model part)
  # These define the structure of the encoder, decoder, etc.
  dia_config_params:
    encoder: { n_layer: 6, n_embd: 512, n_hidden: 2048, n_head: 8, head_dim: 64 }
    decoder: { n_layer: 6, n_embd: 512, n_hidden: 2048, gqa_query_heads: 8, kv_heads: 8, gqa_head_dim: 64, cross_query_heads: 8, cross_head_dim: 64 }
    # src_vocab_size will be overridden by the script based on PHONEME_LIST size.
    # tgt_vocab_size is for the original DiaModel's DAC code prediction, not directly used by acoustic model if outputting mels.
    tgt_vocab_size: 1028 
    dropout: 0.1
    # normalization_layer_epsilon, weight_dtype, rope_min_timescale, rope_max_timescale can also be set here if needed.

  # Dummy DataConfig values for DiaModel initialization (not for dataset loading)
  # These are used to satisfy DiaModel's internal config structure.
  dia_model_data_config_override:
    text_length_dummy_for_model: 512   # Max expected phoneme sequence length for model init
    audio_length_dummy_for_model: 2048 # Max expected mel frames for model init
    # channels_dummy_for_model: 9 # Not critical for acoustic model if not used

  # DAC vocoder settings (if fine-tuning DAC)
  # fine_tune_dac: false # Set to true to attempt fine-tuning the DAC model
  # dac_model_path: "path_or_name_for_dac_model" # Path to pre-trained DAC model if fine_tune_dac is true

# --- Training Configuration ---
training:
  output_directory: "./training_output_ja/" # Directory for checkpoints, logs, samples
  epochs: 1000
  batch_size: 32
  learning_rate: 0.0001
  weight_decay: 0.01
  grad_clip_thresh: 1.0
  
  adam_betas: [0.9, 0.98]    # Betas for AdamW optimizer
  adam_eps: 1.0e-9          # Epsilon for AdamW optimizer

  save_every_n_epochs: 5    # How often to save a checkpoint
  # keep_latest_n_checkpoints: 3 # Optional: for future cleanup logic

  log_every_n_steps: 100    # Log training metrics to console/TensorBoard
  validate_every_n_epochs: 5 # How often to log validation samples (spectrograms)

# --- Environment ---
environment:
  seed: 42
  device: "cuda" # "cuda" or "cpu"
```

**Key Configuration Notes:**
- **`dia_config_params`**: This section is crucial. It defines the architecture of the `DiaModel` being trained as the acoustic model (text-to-mel). The `src_vocab_size` will be automatically set by the script based on the size of the phoneme vocabulary (`PHONEME_LIST` defined in `train_japanese_tts.py`).
- **`PHONEME_LIST` in `train_japanese_tts.py`**: This list defines the vocabulary of phonemes used by the model. You may need to adjust it based on the actual phoneme set output by `pyopenjtalk` for your specific Japanese data, or if you choose a different phonemizer. Ensure special tokens like `<pad>`, `<bos>`, `<eos>`, `<unk>` are included.
- **DiaModel Adaptation**: The training script uses the `DiaModel` architecture. However, the original `DiaModel` is designed for text-to-DAC-codes. For TTS, it needs to predict mel spectrograms. The provided `JapaneseTTSModel` wrapper in the script contains **placeholders** for this adaptation. You might need to modify `DiaModel` internals or implement a more Tacotron2-like decoder if the adaptation is not straightforward. This is a critical point for successful training.

## 3. Running the Training Script

Once your dataset and configuration file are ready:

1.  **Open your terminal.**
2.  **Navigate to the repository directory.**
3.  **Run the script:**
    ```bash
    python train_japanese_tts.py --config_path /path/to/your/config_train_ja.yaml 
    ```

**Optional Command-Line Overrides:**
You can override some parameters from the config file directly via CLI:
- `--dataset_path`: Path to your dataset.
- `--output_directory`: Where to save outputs.
- `--epochs`: Number of training epochs.
- `--batch_size`: Batch size.
- `--learning_rate`: Learning rate.
- `--resume_checkpoint_path`: Path to a `.pth` checkpoint file to resume training.

Example with overrides:
```bash
python train_japanese_tts.py \
    --config_path /path/to/your/config_train_ja.yaml \
    --epochs 500 \
    --batch_size 16 \
    --output_directory ./my_japanese_tts_training
```

## 4. Monitoring Training (TensorBoard)

The script logs training progress to TensorBoard. To view it:
1.  Open a new terminal.
2.  Navigate to your project directory.
3.  Run: `tensorboard --logdir ./training_output_ja/logs` (or your specified output directory + `/logs`).
4.  Open the URL provided by TensorBoard (usually `http://localhost:6006`) in your web browser.

You should see metrics like training loss, learning rate, and generated mel-spectrogram samples from validation.

## 5. Important Considerations & Troubleshooting
- **Model Adaptation**: As mentioned, the core `DiaModel` might require significant changes to effectively predict mel spectrograms from text. The current training script provides a structural placeholder. If the model doesn't learn, this is the first area to investigate.
- **Phoneme Set**: Ensure `PHONEME_LIST` in `train_japanese_tts.py` accurately reflects the output of `pyopenjtalk` for your data. Unknown phonemes will be mapped to `<unk>`, which can degrade quality.
- **Hyperparameters**: The provided learning rate, batch size, etc., are examples. You will likely need to tune these for your specific dataset and GPU capacity.
- **Memory Usage**: Training TTS models can be memory-intensive. If you encounter CUDA out-of-memory errors, reduce `batch_size`.
- **Data Quality**: The quality of your audio data and transcripts is paramount. Noise, inconsistencies, and inaccurate transcriptions will negatively impact model performance.

Good luck with your training!
