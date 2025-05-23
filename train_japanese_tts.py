import os
import re
import unicodedata # For text normalization
import random # Added for seeding
import argparse # Added for __main__

import torch
import torch.nn as nn # Added for model and loss
import torch.nn.functional as F # Added for loss
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader # Added DataLoader
from torch.optim import AdamW # Added AdamW
from torch.optim.lr_scheduler import LambdaLR # Example scheduler
from torch.cuda.amp import GradScaler, autocast # For mixed-precision training
from torch.utils.tensorboard import SummaryWriter # Added SummaryWriter

import sentencepiece as spm
import pyopenjtalk
import pandas as pd # For reading transcript CSV
import numpy as np # Ensure numpy is imported
import yaml # Added for config loading
import matplotlib.pyplot as plt # For saving spectrograms as images
# import soundfile as sf # If we attempt to save dummy audio

# Assuming these can be imported from the project structure
from dia.model import DiaModel 
from dia.config import DiaConfig, ModelConfig, DataConfig, EncoderConfig, DecoderConfig 
import dac

# For AttrDict if used for config
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text) # NFKC normalization
    text = text.lower() # Optional: convert to lowercase if your phonemizer/tokenizer expects it
    text = re.sub(r'[\ufeff]', '', text) # Remove BOM
    # Add any other specific cleaning rules needed for Japanese text if known
    return text

# Define a basic list of Japanese phonemes that pyopenjtalk might output.
# This list will likely need refinement based on the actual output of pyopenjtalk.
# Also include special symbols: PAD, BOS, EOS, UNK
PHONEME_LIST = [
    '<pad>', '<bos>', '<eos>', '<unk>',
    'A', 'I', 'U', 'E', 'O', 'N', # Using uppercase for phonemes for clarity, can be lowercase
    'k', 's', 't', 'n', 'h', 'm', 'y', 'r', 'w', 'g', 'z', 'd', 'b', 'p',
    'ky', 'sh', 'ch', 'ts', 'ny', 'hy', 'f', 'my', 'ry', 'gy', 'j', 'by', 'py',
    'cl', # for っ (sokuon - glottal stop)
    'v', # for ヴ, though 'b' might be used by pyopenjtalk
    # Pause or punctuation phonemes if needed, e.g. 'sp' for short pause
]
# Ensure uniqueness and remove duplicates if any copy-paste errors
PHONEME_LIST = sorted(list(set(PHONEME_LIST))) # Ensure unique and sorted for consistency

PHONEME_TO_ID = {ph: i for i, ph in enumerate(PHONEME_LIST)}
ID_TO_PHONEME = {i: ph for i, ph in enumerate(PHONEME_LIST)}

# Special token IDs (ensure these are in PHONEME_LIST)
PAD_ID = PHONEME_TO_ID['<pad>']
BOS_ID = PHONEME_TO_ID['<bos>']
EOS_ID = PHONEME_TO_ID['<eos>']
UNK_ID = PHONEME_TO_ID['<unk>']

class JapaneseSpeechDataset(Dataset):
    def __init__(self, data_config, sp_model_path, use_phonemizer=True, text_cleaners=None):
        self.data_config = data_config # Should be an object or dict with attributes like dataset_path etc.
        self.audiopath_transcript = self._load_transcripts(
            self.data_config.dataset_path, 
            self.data_config.transcript_filename
        )
        self.audio_dir = os.path.join(self.data_config.dataset_path, self.data_config.audio_directory)
        
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(sp_model_path) # sp_model_path from main config
        
        self.use_phonemizer = use_phonemizer
        self.text_cleaners = text_cleaners if text_cleaners else [normalize_text]

        self.mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=self.data_config.sample_rate,
            n_fft=self.data_config.n_fft,
            win_length=self.data_config.win_length,
            hop_length=self.data_config.hop_length,
            f_min=self.data_config.fmin,
            f_max=self.data_config.fmax,
            n_mels=self.data_config.n_mels,
            power=1.0, 
            normalized=False, 
        )

    def _load_transcripts(self, dataset_path, transcript_filename):
        transcript_path = os.path.join(dataset_path, transcript_filename)
        if not os.path.exists(transcript_path):
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
        
        df = pd.read_csv(transcript_path) 
        if 'wav_filename' not in df.columns or 'transcript' not in df.columns:
            raise ValueError("Transcript CSV must contain 'wav_filename' and 'transcript' columns.")
        return list(zip(df['wav_filename'], df['transcript']))

    def _clean_text(self, text):
        for cleaner_fn in self.text_cleaners:
            text = cleaner_fn(text)
        return text

    def _get_phonemes(self, text):
        # pyopenjtalk.g2p returns phonemes like "k o N n i ch i w a"
        # It can also return with accent information, which we might want to strip for simplicity first.
        # Example: pyopenjtalk.g2p("こんにちは世界", kana=False) -> 'k o N n i ch i w a s e k a i'
        raw_phonemes = pyopenjtalk.g2p(text, kana=False) 
        
        # Basic processing: split and handle potential variations if stress/accent markers are present.
        # For simplicity, we assume space-separated phonemes and map them.
        # More advanced parsing might be needed if pyopenjtalk includes complex markers.
        phoneme_list = raw_phonemes.split(' ')
        phoneme_list = [ph for ph in phoneme_list if ph.strip()] # Filter empty strings
        
        phoneme_sequence = [BOS_ID] + [PHONEME_TO_ID.get(p.lower(), UNK_ID) for p in phoneme_list] + [EOS_ID] # Use .lower() for safety if PHONEME_LIST is lowercase
        return torch.LongTensor(phoneme_sequence)

    def __getitem__(self, index):
        wav_filename, raw_text = self.audiopath_transcript[index]
        audio_path = os.path.join(self.audio_dir, wav_filename)

        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return None 

        if sr != self.data_config.sample_rate:
            resampler = T.Resample(sr, self.data_config.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1: # Ensure mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Mel Spectrogram
        mel_spectrogram = self.mel_spectrogram_transform(waveform.cpu()) 
        mel_spectrogram = torch.squeeze(mel_spectrogram, 0) 
        mel_spectrogram = torch.log(torch.clamp(mel_spectrogram, min=1e-5)) # Log scale

        # Text Processing
        cleaned_text = self._clean_text(raw_text)
        
        if self.use_phonemizer:
            text_sequence = self._get_phonemes(cleaned_text)
        else:
            # Fallback to SentencePiece IDs if not using phonemizer
            # This path would need its own BOS/EOS/PAD ID handling from sp_model
            token_ids = self.sp_model.encode_as_ids(cleaned_text)
            # Make sure sp_model's BOS/EOS are mapped to consistent IDs if mixing with phoneme IDs later,
            # or ensure distinct vocabularies. For now, assume separate usage or consistent special token IDs.
            text_sequence = torch.LongTensor([self.sp_model.bos_id()] + token_ids + [self.sp_model.eos_id()])
            # Note: If use_phonemizer=False, PAD_ID for collate_fn should align with sp_model.pad_id()

        return (text_sequence, mel_spectrogram, len(text_sequence), mel_spectrogram.shape[1])

    def __len__(self):
        return len(self.audiopath_transcript)

def collate_fn_padd(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    texts, mels, text_lengths, mel_lengths = zip(*batch)

    max_text_len = max(text_lengths)
    # Determine pad_id. If using phonemes, it's PAD_ID from phoneme map.
    # If using SP, it should be sp_model.pad_id(). This needs to be consistent.
    # For now, assume global PAD_ID is used, implying phoneme path is primary.
    current_pad_id = PAD_ID 
    # A more robust way would be to pass pad_id to collate_fn or get it from dataset object.
    # if hasattr(batch[0]._dataset_ref_if_needed, 'get_pad_id'): current_pad_id = dataset.get_pad_id()
    
    padded_texts = torch.full((len(texts), max_text_len), current_pad_id, dtype=torch.long)
    for i, t in enumerate(texts):
        padded_texts[i, :t.shape[0]] = t
    
    text_lengths_tensor = torch.LongTensor(text_lengths)

    max_mel_len = max(mel_lengths)
    n_mels = mels[0].shape[0] 
    # Mel padding value should be log(1e-5) if mels are log-scaled and clamped at 1e-5
    mel_pad_value = torch.log(torch.tensor(1e-5)) 
    padded_mels = torch.full((len(mels), n_mels, max_mel_len), mel_pad_value, dtype=torch.float)
    for i, m in enumerate(mels):
        padded_mels[i, :, :m.shape[1]] = m
    
    mel_lengths_tensor = torch.LongTensor(mel_lengths)
    
    return padded_texts, text_lengths_tensor, padded_mels, mel_lengths_tensor

class Tacotron2Loss(nn.Module): # Using Tacotron2-style loss as an example
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, mel_out, mel_out_postnet, mel_target, gate_out, gate_target):
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        # Ensure mel_target is expanded if mel_out/mel_out_postnet have different channel dims
        # Typically, for TTS, mel_target is (B, n_mels, T_mel)
        # mel_out and mel_out_postnet should match this.
        
        mel_loss = F.mse_loss(mel_out, mel_target) + F.mse_loss(mel_out_postnet, mel_target) # Common to use MSE or L1
        gate_loss = nn.BCEWithLogitsLoss()(gate_out.view(-1), gate_target.view(-1)) # Ensure gate_out and gate_target are 1D
        return mel_loss + gate_loss

class JapaneseTTSModel(nn.Module):
    def __init__(self, model_config_dict, data_config_dict_for_model, training_config_dict): # Renamed data_config_dict
        super().__init__()
        
        enc_conf = EncoderConfig(**model_config_dict['encoder'])
        dec_conf = DecoderConfig(**model_config_dict['decoder'])
        
        PHONEME_VOCAB_SIZE = len(PHONEME_TO_ID) 
        
        model_config_pydantic = ModelConfig(
            encoder=enc_conf,
            decoder=dec_conf,
            src_vocab_size=PHONEME_VOCAB_SIZE, 
            tgt_vocab_size=model_config_dict.get('tgt_vocab_size', 1028), 
            dropout=model_config_dict.get('dropout', 0.0),
            normalization_layer_epsilon=model_config_dict.get('normalization_layer_epsilon', 1e-5),
            weight_dtype=model_config_dict.get('weight_dtype', "float32"),
            rope_min_timescale=model_config_dict.get('rope_min_timescale', 1),
            rope_max_timescale=model_config_dict.get('rope_max_timescale', 10000)
        )
        
        # This DataConfig is a minimal one for DiaModel structure, not for dataset loading.
        dia_data_cfg_for_model_init = DataConfig(
            text_length=data_config_dict_for_model.get('text_length_dummy_for_model', 512), 
            audio_length=data_config_dict_for_model.get('audio_length_dummy_for_model', 2048), 
            channels=data_config_dict_for_model.get('channels_dummy_for_model', 9) 
        )

        dia_cfg_for_acoustic_model = DiaConfig(model=model_config_pydantic, data=dia_data_cfg_for_model_init)
        
        self.acoustic_model = DiaModel(dia_cfg_for_acoustic_model)
        
        # **Critical Assumption**: Adapting DiaModel (text-to-DAC-codes) to be a text-to-mel model.
        # This likely requires significant internal changes to DiaModel's decoder 
        # or adding a new projection head to output mel spectrograms instead of code logits.
        # For this subtask, we assume self.acoustic_model can be made to produce mel-like outputs.
        # A common approach is to take the encoder from DiaModel and pair it with a Tacotron2-style decoder.
        # Or, modify DiaModel's decoder output layer. This is a MAJOR simplification for now.

        self.fine_tune_dac = training_config_dict.get('fine_tune_dac', False)
        if self.fine_tune_dac:
            self.dac_model = dac.DAC.load(training_config_dict['dac_model_path']) 
            # Potentially unfreeze parts of DAC model here if needed for fine-tuning
            # for param in self.dac_model.parameters(): param.requires_grad = True 
        else:
            self.dac_model = None

    def forward(self, text_padded, text_lengths, mel_targets_padded, mel_target_lengths):
        # This forward pass is highly conceptual and assumes DiaModel is adapted for mel output.
        # It needs to output:
        # 1. mel_outputs_raw: Mel spectrograms before a potential postnet. (B, n_mels, T_mel)
        # 2. mel_outputs_postnet: Mel spectrograms after a postnet (if any). (B, n_mels, T_mel)
        # 3. gate_outputs: Logits for stop token prediction. (B, T_mel)
        
        # Replace with actual call to the adapted acoustic_model
        # e.g., mel_outputs_raw, mel_outputs_postnet, gate_outputs = self.acoustic_model.generate_mels_for_training(...)
        
        # --- Placeholder for DiaModel text-to-mel output ---
        # This is the most critical part and requires DiaModel to be modified or wrapped.
        # For now, simulate outputs for structural correctness of the training loop.
        B, N_MELS, T_MEL_MAX_TARGET = mel_targets_padded.shape
        
        # These dummy outputs need to have require_grad=True if they were actual model outputs
        dummy_mel_out = torch.randn(B, N_MELS, T_MEL_MAX_TARGET, device=text_padded.device, requires_grad=True)
        dummy_mel_out_postnet = torch.randn(B, N_MELS, T_MEL_MAX_TARGET, device=text_padded.device, requires_grad=True)
        # Gate output should correspond to the max *decoder* steps, which might be different from T_MEL_MAX_TARGET
        # For simplicity, let's assume it matches T_MEL_MAX_TARGET for now.
        dummy_gate_out = torch.randn(B, T_MEL_MAX_TARGET, device=text_padded.device, requires_grad=True) 
        
        # Mask mel targets for loss calculation based on mel_target_lengths
        # Create a mask (B, 1, T_MEL_MAX_TARGET)
        mel_mask = ~self.create_mask(mel_target_lengths, T_MEL_MAX_TARGET).unsqueeze(1)
        mel_targets_padded.masked_fill_(mel_mask, 0.0) # Zero out padded regions of target
        # Also apply mask to outputs if they are longer than targets due to fixed output size
        dummy_mel_out.masked_fill_(mel_mask, 0.0)
        dummy_mel_out_postnet.masked_fill_(mel_mask, 0.0)

        # Create gate targets (binary, 1 if past actual mel length)
        gate_targets = torch.zeros(B, T_MEL_MAX_TARGET, device=text_padded.device)
        for i in range(B):
            gate_targets[i, mel_target_lengths[i]-1:] = 1.0 # Frame becomes 1 at and after the end
        
        # DAC fine-tuning loss (conceptual)
        dac_loss = None
        if self.dac_model and self.fine_tune_dac and self.training:
            # Example: DAC reconstructs predicted mels, and we compare to original audio's mels (or audio)
            # This part is highly dependent on how DAC fine-tuning is implemented.
            # For instance, if DAC can output mels from audio:
            #    mels_from_dac_via_audio = self.dac_model.encode(original_audio_batch) # Requires original audio
            #    dac_loss = F.mse_loss(self.dac_model.decode_to_mels(mel_outputs_postnet), mels_from_dac_via_audio)
            pass # Placeholder for DAC fine-tuning loss calculation

        return dummy_mel_out, dummy_mel_out_postnet, dummy_gate_out, gate_targets, dac_loss

    def create_mask(self, lengths, max_len):
        # (B, max_len) -> True for padding
        return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) >= lengths.unsqueeze(1)

def log_validation_samples(model, writer, global_step, device, config, sp_model, phonemizer_fn_placeholder):
    model.eval() # Set model to evaluation mode
    
    # Define a few fixed Japanese text samples for validation
    # These should be sentences that are representative.
    validation_texts = [
        "こんにちは、これはテストです。",
        "今日の天気は晴れです。",
        "音声合成の品質を確認しています。"
    ]

    # Prepare text processing functions (simplified from dataset, assuming phonemizer_fn_placeholder exists)
    # In a real setup, you'd use the same text cleaning and phonemization as in the dataset.
    def _clean_text_for_val(text): # Simplified version of dataset's cleaner
        text = unicodedata.normalize("NFKC", text)
        text = text.lower()
        return text

    # The phonemizer_fn_placeholder should ideally be the actual phonemization function used in the dataset.
    # For example: lambda text: [BOS_ID] + [PHONEME_TO_ID.get(p, UNK_ID) for p in pyopenjtalk.g2p(text, kana=False).split(' ') if p.strip()] + [EOS_ID]
    
    for i, text in enumerate(validation_texts):
        cleaned_text = _clean_text_for_val(text)
        
        # Process text into phoneme IDs (or SentencePiece IDs if not using phonemizer)
        # This is a placeholder for the actual text processing pipeline from the Dataset
        # For now, let's assume phonemization:
        if config.data.use_phonemizer:
            # This assumes phonemizer_fn_placeholder is correctly defined and works
            # For simplicity, we'll use a direct call to pyopenjtalk here,
            # but it should ideally share the exact logic with the dataset.
            try:
                raw_phonemes = pyopenjtalk.g2p(cleaned_text, kana=False)
                phoneme_list = [ph for ph in raw_phonemes.split(' ') if ph.strip()]
                phoneme_ids = [BOS_ID] + [PHONEME_TO_ID.get(p.lower(), UNK_ID) for p in phoneme_list] + [EOS_ID] # Use .lower()
            except Exception as e:
                print(f"Error phonemizing validation text '{cleaned_text}': {e}")
                continue
        else: # Using SentencePiece
            phoneme_ids = [sp_model.bos_id()] + sp_model.encode_as_ids(cleaned_text) + [sp_model.eos_id()]

        text_tensor = torch.LongTensor(phoneme_ids).unsqueeze(0).to(device) # (1, seq_len)
        text_length_tensor = torch.LongTensor([len(phoneme_ids)]).to(device) # (1)

        with torch.no_grad():
            # The model.forward or a dedicated inference method should generate mel spectrograms
            # Current model.forward is (text_padded, text_lengths, mel_targets_padded, mel_target_lengths)
            # We need a way to call the model for inference (text -> mel)
            # This might mean calling a sub-module or having a different signature for eval.
            # For now, let's assume the model's forward pass can be called with only text inputs
            # for inference, or we call a specific inference method.
            # This is a MAJOR simplification and likely needs a dedicated inference entry point in the model.
            try:
                # Placeholder: This assumes JapaneseTTSModel can run inference this way.
                # It's very likely the current `model.forward` signature is not suitable for direct inference.
                # A dedicated `model.infer(text_tensor, text_length_tensor)` would be better.
                # For now, we'll just get the dummy outputs to show logging structure.
                # We pass None for mel targets as we are in inference/validation mode.
                # The model's forward should handle this case.
                
                # This is a conceptual call. The actual inference method might differ.
                # It should return mel_outputs_postnet.
                # For now, let's simulate a mel output for logging purposes.
                # This part needs to be correctly implemented in JapaneseTTSModel.
                
                # If JapaneseTTSModel.forward can take mel_targets_padded=None
                # mel_output_postnet, _, _, _, _ = model(text_tensor, text_length_tensor, None, None)
                
                # SIMULATED MEL OUTPUT FOR LOGGING (replace with actual model inference)
                simulated_mel_output = torch.randn(1, config.data.n_mels, 100).to(device) # (B, n_mels, T_mel_simulated)
                mel_to_log = simulated_mel_output.squeeze(0).cpu().numpy()

                # Log spectrogram image to TensorBoard
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.imshow(mel_to_log, aspect="auto", origin="lower", interpolation='none')
                ax.set_title(f"Validation Mel: {text[:20]}...")
                ax.set_xlabel("Frames")
                ax.set_ylabel("Mel Bins")
                plt.tight_layout()
                writer.add_figure(f"Validation/mel_spectrogram_sample_{i}", fig, global_step)
                plt.close(fig) # Close the figure to free memory

                # Placeholder: If DAC is available and can convert mel to audio
                # if model.dac_model:
                #     try:
                #         # This assumes dac_model has a method to decode mels from our acoustic model
                #         # And that mel_output_postnet is the correct input format for it.
                        # audio_tensor = model.dac_model.decode(mel_output_postnet) # Fictional method
                        # audio_np = audio_tensor.squeeze().cpu().numpy()
                        # writer.add_audio(f"Validation/audio_sample_{i}", audio_np, global_step, sample_rate=config.data.sample_rate)
                        # sf.write(os.path.join(config.training.output_directory, "samples", f"val_sample_epoch{global_step}_{i}.wav"), audio_np, config.data.sample_rate)
                #     except Exception as e:
                #         print(f"Error generating or logging audio for validation sample {i}: {e}")

            except Exception as e:
                print(f"Error during validation sample generation for text '{text}': {e}")
                import traceback
                traceback.print_exc()
                continue
    
    model.train() # Set model back to training mode

def train(rank, cli_args, config_obj): # Changed config to config_obj to avoid conflict
    # Setup (logging, seeding, device)
    torch.manual_seed(config_obj.environment.seed)
    np.random.seed(config_obj.environment.seed)
    random.seed(config_obj.environment.seed)
    device = torch.device(config_obj.environment.device)
    
    output_dir = config_obj.training.output_directory
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    logs_dir = os.path.join(output_dir, "logs")
    samples_dir = os.path.join(output_dir, "samples") # For validation samples
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True) # Create samples directory
    writer = SummaryWriter(log_dir=logs_dir)

    # Initialize Tokenizer and Phonemizer (as before, ensure they are loaded)
    sp_model = spm.SentencePieceProcessor()
    sp_model.load(config_obj.data.sentencepiece_model_path)
    # phonemizer setup (pyopenjtalk is used directly in dataset for now)

    # Datasets and DataLoaders
    train_dataset = JapaneseSpeechDataset(config_obj.data, config_obj.data.sentencepiece_model_path, use_phonemizer=config_obj.data.use_phonemizer)
    train_loader = DataLoader(train_dataset, batch_size=config_obj.training.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn_padd)

    # Model, Optimizer, Loss
    model = JapaneseTTSModel(
        model_config_dict=config_obj.model.dia_config_params, 
        data_config_dict_for_model=config_obj.model.dia_model_data_config_override, # new config section for this
        training_config_dict=config_obj.training
    ).to(device)
    
    optimizer_params = list(model.acoustic_model.parameters())
    if model.fine_tune_dac and model.dac_model:
        optimizer_params += list(model.dac_model.parameters())

    optimizer = AdamW(optimizer_params, lr=config_obj.training.learning_rate, betas=config_obj.training.adam_betas, eps=config_obj.training.adam_eps, weight_decay=config_obj.training.weight_decay)
    criterion_tts = Tacotron2Loss().to(device)
    
    scaler = GradScaler(enabled=(config_obj.environment.device == 'cuda'))

    start_epoch = 0
    global_step = 0

    if cli_args.resume_checkpoint_path and os.path.exists(cli_args.resume_checkpoint_path):
        print(f"Resuming from checkpoint: {cli_args.resume_checkpoint_path}")
        checkpoint = torch.load(cli_args.resume_checkpoint_path, map_location=device)
        # Robust loading: only load if keys exist and shapes match (simplified here)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False) 
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('global_step', 0)
        if 'scaler_state_dict' in checkpoint and scaler:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
    else:
        print("Starting training from scratch.")


    # Training Loop
    for epoch in range(start_epoch, config_obj.training.epochs):
        model.train()
        epoch_loss_total = 0
        for batch_idx, batch_data in enumerate(train_loader):
            if batch_data is None: continue 

            texts_padded, text_lengths, mels_padded, mel_lengths = batch_data
            texts_padded = texts_padded.to(device, non_blocking=True)
            text_lengths = text_lengths.to(device, non_blocking=True)
            mels_padded = mels_padded.to(device, non_blocking=True) 
            mel_lengths = mel_lengths.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=scaler.is_enabled()):
                mel_out, mel_out_postnet, gate_out, gate_targets, dac_loss_val = model(texts_padded, text_lengths, mels_padded, mel_lengths)
                
                # Main TTS loss
                tts_loss = criterion_tts(mel_out, mel_out_postnet, mels_padded, gate_out, gate_targets)
                
                total_loss = tts_loss
                if dac_loss_val is not None: # Add DAC fine-tuning loss if applicable
                    total_loss += dac_loss_val 
            
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer) 
            torch.nn.utils.clip_grad_norm_(model.parameters(), config_obj.training.grad_clip_thresh)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss_total += total_loss.item()
            
            if global_step % config_obj.training.log_every_n_steps == 0 and rank == 0: # Log only for rank 0
                writer.add_scalar('Loss/train_total', total_loss.item(), global_step)
                writer.add_scalar('Loss/tts_mel_gate', tts_loss.item(), global_step)
                if dac_loss_val is not None:
                    writer.add_scalar('Loss/dac_finetune', dac_loss_val.item(), global_step)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                print(f"Epoch: {epoch}, Step: {global_step}, Loss: {total_loss.item():.4f}, TTS Loss: {tts_loss.item():.4f}")
            
            global_step += 1
        
        avg_epoch_loss = epoch_loss_total / len(train_loader)
        if rank == 0:
            writer.add_scalar('Loss/epoch_avg_total', avg_epoch_loss, epoch)
            print(f"Epoch: {epoch} completed. Average Loss: {avg_epoch_loss:.4f}")
        
        # Call validation logging function
        if rank == 0 and (epoch % config_obj.training.get('validate_every_n_epochs', 5) == 0 or epoch == config_obj.training.epochs - 1): # Added .get for default
            print(f"Logging validation samples for epoch {epoch}...")
            # We need a placeholder for the phonemizer function if it's not directly accessible
            # This is a simplification. In a full setup, the phonemizer instance or a function
            # that uses it would be passed around or accessible.
            # For now, pyopenjtalk is called directly in log_validation_samples.
            log_validation_samples(model, writer, global_step, device, config_obj, sp_model, None)
            # The 'None' for phonemizer_fn_placeholder means pyopenjtalk is used directly inside.


        if (epoch % config_obj.training.save_every_n_epochs == 0 or epoch == config_obj.training.epochs - 1) and rank == 0:
            checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch}.pth")
            save_payload = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(), # Or model.module.state_dict() for DDP
                'optimizer_state_dict': optimizer.state_dict(),
                # 'config': config_obj.to_dict() # If config_obj is Pydantic or has to_dict()
            }
            if scaler: save_payload['scaler_state_dict'] = scaler.state_dict()
            torch.save(save_payload, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            
    if rank == 0: writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Japanese TTS Model")
    parser.add_argument("--config_path", type=str, required=True, help="Path to training config YAML file.")
    parser.add_argument("--dataset_path", type=str, help="Override dataset_path in config.")
    parser.add_argument("--output_directory", type=str, help="Override output_directory in config.")
    parser.add_argument("--epochs", type=int, help="Override training epochs in config.")
    parser.add_argument("--batch_size", type=int, help="Override batch_size in config.")
    parser.add_argument("--learning_rate", type=float, help="Override learning_rate in config.")
    parser.add_argument("--resume_checkpoint_path", type=str, default="", help="Path to checkpoint to resume training.") # Default to empty string
    
    cli_args = parser.parse_args()

    with open(cli_args.config_path, 'r', encoding='utf-8') as f: # Added encoding
        config_dict = yaml.safe_load(f)

    # Override config with CLI args
    if cli_args.dataset_path: config_dict['data']['dataset_path'] = cli_args.dataset_path
    if cli_args.output_directory: config_dict['training']['output_directory'] = cli_args.output_directory
    if cli_args.epochs is not None: config_dict['training']['epochs'] = cli_args.epochs
    if cli_args.batch_size is not None: config_dict['training']['batch_size'] = cli_args.batch_size
    if cli_args.learning_rate is not None: config_dict['training']['learning_rate'] = cli_args.learning_rate
    
    def to_attr_dict(d):
        if isinstance(d, dict):
            new_d = AttrDict()
            for k, v in d.items():
                new_d[k] = to_attr_dict(v)
            return new_d
        elif isinstance(d, list):
            return [to_attr_dict(i) for i in d]
        return d
    config_obj = to_attr_dict(config_dict)

    if 'PHONEME_LIST' not in globals():
        print("Error: PHONEME_LIST not defined. Make sure it's defined globally in train_japanese_tts.py")
        exit(1)
    
    # Ensure model.dia_config_params and model.dia_model_data_config_override are present
    if 'dia_config_params' not in config_obj.model:
        print("Error: config.model.dia_config_params not found in YAML.")
        exit(1)
    if 'dia_model_data_config_override' not in config_obj.model:
        config_obj.model.dia_model_data_config_override = AttrDict({}) # Provide empty if not present
        print("WARN: config.model.dia_model_data_config_override not found. Using default empty dict.")

    # Add default for validate_every_n_epochs if not in config
    if 'validate_every_n_epochs' not in config_obj.training:
        print("WARN: 'validate_every_n_epochs' not found in training config. Defaulting to 5.")
        config_obj.training.validate_every_n_epochs = 5


    print("Starting training...")
    train(0, cli_args, config_obj) # rank 0 for single GPU
    print("Training finished.")
