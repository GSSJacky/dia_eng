<p align="center">
<a href="https://github.com/nari-labs/dia">
<img src="./dia/static/images/banner.png">
</a>
</p>
<p align="center">
<a href="https://tally.so/r/meokbo" target="_blank"><img alt="Static Badge" src="https://img.shields.io/badge/Join-Waitlist-white?style=for-the-badge"></a>
<a href="https://discord.gg/bJq6vjRRKv" target="_blank"><img src="https://img.shields.io/badge/Discord-Join%20Chat-7289DA?logo=discord&style=for-the-badge"></a>
<a href="https://github.com/nari-labs/dia/blob/main/LICENSE" target="_blank"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge" alt="LICENSE"></a>
</p>
<p align="center">
<a href="https://huggingface.co/nari-labs/Dia-1.6B"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-lg-dark.svg" alt="Dataset on HuggingFace" height=42 ></a>
<a href="https://huggingface.co/spaces/nari-labs/Dia-1.6B"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-lg-dark.svg" alt="Space on HuggingFace" height=38></a>
</p>

Dia is a 1.6B parameter text to speech model created by Nari Labs.

Dia **directly generates highly realistic dialogue from a transcript**. You can condition the output on audio, enabling emotion and tone control. The model can also produce nonverbal communications like laughter, coughing, clearing throat, etc.

To accelerate research, we are providing access to pretrained model checkpoints and inference code. The model weights are hosted on [Hugging Face](https://huggingface.co/nari-labs/Dia-1.6B). The model only supports English generation at the moment.

We also provide a [demo page](https://yummy-fir-7a4.notion.site/dia) comparing our model to [ElevenLabs Studio](https://elevenlabs.io/studio) and [Sesame CSM-1B](https://github.com/SesameAILabs/csm).

- (Update) We have a ZeroGPU Space running! Try it now [here](https://huggingface.co/spaces/nari-labs/Dia-1.6B). Thanks to the HF team for the support :)
- Join our [discord server](https://discord.gg/bJq6vjRRKv) for community support and access to new features.
- Play with a larger version of Dia: generate fun conversations, remix content, and share with friends. 🔮 Join the [waitlist](https://tally.so/r/meokbo) for early access.

## Generation Guidelines

- Keep input text length moderate 
    - Short input (corresponding to under 5s of audio) will sound unnatural
    - Very long input (corresponding to over 20s of audio) will make the speech unnaturally fast.
- Use non-verbal tags sparingly, from the list in the README. Overusing or using unlisted non-verbals may cause weird artifacts.
- Always begin input text with `[S1]`, and always alternate between `[S1]` and `[S2]` (i.e. `[S1]`... `[S1]`... is not good)
- When using audio prompts (voice cloning), follow these instructions carefully:
    - Provide the transcript of the to-be cloned audio before the generation text.
    - Transcript must use `[S1]`, `[S2]` speaker tags correctly (i.e. single speaker: `[S1]`..., two speakers: `[S1]`... `[S2]`...)
    - Duration of the to-be cloned audio should be 5~10 seconds for the best results.
        (Keep in mind: 1 second ≈ 86 tokens)
- Put `[S1]` or `[S2]` (the second-to-last speaker's tag) at the end of the audio to improve audio quality at the end

### Install via pip

```bash
# Install directly from GitHub
pip install git+https://github.com/nari-labs/dia.git
```

### Set HF_TOKEN ENV var

```bash
# Set the HF_TOKEN ENV var to auto download config from HF Hub
export HF_TOKEN="your token"
```

### Run the Gradio UI

This will open a Gradio UI that you can work on.

```bash
git clone https://github.com/nari-labs/dia.git
cd dia && uv run app.py
```

or if you do not have `uv` pre-installed:

```bash
git clone https://github.com/nari-labs/dia.git
cd dia
python -m venv .venv
source .venv/bin/activate
pip install -e .
python app.py
```

Note that the model was not fine-tuned on a specific voice. Hence, you will get different voices every time you run the model.
You can keep speaker consistency by either adding an audio prompt (a guide coming VERY soon - try it with the second example on Gradio for now), or fixing the seed.

## Features

- Generate dialogue via `[S1]` and `[S2]` tag
- Generate non-verbal like `(laughs)`, `(coughs)`, etc.
  - Below verbal tags will be recognized, but might result in unexpected output.
  - `(laughs), (clears throat), (sighs), (gasps), (coughs), (singing), (sings), (mumbles), (beep), (groans), (sniffs), (claps), (screams), (inhales), (exhales), (applause), (burps), (humming), (sneezes), (chuckle), (whistles)`
- Voice cloning. See [`example/voice_clone.py`](example/voice_clone.py) for more information.
  - In the Hugging Face space, you can upload the audio you want to clone and place its transcript before your script. Make sure the transcript follows the required format. The model will then output only the content of your script.

## Japanese Language Support (Experimental)

This model includes experimental support for processing Japanese text input. However, please note the significant limitations outlined below.

### Configuration

To enable Japanese language processing, you need to configure the model as follows:

1.  **Set Language in `DataConfig`**:
    *   In your `DataConfig` (part of `DiaConfig`), set the `language` field to `"ja"`.
    *   Example: `DataConfig(language="ja", tokenizer_path="/path/to/your/japanese.model", ...)`

2.  **Provide a SentencePiece Tokenizer Model**:
    *   You **must** provide your own SentencePiece model file (`.model`) trained specifically on Japanese text.
    *   Set the `tokenizer_path` field in `DataConfig` to the absolute or relative path of this `.model` file.

### Dependency

*   The `sentencepiece` Python library is required for Japanese tokenization. It has been added as a project dependency and should be installed automatically when you install or update the `nari-tts` package.

### Limitations and Important Notes

*   **Pre-trained Models:** The default pre-trained TTS model weights (e.g., `nari-labs/Dia-1.6B`) and the default Descript Audio Codec (DAC) model **are not trained or optimized for Japanese**.
*   **Output Quality:** Attempting to generate Japanese speech with the current default English-trained models will likely result in **suboptimal, incoherent, or non-Japanese audio**.
*   **Audio Prompts:** Using Japanese audio prompts with the default DAC model may also not work as expected due to the mismatch in training data.

For effective Japanese speech synthesis and audio prompt support with this codebase, you would generally need:

*   A `DiaModel` (or a model with a compatible architecture) that has been trained on a substantial dataset of Japanese speech.
*   A DAC model that has been trained or fine-tuned specifically on Japanese audio data.
*   Optionally, for robust Japanese audio prompt processing, integration with a Japanese Speech-to-Text (STT) model might be necessary to convert audio prompts into text that the Japanese TTS model can better understand.

### Command-Line (CLI) Usage

When using `cli.py`, you can specify the language using the `--language` flag:

```bash
python cli.py "こんにちは世界" --output japanese_output.wav --language ja --repo-id your/model/id
```

Ensure that the configuration file (`config.json`) associated with your model (whether local or downloaded via `repo-id`) has the `tokenizer_path` field in its `data` configuration correctly pointing to your Japanese SentencePiece model file. The `--language ja` flag will override the language setting in the config, but `tokenizer_path` must be pre-set in the config for Japanese.

---

## ⚙️ Usage

### As a Python Library

```python
from dia.model import Dia


model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16")

text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."

output = model.generate(text, use_torch_compile=True, verbose=True)

model.save_audio("simple.mp3", output)
```

If you're on Mac with Apple Silicon, you can use the following code to make it work. For MPS to work `use_torch_compile` must be set to `False`. As that feature isn't supported yet.

```python
from dia.model import Dia


model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16")

text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."

# It is important to set the `use_torch_compile` argument to `False` when using Dia on MacOS.
# This is because the `torch.compile` function is not supported on MacOS.
output = model.generate(text, use_torch_compile=False, verbose=True)

model.save_audio("simple.mp3", output)
```

A pypi package and a working CLI tool will be available soon.

## 💻 Hardware and Inference Speed

Dia has been tested on only GPUs (pytorch 2.0+, CUDA 12.6). CPU support is to be added soon.
The initial run will take longer as the Descript Audio Codec also needs to be downloaded.

These are the speed we benchmarked in RTX 4090.

| precision | realtime factor w/ compile | realtime factor w/o compile | VRAM |
|:-:|:-:|:-:|:-:|
| `bfloat16` | x2.1 | x1.5 | ~10GB |
| `float16` | x2.2 | x1.3 | ~10GB |
| `float32` | x1 | x0.9 | ~13GB |

We will be adding a quantized version in the future.

If you don't have hardware available or if you want to play with bigger versions of our models, join the waitlist [here](https://tally.so/r/meokbo).

## 🪪 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This project offers a high-fidelity speech generation model intended for research and educational use. The following uses are **strictly forbidden**:

- **Identity Misuse**: Do not produce audio resembling real individuals without permission.
- **Deceptive Content**: Do not use this model to generate misleading content (e.g. fake news)
- **Illegal or Malicious Use**: Do not use this model for activities that are illegal or intended to cause harm.

By using this model, you agree to uphold relevant legal standards and ethical responsibilities. We **are not responsible** for any misuse and firmly oppose any unethical usage of this technology.

## 🔭 TODO / Future Work

- Docker support for ARM architecture and MacOS.
- Optimize inference speed.
- Add quantization for memory efficiency.

## 🤝 Contributing

We are a tiny team of 1 full-time and 1 part-time research-engineers. We are extra-welcome to any contributions!
Join our [Discord Server](https://discord.gg/bJq6vjRRKv) for discussions.

## 🤗 Acknowledgements

- We thank the [Google TPU Research Cloud program](https://sites.research.google/trc/about/) for providing computation resources.
- Our work was heavily inspired by [SoundStorm](https://arxiv.org/abs/2305.09636), [Parakeet](https://jordandarefsky.com/blog/2024/parakeet/), and [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec).
- Hugging Face for providing the ZeroGPU Grant.
- "Nari" is a pure Korean word for lily.
- We thank Jason Y. for providing help with data filtering.


## ⭐ Star History

<a href="https://www.star-history.com/#nari-labs/dia&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=nari-labs/dia&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=nari-labs/dia&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=nari-labs/dia&type=Date" />
 </picture>
</a>
