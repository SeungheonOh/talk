# voicer

Voice-to-text dictation tool for macOS. Listens for a wake word, transcribes speech, and types it into the focused application.

## How it works

1. Sits idle, listening for the wake word ("clanker mic")
2. On activation, transcribes everything you say and types it in real time
3. Say "done" or "stop" to deactivate
4. Deactivates automatically after 30 seconds of silence

If a llama.cpp server is running on `localhost:8080`, transcribed text is also sent through an LLM for punctuation/capitalization cleanup.

## Requirements

- macOS (uses CoreAudio via cpal, afplay for sounds, enigo for keyboard simulation)
- Rust (edition 2024)
- Accessibility permissions for the terminal app (System Settings > Privacy & Security > Accessibility)

## Models

Create a `models/` directory in the project root. You need the VAD model plus at least one STT engine.

### VAD (required)

Download `silero_vad.onnx` from https://github.com/snakers4/silero-vad/tree/master/files and place it at:

```
models/silero_vad.onnx
```

### STT engine (pick one)

#### Whisper (default)

Download a ggml Whisper model. The code looks for these paths in order:

```
models/ggml-base.en.bin
models/ggml-large-v3-turbo.bin
```

Get them from https://huggingface.co/ggerganov/whisper.cpp/tree/main

#### Parakeet TDT

Export the NVIDIA Parakeet TDT 1.1B model to ONNX. The code expects:

```
models/parakeet-tdt/encoder-model.onnx
models/parakeet-tdt/decoder_joint-model.onnx
models/parakeet-tdt/vocab.txt
```

### LLM post-processing (optional)

Run a llama.cpp server on port 8080 with any instruction-following model. If unreachable, voicer runs without post-processing.

## Build and run

```
cargo build --release
./target/release/voicer
```

Select STT engine explicitly:

```
./target/release/voicer --engine whisper
./target/release/voicer --engine parakeet
```
