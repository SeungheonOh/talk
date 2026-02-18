use std::sync::Arc;

use anyhow::{Context, Result};

use crate::stt_parakeet::ParakeetEngine;
use crate::stt_whisper::WhisperEngine;
#[cfg(feature = "voxtral")]
use crate::stt_voxtral::VoxtralEngine;

/// Common interface for speech-to-text backends.
pub trait SttEngine: Send + Sync {
    /// Transcribe a complete audio segment (16kHz mono f32).
    fn transcribe(&self, audio: &[f32]) -> Result<String>;

    // ── Optional streaming support ───────────────────────────────────

    fn supports_streaming(&self) -> bool { false }

    /// Begin a new streaming session.
    fn stream_start(&self) -> Result<()> { Ok(()) }

    /// Feed a chunk of audio. Returns any new text produced.
    fn stream_feed(&self, _audio: &[f32]) -> Result<String> { Ok(String::new()) }

    /// Force the encoder to process buffered audio (call on silence).
    /// Returns any new text produced.
    fn stream_flush(&self) -> Result<String> { Ok(String::new()) }

    /// Signal end of audio. Returns any remaining text.
    fn stream_finish(&self) -> Result<String> { Ok(String::new()) }
}

const PARAKEET_DIR: &str = "models/parakeet-tdt";
const WHISPER_MODEL_PATHS: &[&str] = &[
    "models/ggml-base.en.bin",
    "models/ggml-large-v3-turbo.bin",
];
/// Create an STT engine by name, or auto-detect if None.
/// Default preference: whisper, then parakeet.
/// `quant` selects the Voxtral quantization variant (e.g. "q8", "f16").
pub fn create_engine(choice: Option<&str>, #[allow(unused)] quant: Option<&str>) -> Result<Arc<dyn SttEngine>> {
    match choice {
        Some("whisper") => load_whisper(),
        Some("parakeet") => load_parakeet(),
        #[cfg(feature = "voxtral")]
        Some("voxtral") => load_voxtral(quant),
        #[cfg(not(feature = "voxtral"))]
        Some("voxtral") => anyhow::bail!("Voxtral support not compiled. Rebuild with: cargo build --features voxtral"),
        Some(other) => anyhow::bail!("Unknown engine '{}'.", other),
        None => load_whisper().or_else(|_| load_parakeet()),
    }
}

fn load_whisper() -> Result<Arc<dyn SttEngine>> {
    let model = WHISPER_MODEL_PATHS
        .iter()
        .find(|p| std::path::Path::new(p).exists())
        .context("No Whisper model found in models/")?;
    eprintln!("Loading Whisper model from '{}' ...", model);
    let engine = WhisperEngine::new(model)
        .with_context(|| format!("Failed to load Whisper model from '{}'", model))?;
    eprintln!("Whisper model loaded.");
    Ok(Arc::new(engine))
}

fn load_parakeet() -> Result<Arc<dyn SttEngine>> {
    let dir = std::path::Path::new(PARAKEET_DIR);
    anyhow::ensure!(
        dir.join("encoder-model.onnx").exists()
            && dir.join("decoder_joint-model.onnx").exists()
            && dir.join("vocab.txt").exists(),
        "Parakeet TDT models not found in {}", PARAKEET_DIR
    );
    eprintln!("Loading Parakeet TDT from '{}'...", PARAKEET_DIR);
    let engine = ParakeetEngine::new(PARAKEET_DIR)
        .context("Failed to load Parakeet TDT engine")?;
    eprintln!("Parakeet TDT loaded.");
    Ok(Arc::new(engine))
}

#[cfg(feature = "voxtral")]
fn load_voxtral(quant: Option<&str>) -> Result<Arc<dyn SttEngine>> {
    let dir_name = match quant {
        Some(q) => format!("models/voxtral-{}", q),
        None => "models/voxtral".to_string(),
    };
    let dir = std::path::Path::new(&dir_name);
    anyhow::ensure!(
        dir.join("consolidated.safetensors").exists()
            && dir.join("tekken.json").exists()
            && dir.join("params.json").exists(),
        "Voxtral model not found in {} (need consolidated.safetensors, tekken.json, params.json)",
        dir_name
    );
    eprintln!("Loading Voxtral from '{}'...", dir_name);
    let engine = VoxtralEngine::new(&dir_name)
        .context("Failed to load Voxtral engine")?;
    eprintln!("Voxtral loaded.");
    Ok(Arc::new(engine))
}
