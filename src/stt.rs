use std::sync::Arc;

use anyhow::{Context, Result};

use crate::stt_parakeet::ParakeetEngine;
use crate::stt_whisper::WhisperEngine;

/// Common interface for speech-to-text backends.
pub trait SttEngine: Send + Sync {
    /// Transcribe a complete audio segment (16kHz mono f32).
    fn transcribe(&self, audio: &[f32]) -> Result<String>;
}

const PARAKEET_DIR: &str = "models/parakeet-tdt";
const WHISPER_MODEL_PATHS: &[&str] = &[
    "models/ggml-base.en.bin",
    "models/ggml-large-v3-turbo.bin",
];

/// Create an STT engine by name, or auto-detect if None.
/// Default preference: whisper, then parakeet.
pub fn create_engine(choice: Option<&str>) -> Result<Arc<dyn SttEngine>> {
    match choice {
        Some("whisper") => load_whisper(),
        Some("parakeet") => load_parakeet(),
        Some(other) => anyhow::bail!("Unknown engine '{}'. Use 'whisper' or 'parakeet'.", other),
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
