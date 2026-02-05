use std::sync::Arc;

use anyhow::Result;

use crate::stt::SttEngine;

/// Commands that trigger deactivation (lowercase).
const DEACTIVATION_COMMANDS: &[&str] = &["done", "stop"];

/// Speech transcription manager that wraps any STT backend.
pub struct Transcriber {
    engine: Arc<dyn SttEngine>,
}

impl Transcriber {
    pub fn new(engine: Arc<dyn SttEngine>) -> Self {
        Self { engine }
    }

    /// Transcribe an audio chunk and return the text.
    pub fn transcribe(&self, audio: &[f32]) -> Result<String> {
        self.engine.transcribe(audio)
    }

    /// Check if transcribed text contains a deactivation command.
    pub fn is_deactivation_command(text: &str) -> bool {
        let lower = text.to_lowercase();
        DEACTIVATION_COMMANDS
            .iter()
            .any(|cmd| lower.contains(cmd))
    }
}
