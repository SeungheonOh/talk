use anyhow::Result;

use crate::stt::SttEngine;

/// Wake word phrases to detect (lowercase).
const WAKE_PHRASES: &[&str] = &["clanker mic", "voice", "speech"];

/// Commands that trigger deactivation (lowercase).
const DEACTIVATION_COMMANDS: &[&str] = &["done", "stop"];

/// Speech transcription manager that wraps any STT backend.
/// Handles wake word detection and deactivation commands.
pub struct Transcriber {
    engine: Box<dyn SttEngine>,
}

impl Transcriber {
    pub fn new(engine: Box<dyn SttEngine>) -> Self {
        Self { engine }
    }

    /// Check if the audio buffer contains a wake word.
    pub fn check_wake_word(&self, audio: &[f32]) -> Result<bool> {
        let text = self.engine.transcribe(audio)?;
        let lower = text.to_lowercase();
        eprintln!("  [wake check] heard: {:?}", lower.trim());
        Ok(WAKE_PHRASES.iter().any(|phrase| lower.contains(phrase)))
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
