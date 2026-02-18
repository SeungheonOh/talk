use std::sync::Arc;

use anyhow::Result;

use crate::stt::SttEngine;

/// Voice commands triggered by specific words at the end of an utterance.
#[derive(Debug, Clone, PartialEq)]
pub enum VoiceCommand {
    /// Press Enter after committing text.
    Presto,
    /// Discard all text and deactivate.
    Disco,
    /// Stop transcription, keep text as-is.
    Apex,
}

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

    /// Check if the last word of the utterance is a voice command.
    /// Returns the command (if any) and the text with the command word removed.
    pub fn check_voice_command(text: &str) -> (Option<VoiceCommand>, String) {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return (None, String::new());
        }

        let last_word_raw = trimmed.split_whitespace().last().unwrap_or("");
        let last_word = last_word_raw
            .trim_end_matches(|c: char| c.is_ascii_punctuation())
            .to_lowercase();

        let cmd = match last_word.as_str() {
            "presto" => Some(VoiceCommand::Presto),
            "disco" => Some(VoiceCommand::Disco),
            "apex" => Some(VoiceCommand::Apex),
            _ => None,
        };

        if cmd.is_some() {
            let without = trimmed
                .rfind(last_word_raw)
                .map(|pos| trimmed[..pos].trim_end().to_string())
                .unwrap_or_default();
            (cmd, without)
        } else {
            (None, trimmed.to_string())
        }
    }
}
