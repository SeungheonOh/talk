use anyhow::{Context, Result};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Wake word phrases to detect (lowercase).
const WAKE_PHRASES: &[&str] = &["voice"];

/// Commands that trigger deactivation (lowercase).
const DEACTIVATION_COMMANDS: &[&str] = &["done"];

/// Whisper-based speech transcription engine.
pub struct Transcriber {
    ctx: WhisperContext,
    n_threads: i32,
}

impl Transcriber {
    /// Load a Whisper model. This is expensive (~2-5s for large models). Do once at startup.
    pub fn new(model_path: &str, n_threads: i32) -> Result<Self> {
        let ctx = WhisperContext::new_with_params(model_path, WhisperContextParameters::default())
            .with_context(|| format!("Failed to load Whisper model from '{}'", model_path))?;

        Ok(Self { ctx, n_threads })
    }

    /// Create params configured for fast English transcription.
    fn make_params(&self) -> FullParams<'_, '_> {
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(self.n_threads);
        params.set_language(Some("en"));
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_print_special(false);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);
        params
    }

    /// Run transcription on audio and return all segment texts joined.
    fn run_transcription(&self, audio: &[f32]) -> Result<String> {
        let mut state = self
            .ctx
            .create_state()
            .context("Failed to create Whisper state")?;

        let params = self.make_params();
        state
            .full(params, audio)
            .context("Whisper transcription failed")?;

        let n_segments = state.full_n_segments();

        let mut text = String::new();
        for i in 0..n_segments {
            if let Some(segment) = state.get_segment(i) {
                if let Ok(segment_text) = segment.to_str() {
                    text.push_str(segment_text);
                }
            }
        }

        Ok(text)
    }

    /// Check if the 3-second audio buffer contains a wake word.
    /// Returns true if a wake phrase is detected.
    pub fn check_wake_word(&self, audio: &[f32]) -> Result<bool> {
        let text = self.run_transcription(audio)?;
        let lower = text.to_lowercase();

        let found = WAKE_PHRASES.iter().any(|phrase| lower.contains(phrase));

        Ok(found)
    }

    /// Transcribe an audio chunk and return the text.
    pub fn transcribe(&self, audio: &[f32]) -> Result<String> {
        self.run_transcription(audio)
    }

    /// Check if transcribed text contains a deactivation command.
    pub fn is_deactivation_command(text: &str) -> bool {
        let lower = text.to_lowercase();
        DEACTIVATION_COMMANDS
            .iter()
            .any(|cmd| lower.contains(cmd))
    }

}
