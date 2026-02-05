use anyhow::{Context, Result};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::stt::SttEngine;

const WHISPER_THREADS: i32 = 4;

/// Whisper-based STT engine.
pub struct WhisperEngine {
    ctx: WhisperContext,
}

impl WhisperEngine {
    pub fn new(model_path: &str) -> Result<Self> {
        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.flash_attn(true);
        let ctx = WhisperContext::new_with_params(model_path, ctx_params)
            .with_context(|| format!("Failed to load Whisper model from '{}'", model_path))?;
        Ok(Self { ctx })
    }

    fn make_params(&self) -> FullParams<'_, '_> {
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(WHISPER_THREADS);
        params.set_language(Some("en"));
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_print_special(false);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);
        params
    }
}

impl SttEngine for WhisperEngine {
    fn transcribe(&self, audio: &[f32]) -> Result<String> {
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
}
