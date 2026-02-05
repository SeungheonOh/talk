use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;

use crate::buffer::RollingBuffer;
use crate::stt::SttEngine;
use crate::wake::{WakeDetector, WakeResult};

/// Wake word phrases to detect (lowercase).
const WAKE_PHRASES: &[&str] = &["clanker mic", "voice", "speech"];

const VAD_THRESHOLD: f32 = 0.5;
const DEBOUNCE_FRAMES: u32 = 4; // ~128ms of consecutive speech
const SILENCE_FRAMES: u32 = 10; // ~320ms silence = end of utterance
const TIMEOUT_MS: u64 = 3000; // max time to wait for wake word utterance

enum Phase {
    /// Waiting for speech onset via VAD debouncing.
    Idle,
    /// Speech detected; accumulating until utterance ends.
    Accumulating {
        started_at: Instant,
        silence_frames: u32,
    },
}

pub struct WakeWordDetector {
    engine: Arc<dyn SttEngine>,
    phase: Phase,
    consecutive_speech: u32,
}

impl WakeWordDetector {
    pub fn new(engine: Arc<dyn SttEngine>) -> Self {
        Self {
            engine,
            phase: Phase::Idle,
            consecutive_speech: 0,
        }
    }
}

impl WakeDetector for WakeWordDetector {
    fn feed(&mut self, _window: &[f32], vad_prob: f32, rolling: &RollingBuffer) -> Result<WakeResult> {
        match self.phase {
            Phase::Idle => {
                if vad_prob >= VAD_THRESHOLD {
                    self.consecutive_speech += 1;
                } else {
                    self.consecutive_speech = 0;
                }

                if self.consecutive_speech >= DEBOUNCE_FRAMES {
                    self.consecutive_speech = 0;
                    self.phase = Phase::Accumulating {
                        started_at: Instant::now(),
                        silence_frames: 0,
                    };
                    eprintln!("\u{1f50d} Speech detected, waiting for utterance to finish...");
                }

                Ok(WakeResult::Nothing)
            }

            Phase::Accumulating {
                started_at,
                ref mut silence_frames,
            } => {
                if vad_prob < VAD_THRESHOLD {
                    *silence_frames += 1;
                } else {
                    *silence_frames = 0;
                }

                let utterance_ended = *silence_frames >= SILENCE_FRAMES;
                let timed_out = started_at.elapsed() >= Duration::from_millis(TIMEOUT_MS);

                if !utterance_ended && !timed_out {
                    return Ok(WakeResult::Nothing);
                }

                // Utterance ended or timed out — check for wake word
                let snapshot = rolling.snapshot();
                let text = self.engine.transcribe(&snapshot)?;
                let lower = text.to_lowercase();
                eprintln!("  [wake check] heard: {:?}", lower.trim());

                self.phase = Phase::Idle;
                self.consecutive_speech = 0;

                if WAKE_PHRASES.iter().any(|phrase| lower.contains(phrase)) {
                    Ok(WakeResult::Activated(format!("Wake word: {}", lower.trim())))
                } else {
                    eprintln!("\u{1f4a4} Not a wake word, returning to sleep.\n");
                    Ok(WakeResult::Nothing)
                }
            }
        }
    }

    fn reset(&mut self) {
        self.phase = Phase::Idle;
        self.consecutive_speech = 0;
    }
}
