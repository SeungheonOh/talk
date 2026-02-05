use std::time::Instant;

/// Application state machine for wake-word activated transcription.
#[derive(Debug)]
pub enum AppState {
    /// Low-power mode: only VAD running.
    Sleep,
    /// VAD detected speech; accumulating audio until the utterance ends, then check for wake word.
    WakeWordCheck {
        started_at: Instant,
        silence_frames: u32,
    },
    /// Full transcription mode.
    Active {
        activated_at: Instant,
        last_speech_at: Instant,
    },
}

impl AppState {
    pub fn wake_word_check() -> Self {
        AppState::WakeWordCheck {
            started_at: Instant::now(),
            silence_frames: 0,
        }
    }

    pub fn activate() -> Self {
        let now = Instant::now();
        AppState::Active {
            activated_at: now,
            last_speech_at: now,
        }
    }

    /// Update the last_speech_at timestamp. No-op if not in Active state.
    pub fn touch_speech(&mut self) {
        if let AppState::Active { last_speech_at, .. } = self {
            *last_speech_at = Instant::now();
        }
    }

    /// Returns true if in Active state and silence timeout has been exceeded.
    pub fn silence_timeout_exceeded(&self, timeout_secs: u64) -> bool {
        if let AppState::Active { last_speech_at, .. } = self {
            last_speech_at.elapsed().as_secs() >= timeout_secs
        } else {
            false
        }
    }
}

/// VAD speech debouncer. Requires N consecutive frames above threshold
/// before confirming speech onset.
pub struct SpeechDebouncer {
    threshold: f32,
    required_frames: u32,
    consecutive_count: u32,
}

impl SpeechDebouncer {
    pub fn new(threshold: f32, required_frames: u32) -> Self {
        Self {
            threshold,
            required_frames,
            consecutive_count: 0,
        }
    }

    /// Feed a VAD probability. Returns true when speech is confirmed
    /// (N consecutive frames above threshold).
    pub fn update(&mut self, prob: f32) -> bool {
        if prob >= self.threshold {
            self.consecutive_count += 1;
        } else {
            self.consecutive_count = 0;
        }
        self.consecutive_count >= self.required_frames
    }

    pub fn reset(&mut self) {
        self.consecutive_count = 0;
    }
}
