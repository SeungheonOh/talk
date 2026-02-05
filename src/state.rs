use std::time::Instant;

/// Application state machine.
#[derive(Debug)]
pub enum AppState {
    /// Low-power mode: wake detector running.
    Sleep,
    /// Full transcription mode.
    Active {
        activated_at: Instant,
        last_speech_at: Instant,
    },
}

impl AppState {
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
