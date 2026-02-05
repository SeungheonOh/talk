use anyhow::Result;

use crate::buffer::RollingBuffer;

/// Result from a wake detector's per-frame check.
pub enum WakeResult {
    /// No activation.
    Nothing,
    /// Wake event detected. String describes what triggered it.
    Activated(String),
}

/// Common interface for wake detection backends.
/// Each implementation manages its own internal state machine.
pub trait WakeDetector {
    /// Feed one VAD-sized audio window. Called every frame during Sleep.
    fn feed(&mut self, window: &[f32], vad_prob: f32, rolling: &RollingBuffer) -> Result<WakeResult>;
    /// Reset internal state (called on deactivation).
    fn reset(&mut self);
}
