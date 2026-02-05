use std::collections::VecDeque;

/// Rolling audio buffer that keeps the last N seconds of audio.
/// Used for wake word detection — we always have recent audio available.
pub struct RollingBuffer {
    buf: VecDeque<f32>,
    capacity: usize,
}

impl RollingBuffer {
    /// Create a rolling buffer that holds `duration_secs` seconds of 16kHz audio.
    pub fn new(duration_secs: f32) -> Self {
        let capacity = (16000.0 * duration_secs) as usize;
        Self {
            buf: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Push new samples. Oldest samples are dropped if buffer exceeds capacity.
    pub fn push(&mut self, samples: &[f32]) {
        self.buf.extend(samples.iter());
        while self.buf.len() > self.capacity {
            self.buf.pop_front();
        }
    }

    /// Snapshot the entire buffer as a contiguous Vec.
    pub fn snapshot(&self) -> Vec<f32> {
        self.buf.iter().copied().collect()
    }
}

/// Simple audio accumulation buffer for active transcription mode.
/// Audio is pushed in, then taken all at once when an utterance ends.
pub struct AudioBuffer {
    buf: Vec<f32>,
}

impl AudioBuffer {
    pub fn new() -> Self {
        Self {
            buf: Vec::with_capacity(16000 * 15),
        }
    }

    pub fn push(&mut self, samples: &[f32]) {
        self.buf.extend_from_slice(samples);
    }

    /// Take all accumulated audio, leaving the buffer empty.
    pub fn take(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.buf)
    }

    pub fn clear(&mut self) {
        self.buf.clear();
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }
}
