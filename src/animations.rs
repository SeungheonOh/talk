use std::time::{Duration, Instant};

const FADE_IN_MS: u64 = 350;
const FADE_OUT_MS: u64 = 250;
const SLIDE_DISTANCE: f32 = 10.0;

/// Ease-out quartic: snappy entrance that decelerates smoothly.
fn ease_out(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    1.0 - (1.0 - t).powi(4)
}

/// Ease-in cubic: gentle start, accelerates to finish.
fn ease_in(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t.powi(3)
}

/// Animation phase for the overlay visibility.
#[derive(Debug, Clone)]
pub enum AnimationPhase {
    Hidden,
    FadingIn { start: Instant },
    Visible,
    FadingOut { start: Instant },
}

impl AnimationPhase {
    pub fn opacity(&self) -> f32 {
        match self {
            Self::Hidden => 0.0,
            Self::Visible => 1.0,
            Self::FadingIn { start } => {
                let t = start.elapsed().as_millis() as f32 / FADE_IN_MS as f32;
                ease_out(t)
            }
            Self::FadingOut { start } => {
                let t = start.elapsed().as_millis() as f32 / FADE_OUT_MS as f32;
                1.0 - ease_in(t)
            }
        }
    }

    /// Vertical offset from target position. Positive = below target.
    /// Slides up during fade-in, slides down during fade-out.
    pub fn slide_offset(&self) -> f32 {
        match self {
            Self::Hidden => SLIDE_DISTANCE,
            Self::Visible => 0.0,
            Self::FadingIn { .. } => SLIDE_DISTANCE * (1.0 - self.opacity()),
            Self::FadingOut { .. } => SLIDE_DISTANCE * (1.0 - self.opacity()),
        }
    }

    pub fn is_animating(&self) -> bool {
        matches!(self, Self::FadingIn { .. } | Self::FadingOut { .. })
    }

    pub fn is_visible(&self) -> bool {
        !matches!(self, Self::Hidden)
    }

    pub fn tick(&mut self) -> bool {
        match self {
            Self::FadingIn { start } => {
                if start.elapsed() >= Duration::from_millis(FADE_IN_MS) {
                    *self = Self::Visible;
                    return true;
                }
            }
            Self::FadingOut { start } => {
                if start.elapsed() >= Duration::from_millis(FADE_OUT_MS) {
                    *self = Self::Hidden;
                    return true;
                }
            }
            _ => {}
        }
        false
    }

    pub fn show(&mut self) {
        match self {
            Self::Hidden | Self::FadingOut { .. } => {
                *self = Self::FadingIn {
                    start: Instant::now(),
                };
            }
            _ => {}
        }
    }

    pub fn hide(&mut self) {
        match self {
            Self::Visible | Self::FadingIn { .. } => {
                *self = Self::FadingOut {
                    start: Instant::now(),
                };
            }
            _ => {}
        }
    }
}

/// Pulsing animation for the listening indicator dot.
#[derive(Debug, Clone)]
pub struct PulseAnimation {
    start: Instant,
}

impl PulseAnimation {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Returns opacity in [0.5, 1.0] on a gentle 2-second sine cycle.
    pub fn opacity(&self) -> f32 {
        let t = self.start.elapsed().as_secs_f32();
        let cycle = (t * std::f32::consts::TAU / 2.0).sin();
        0.75 + 0.25 * cycle
    }

    pub fn restart(&mut self) {
        self.start = Instant::now();
    }
}
