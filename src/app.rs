use std::hash::{Hash, Hasher};
use std::pin::Pin;
use std::sync::mpsc;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use enigo::{Direction, Enigo, Key, Keyboard, Settings};
use iced::widget::container;
use iced::window;
use iced::{Element, Size, Subscription, Task, Theme};

use crate::animations::{AnimationPhase, PulseAnimation};
use crate::overlay_ui;
use crate::processing::VoiceEvent;

/// Visual mode of the overlay.
#[derive(Debug, Clone, PartialEq)]
enum VisualState {
    Idle,
    Listening,
    Streaming,
}

/// Messages handled by the iced runtime.
#[derive(Debug, Clone)]
pub enum Message {
    VoiceEvent(VoiceEvent),
    AnimationTick,
    MoveWindow(window::Id, f32, f32),
}

/// Wrapper for the voice receiver that implements Hash for subscription identity.
#[derive(Clone)]
struct VoiceRx(Arc<Mutex<mpsc::Receiver<VoiceEvent>>>);

impl Hash for VoiceRx {
    fn hash<H: Hasher>(&self, state: &mut H) {
        "voice_event_subscription".hash(state);
    }
}

/// Per-word staggered fade-in timing.
const WORD_STAGGER_MS: f32 = 20.0;
const WORD_FADE_MS: f32 = 120.0;
const MAX_STAGGER_DELAY_MS: f32 = 150.0;

pub struct VoicerApp {
    rx: VoiceRx,
    #[allow(dead_code)]
    running: Arc<AtomicBool>,
    phase: AnimationPhase,
    pulse: PulseAnimation,
    visual_state: VisualState,
    enigo: Option<Enigo>,
    overlay_pos: (f32, f32),

    /// All raw utterances in order.
    all_raw: Vec<String>,
    /// Latest LLM correction and how many raw items it covers.
    last_corrected: Option<String>,
    corrected_raw_count: usize,
    /// Current in-progress streaming text (not yet finalized).
    stream_text: String,
    /// Per-word animation: (change_base_time, stagger_index).
    word_anims: Vec<(Instant, u32)>,
    /// Previous display words for diffing.
    prev_words: Vec<String>,
    /// Use dark overlay (detected from background brightness).
    use_dark_overlay: bool,
}

impl VoicerApp {
    pub fn new(
        rx: mpsc::Receiver<VoiceEvent>,
        running: Arc<AtomicBool>,
        overlay_pos: (f32, f32),
    ) -> (Self, Task<Message>) {
        let enigo = Enigo::new(&Settings {
            mac_delay: 1,
            ..Settings::default()
        })
        .ok();

        let app = Self {
            rx: VoiceRx(Arc::new(Mutex::new(rx))),
            running,
            phase: AnimationPhase::Hidden,
            pulse: PulseAnimation::new(),
            visual_state: VisualState::Idle,
            enigo,
            overlay_pos,
            all_raw: Vec::new(),
            last_corrected: None,
            corrected_raw_count: 0,
            stream_text: String::new(),
            word_anims: Vec::new(),
            prev_words: Vec::new(),
            use_dark_overlay: false,
        };

        (app, Task::none())
    }
}

/// Build the overlay display text.
/// Shows corrected text + any uncovered raw utterances + in-progress stream.
fn build_display(
    all_raw: &[String],
    last_corrected: &Option<String>,
    corrected_count: usize,
    stream: &str,
) -> String {
    let base = match last_corrected {
        Some(c) => {
            let uncovered = &all_raw[corrected_count.min(all_raw.len())..];
            if uncovered.is_empty() {
                c.clone()
            } else {
                format!("{} {}", c, uncovered.join(" "))
            }
        }
        None => all_raw.join(" "),
    };
    if stream.is_empty() {
        base
    } else if base.is_empty() {
        stream.to_string()
    } else {
        format!("{} {}", base, stream)
    }
}

fn ease_out_quad(t: f32) -> f32 {
    1.0 - (1.0 - t) * (1.0 - t)
}

fn compute_word_opacity(base_time: &Instant, stagger_idx: u32) -> f32 {
    let elapsed = base_time.elapsed().as_millis() as f32;
    let delay = (stagger_idx as f32 * WORD_STAGGER_MS).min(MAX_STAGGER_DELAY_MS);
    let t = ((elapsed - delay).max(0.0) / WORD_FADE_MS).min(1.0);
    ease_out_quad(t)
}

/// Diff current display words against previous and update per-word timestamps.
fn sync_word_animations(app: &mut VoicerApp) {
    let display = build_display(
        &app.all_raw,
        &app.last_corrected,
        app.corrected_raw_count,
        &app.stream_text,
    );

    let new_words: Vec<String> = display
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();

    let mut new_anims: Vec<(Instant, u32)> = Vec::with_capacity(new_words.len());
    let mut stagger = 0u32;
    let now = Instant::now();

    for (i, word) in new_words.iter().enumerate() {
        let is_same = i < app.prev_words.len() && {
            let prev = &app.prev_words[i];
            *prev == *word
                || word.starts_with(prev.as_str())
                || prev.starts_with(word.as_str())
        };

        if is_same {
            new_anims.push(app.word_anims[i]);
        } else {
            new_anims.push((now, stagger));
            stagger += 1;
        }
    }

    app.prev_words = new_words;
    app.word_anims = new_anims;
}

fn move_window_to(x: f32, y: f32) -> Task<Message> {
    window::latest().then(move |id_opt| {
        if let Some(id) = id_opt {
            Task::done(Message::MoveWindow(id, x, y))
        } else {
            Task::none()
        }
    })
}

fn reset_state(app: &mut VoicerApp) {
    app.all_raw.clear();
    app.last_corrected = None;
    app.corrected_raw_count = 0;
    app.stream_text.clear();
    app.word_anims.clear();
    app.prev_words.clear();
    app.phase.hide();
    app.visual_state = VisualState::Idle;
    crate::panel::set_tray_capturing(false);
}

pub fn update(app: &mut VoicerApp, message: Message) -> Task<Message> {
    match message {
        Message::MoveWindow(id, x, y) => window::move_to(id, iced::Point::new(x, y)),

        Message::VoiceEvent(event) => match event {
            VoiceEvent::Activated => {
                let (x, y) = app.overlay_pos;
                let brightness = crate::panel::sample_background_brightness(x, y, 500.0, 50.0);
                app.use_dark_overlay = brightness > 0.55;
                app.phase.show();
                app.pulse.restart();
                app.visual_state = VisualState::Listening;
                app.all_raw.clear();
                app.last_corrected = None;
                app.corrected_raw_count = 0;
                app.stream_text.clear();
                app.word_anims.clear();
                app.prev_words.clear();
                crate::panel::set_tray_capturing(true);
                let offset = app.phase.slide_offset();
                move_window_to(x, y + offset)
            }

            VoiceEvent::StreamUpdate(text) => {
                app.stream_text = text;
                sync_word_animations(app);
                app.visual_state = if app.prev_words.is_empty() {
                    VisualState::Listening
                } else {
                    VisualState::Streaming
                };
                Task::none()
            }

            VoiceEvent::UtteranceFinalized(text) => {
                app.all_raw.push(text);
                app.stream_text.clear();
                sync_word_animations(app);
                app.visual_state = VisualState::Streaming;
                Task::none()
            }

            VoiceEvent::LlmCorrected(corrected, raw_count) => {
                app.last_corrected = Some(corrected);
                app.corrected_raw_count = raw_count;
                sync_word_animations(app);
                Task::none()
            }

            VoiceEvent::Deactivated => {
                let full = build_display(&app.all_raw, &app.last_corrected, app.corrected_raw_count, "");
                if !full.is_empty() {
                    if let Some(e) = &mut app.enigo {
                        if let Err(err) = e.text(&full) {
                            eprintln!("enigo.text error: {}", err);
                        }
                    }
                }
                reset_state(app);
                Task::none()
            }

            VoiceEvent::DeactivatedEnter => {
                let full = build_display(&app.all_raw, &app.last_corrected, app.corrected_raw_count, "");
                if let Some(e) = &mut app.enigo {
                    if !full.is_empty() {
                        if let Err(err) = e.text(&full) {
                            eprintln!("enigo.text error: {}", err);
                        }
                    }
                    if let Err(err) = e.key(Key::Return, Direction::Click) {
                        eprintln!("enigo.key error: {}", err);
                    }
                }
                reset_state(app);
                Task::none()
            }

            VoiceEvent::DeactivatedDiscard => {
                reset_state(app);
                Task::none()
            }
        },

        Message::AnimationTick => {
            app.phase.tick();
            if !app.phase.is_visible() {
                return move_window_to(-1000.0, -1000.0);
            }
            if app.phase.is_animating() {
                let (x, y) = app.overlay_pos;
                let offset = app.phase.slide_offset();
                return move_window_to(x, y + offset);
            }
            Task::none()
        }
    }
}

pub fn view(app: &VoicerApp) -> Element<'_, Message> {
    let opacity = app.phase.opacity();

    if opacity < 0.01 {
        return container("").width(0).height(0).into();
    }

    let words_with_opacity: Vec<(String, f32)> = app
        .prev_words
        .iter()
        .zip(app.word_anims.iter())
        .map(|(word, (base_time, stagger_idx))| {
            (word.clone(), compute_word_opacity(base_time, *stagger_idx))
        })
        .collect();

    overlay_ui::overlay_pill::<Message>(&words_with_opacity, &app.pulse, opacity, app.use_dark_overlay)
}

/// Build a stream that polls the voice event receiver.
fn build_voice_stream(
    data: &VoiceRx,
) -> Pin<Box<dyn iced::futures::Stream<Item = VoiceEvent> + Send>> {
    let rx = data.0.clone();
    Box::pin(iced::stream::channel::<VoiceEvent>(
        64,
        move |mut sender: iced::futures::channel::mpsc::Sender<VoiceEvent>| async move {
            loop {
                let event = { rx.lock().unwrap().try_recv().ok() };
                if let Some(event) = event {
                    if sender.try_send(event).is_err() {
                        break;
                    }
                } else {
                    tokio::time::sleep(Duration::from_millis(8)).await;
                }
            }
        },
    ))
}

pub fn subscription(app: &VoicerApp) -> Subscription<Message> {
    let mut subs = Vec::new();

    // Voice event channel
    subs.push(
        Subscription::run_with(app.rx.clone(), build_voice_stream).map(Message::VoiceEvent),
    );

    let needs_tick = app.phase.is_animating()
        || app.visual_state == VisualState::Listening
        || app.visual_state == VisualState::Streaming;
    if needs_tick {
        subs.push(iced::time::every(Duration::from_millis(16)).map(|_| Message::AnimationTick));
    }

    Subscription::batch(subs)
}

pub fn theme(_app: &VoicerApp) -> Theme {
    Theme::Dark
}

pub fn window_settings() -> window::Settings {
    window::Settings {
        size: Size::new(600.0, 300.0),
        position: window::Position::Specific(iced::Point::new(-1000.0, -1000.0)),
        transparent: true,
        decorations: false,
        level: window::Level::AlwaysOnTop,
        resizable: false,
        #[cfg(target_os = "macos")]
        platform_specific: iced::window::settings::PlatformSpecific {
            title_hidden: true,
            titlebar_transparent: true,
            fullsize_content_view: true,
        },
        ..window::Settings::default()
    }
}
