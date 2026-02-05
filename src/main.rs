mod audio;
mod buffer;
mod state;
mod transcriber;
mod vad;

use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use enigo::{Enigo, Keyboard, Settings};
use ringbuf::traits::*;

use buffer::{AudioBuffer, RollingBuffer};
use state::{AppState, SpeechDebouncer};
use transcriber::Transcriber;
use vad::Vad;

const VAD_MODEL_PATH: &str = "models/silero_vad.onnx";
const WHISPER_MODEL_PATHS: &[&str] = &[
    "models/ggml-base.en.bin",
    "models/ggml-large-v3-turbo.bin",
];

const VAD_THRESHOLD: f32 = 0.5;
const VAD_DEBOUNCE_FRAMES: u32 = 4; // ~128ms of consecutive speech
const WAKE_WORD_SILENCE_FRAMES: u32 = 10; // ~320ms silence = end of utterance
const WAKE_WORD_TIMEOUT_MS: u64 = 3000; // max time to wait for wake word utterance
const SILENCE_TIMEOUT_SECS: u64 = 30;
const WHISPER_THREADS: i32 = 4;

const ROLLING_BUFFER_SECS: f32 = 3.0;

// VAD-based chunking: transcribe after a pause or when max duration is hit
const UTTERANCE_SILENCE_FRAMES: u32 = 16; // ~512ms silence = utterance boundary
const MAX_UTTERANCE_SAMPLES: usize = 16000 * 15; // 15 sec failsafe for non-stop speech

fn main() -> Result<()> {
    // Graceful shutdown on Ctrl+C
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        eprintln!("\nShutting down...");
        r.store(false, Ordering::SeqCst);
    })
    .context("Failed to set Ctrl+C handler")?;

    // Load models
    eprintln!("Loading VAD model from '{}'...", VAD_MODEL_PATH);
    let mut vad = Vad::new(VAD_MODEL_PATH)?;
    eprintln!("VAD model loaded.");

    let whisper_model = WHISPER_MODEL_PATHS
        .iter()
        .find(|p| std::path::Path::new(p).exists())
        .context("No Whisper model found. Download one to models/ (e.g. ggml-base.en.bin)")?;
    eprintln!("Loading Whisper model from '{}' ...", whisper_model);
    let transcriber = Transcriber::new(whisper_model, WHISPER_THREADS)?;
    eprintln!("Whisper model loaded.");

    // Start audio capture
    let (_stream, mut consumer) = audio::build_input_stream()?;

    // Keyboard simulator
    let mut enigo = Enigo::new(&Settings::default()).context("Failed to init keyboard simulator")?;

    // State
    let mut app_state = AppState::Sleep;
    let mut debouncer = SpeechDebouncer::new(VAD_THRESHOLD, VAD_DEBOUNCE_FRAMES);
    let mut rolling = RollingBuffer::new(ROLLING_BUFFER_SECS);
    let mut audio_buf = AudioBuffer::new();
    let mut typed_any = false;
    let mut utterance_silence: u32 = 0; // consecutive silence frames in Active mode

    let window_size = vad.window_size();
    let mut vad_window = vec![0.0f32; window_size];

    eprintln!("\n\u{1f4a4} System ready — say \"voice\" to start, \"done\" to stop.\n");

    // Main processing loop
    while running.load(Ordering::SeqCst) {
        // Wait until a full VAD window is available before consuming
        if consumer.occupied_len() < window_size {
            thread::sleep(Duration::from_millis(5));
            continue;
        }
        consumer.pop_slice(&mut vad_window);

        // Always feed rolling buffer
        rolling.push(&vad_window);

        // Run VAD
        let prob = match vad.predict(&vad_window) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("VAD error: {:#}", e);
                continue;
            }
        };

        match app_state {
            AppState::Sleep => {
                if debouncer.update(prob) {
                    app_state = AppState::wake_word_check();
                    debouncer.reset();
                    eprintln!("\u{1f50d} Speech detected, waiting for utterance to finish...");
                }
            }

            AppState::WakeWordCheck { started_at, ref mut silence_frames } => {
                if prob < VAD_THRESHOLD {
                    *silence_frames += 1;
                } else {
                    *silence_frames = 0;
                }

                let utterance_ended = *silence_frames >= WAKE_WORD_SILENCE_FRAMES;
                let timed_out = started_at.elapsed() >= Duration::from_millis(WAKE_WORD_TIMEOUT_MS);

                if utterance_ended || timed_out {
                    let snapshot = rolling.snapshot();
                    match transcriber.check_wake_word(&snapshot) {
                        Ok(true) => {
                            app_state = AppState::activate();
                            audio_buf.clear();
                            vad.reset();
                            debouncer.reset();
                            typed_any = false;
                            utterance_silence = 0;
                            play_sound("Tink");
                            eprintln!("\n\u{1f3a4} [ACTIVATED] — Listening and transcribing...\n");
                        }
                        Ok(false) => {
                            app_state = AppState::Sleep;
                            debouncer.reset();
                            eprintln!("\u{1f4a4} Not a wake word, returning to sleep.\n");
                        }
                        Err(e) => {
                            eprintln!("Wake word check error: {:#}", e);
                            app_state = AppState::Sleep;
                            debouncer.reset();
                        }
                    }
                }
            }

            AppState::Active { .. } => {
                // Track speech activity for silence timeout
                if prob >= VAD_THRESHOLD {
                    app_state.touch_speech();
                    utterance_silence = 0;
                } else {
                    utterance_silence += 1;
                }

                // Accumulate audio
                audio_buf.push(&vad_window);

                // Check global silence timeout (no speech at all for 30s)
                if app_state.silence_timeout_exceeded(SILENCE_TIMEOUT_SECS) {
                    eprintln!(
                        "\n\u{23f8}\u{fe0f} [DEACTIVATED] — Silence timeout ({}s)\n",
                        SILENCE_TIMEOUT_SECS
                    );
                    transcribe_and_type(&transcriber, &mut audio_buf, &mut enigo, &mut typed_any);
                    play_sound("Funk");
                    deactivate(&mut app_state, &mut audio_buf, &mut vad, &mut debouncer);
                    utterance_silence = 0;
                    continue;
                }

                // Transcribe when utterance ends (pause detected) or max duration hit
                let pause_detected = utterance_silence >= UTTERANCE_SILENCE_FRAMES
                    && audio_buf.len() > 0;
                let max_duration = audio_buf.len() >= MAX_UTTERANCE_SAMPLES;

                if pause_detected || max_duration {
                    let chunk = audio_buf.take();
                    match transcriber.transcribe(&chunk) {
                        Ok(text) => {
                            let trimmed = text.trim();
                            if !trimmed.is_empty() {
                                if Transcriber::is_deactivation_command(trimmed) {
                                    eprintln!(
                                        "\n\u{23f8}\u{fe0f} [DEACTIVATED] — Command recognized\n"
                                    );
                                    play_sound("Funk");
                                    deactivate(
                                        &mut app_state,
                                        &mut audio_buf,
                                        &mut vad,
                                        &mut debouncer,
                                    );
                                    utterance_silence = 0;
                                    continue;
                                }

                                if typed_any {
                                    let _ = enigo.text(" ");
                                }
                                let _ = enigo.text(trimmed);
                                typed_any = true;
                            }
                        }
                        Err(e) => {
                            eprintln!("Transcription error: {:#}", e);
                        }
                    }
                    utterance_silence = 0;
                }
            }
        }
    }

    eprintln!("\n\u{1f44b} Goodbye!");
    Ok(())
}

/// Transcribe any remaining audio in the buffer and type it out.
fn transcribe_and_type(
    transcriber: &Transcriber,
    audio_buf: &mut AudioBuffer,
    enigo: &mut Enigo,
    typed_any: &mut bool,
) {
    if audio_buf.len() > 0 {
        let audio = audio_buf.take();
        match transcriber.transcribe(&audio) {
            Ok(text) => {
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    if *typed_any {
                        let _ = enigo.text(" ");
                    }
                    let _ = enigo.text(trimmed);
                    *typed_any = true;
                }
            }
            Err(e) => {
                eprintln!("Final transcription error: {:#}", e);
            }
        }
    }
}

/// Play a macOS system sound asynchronously.
fn play_sound(name: &str) {
    let path = format!("/System/Library/Sounds/{}.aiff", name);
    let _ = Command::new("afplay")
        .arg(&path)
        .spawn();
}

/// Transition back to sleep mode, resetting all buffers and state.
fn deactivate(
    app_state: &mut AppState,
    audio_buf: &mut AudioBuffer,
    vad: &mut Vad,
    debouncer: &mut SpeechDebouncer,
) {
    *app_state = AppState::Sleep;
    audio_buf.clear();
    vad.reset();
    debouncer.reset();
    eprintln!("\u{1f4a4} Listening for wake word...\n");
}
