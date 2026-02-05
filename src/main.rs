mod audio;
mod buffer;
mod llm;
mod state;
mod stt;
mod stt_parakeet;
mod stt_whisper;
mod transcriber;
mod vad;

use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use enigo::{Direction, Enigo, Key, Keyboard, Settings};
use ringbuf::traits::*;

use buffer::{AudioBuffer, RollingBuffer};
use state::{AppState, SpeechDebouncer};
use transcriber::Transcriber;
use vad::Vad;

const VAD_MODEL_PATH: &str = "models/silero_vad.onnx";

const VAD_THRESHOLD: f32 = 0.5;
const VAD_DEBOUNCE_FRAMES: u32 = 4; // ~128ms of consecutive speech
const WAKE_WORD_SILENCE_FRAMES: u32 = 10; // ~320ms silence = end of utterance
const WAKE_WORD_TIMEOUT_MS: u64 = 3000; // max time to wait for wake word utterance
const SILENCE_TIMEOUT_SECS: u64 = 30;

const ROLLING_BUFFER_SECS: f32 = 3.0;

// VAD-based chunking: transcribe after a pause or when max duration is hit
const UTTERANCE_SILENCE_FRAMES: u32 = 10; // ~320ms silence = utterance boundary
const MAX_UTTERANCE_SAMPLES: usize = 16000 * 15; // 15 sec failsafe for non-stop speech

/// Return the last `n` chars of `s`, or all of `s` if shorter.
fn tail_chars(s: &str, n: usize) -> &str {
    let char_count = s.chars().count();
    if char_count <= n {
        return s;
    }
    let skip = char_count - n;
    let byte_offset: usize = s.chars().take(skip).map(|c| c.len_utf8()).sum();
    &s[byte_offset..]
}

fn build_display(finalized: &str, pending: &str) -> String {
    if finalized.is_empty() {
        pending.to_string()
    } else if pending.is_empty() {
        finalized.to_string()
    } else {
        format!("{} {}", finalized, pending)
    }
}

/// Build the pending portion of the display, handling partial LLM coverage.
fn build_pending_display(
    pending_raw: &[String],
    pending_corrected: &Option<String>,
    corrected_count: usize,
) -> String {
    match pending_corrected {
        Some(c) if corrected_count < pending_raw.len() => {
            let uncovered = pending_raw[corrected_count..].join(" ");
            format!("{} {}", c, uncovered)
        }
        Some(c) => c.clone(),
        None => pending_raw.join(" "),
    }
}

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

    let engine_name = std::env::args().nth(1).and_then(|arg| {
        if arg == "--engine" {
            std::env::args().nth(2)
        } else if let Some(val) = arg.strip_prefix("--engine=") {
            Some(val.to_string())
        } else {
            None
        }
    });
    let engine = stt::create_engine(engine_name.as_deref())?;
    let transcriber = Transcriber::new(engine);

    // Start audio capture
    let (_stream, mut consumer) = audio::build_input_stream()?;

    // Keyboard simulator (mac_delay=0: fast_text handles its own delay for typing,
    // and we don't want 20ms×2 per backspace during retype)
    let mut enigo = Enigo::new(&Settings {
        mac_delay: 0,
        ..Settings::default()
    }).context("Failed to init keyboard simulator")?;

    // State
    let mut app_state = AppState::Sleep;
    let mut debouncer = SpeechDebouncer::new(VAD_THRESHOLD, VAD_DEBOUNCE_FRAMES);
    let mut rolling = RollingBuffer::new(ROLLING_BUFFER_SECS);
    let mut audio_buf = AudioBuffer::new();
    let mut typed_any = false;
    let mut utterance_silence: u32 = 0; // consecutive silence frames in Active mode
    let mut chunk_has_speech = false; // whether current chunk contains any speech

    // LLM post-processing (optional)
    let llm = llm::LlmHandle::spawn("http://localhost:8080");
    let mut finalized = String::new();
    let mut pending_raw: Vec<String> = Vec::new();
    let mut pending_corrected: Option<String> = None;
    let mut pending_corrected_count: usize = 0;
    let mut last_sent_count: usize = 0;
    let mut displayed_text = String::new();
    let mut llm_seq: u64 = 0;

    let window_size = vad.window_size();
    let mut vad_window = vec![0.0f32; window_size];

    eprintln!("\n\u{1f4a4} System ready — say \"clanker mic\" to start, \"done\" to stop.\n");

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
                            chunk_has_speech = false;
                            finalized.clear();
                            pending_raw.clear();
                            pending_corrected = None;
                            pending_corrected_count = 0;
                            last_sent_count = 0;
                            displayed_text.clear();
                            llm_seq = 0;
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
                // Check for LLM corrections
                if let Some(ref llm_handle) = llm {
                    while let Some((seq, corrected)) = llm_handle.try_recv() {
                        if seq == llm_seq - 1 {
                            pending_corrected = Some(corrected);
                            pending_corrected_count = last_sent_count;
                            let pending = build_pending_display(
                                &pending_raw, &pending_corrected, pending_corrected_count,
                            );
                            let display = build_display(&finalized, &pending);
                            retype(&mut enigo, &mut displayed_text, &display);
                        }
                    }
                }

                // Track speech activity for silence timeout
                if prob >= VAD_THRESHOLD {
                    app_state.touch_speech();
                    utterance_silence = 0;
                    chunk_has_speech = true;
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
                    if chunk_has_speech {
                        transcribe_and_type(&transcriber, &mut audio_buf, &mut enigo, &mut typed_any);
                    }
                    play_sound("Funk");
                    deactivate(&mut app_state, &mut audio_buf, &mut vad, &mut debouncer);
                    finalized.clear();
                    pending_raw.clear();
                    pending_corrected = None;
                    pending_corrected_count = 0;
                    last_sent_count = 0;
                    displayed_text.clear();
                    utterance_silence = 0;
                    chunk_has_speech = false;
                    continue;
                }

                // Transcribe when utterance ends (pause detected) or max duration hit
                let pause_detected = utterance_silence >= UTTERANCE_SILENCE_FRAMES
                    && audio_buf.len() > 0;
                let max_duration = audio_buf.len() >= MAX_UTTERANCE_SAMPLES;

                if pause_detected || max_duration {
                    if chunk_has_speech {
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
                                        finalized.clear();
                                        pending_raw.clear();
                                        pending_corrected = None;
                                        pending_corrected_count = 0;
                                        last_sent_count = 0;
                                        displayed_text.clear();
                                        utterance_silence = 0;
                                        chunk_has_speech = false;
                                        continue;
                                    }

                                    // Add new raw utterance (never fold corrections back)
                                    pending_raw.push(trimmed.to_string());

                                    // Finalize if window is too large and we have a correction
                                    let raw_joined = pending_raw.join(" ");
                                    if raw_joined.len() > 300 && pending_corrected.is_some() {
                                        let corrected = pending_corrected.take().unwrap();
                                        if finalized.is_empty() {
                                            finalized = corrected;
                                        } else {
                                            finalized = format!("{} {}", finalized, corrected);
                                        }
                                        pending_raw = pending_raw.split_off(pending_corrected_count);
                                        pending_corrected_count = 0;
                                    }

                                    // Optimistic display
                                    let pending = build_pending_display(
                                        &pending_raw, &pending_corrected, pending_corrected_count,
                                    );
                                    let display = build_display(&finalized, &pending);
                                    retype(&mut enigo, &mut displayed_text, &display);
                                    typed_any = true;

                                    // Send all raw to LLM
                                    if let Some(ref llm_handle) = llm {
                                        let raw_text = pending_raw.join(" ");
                                        let hint = tail_chars(&finalized, 80);
                                        llm_handle.request(llm_seq, hint, &raw_text);
                                        last_sent_count = pending_raw.len();
                                        llm_seq += 1;
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("Transcription error: {:#}", e);
                            }
                        }
                    } else {
                        audio_buf.clear();
                    }
                    utterance_silence = 0;
                    chunk_has_speech = false;
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

/// Diff-based retype: backspace the changed suffix and type the new one.
fn retype(enigo: &mut Enigo, displayed: &mut String, new_text: &str) {
    let common_chars = displayed
        .chars()
        .zip(new_text.chars())
        .take_while(|(a, b)| a == b)
        .count();
    let common_byte_len: usize = displayed
        .chars()
        .take(common_chars)
        .map(|c| c.len_utf8())
        .sum();

    // Backspace the suffix of displayed that changed
    let remove_chars = displayed[common_byte_len..].chars().count();
    for _ in 0..remove_chars {
        let _ = enigo.key(Key::Backspace, Direction::Click);
    }

    // Type the new suffix
    let new_suffix = &new_text[common_byte_len..];
    if !new_suffix.is_empty() {
        let _ = enigo.text(new_suffix);
    }

    *displayed = new_text.to_string();
}
