mod audio;
mod buffer;
mod llm;
mod state;
mod stt;
mod stt_parakeet;
mod stt_whisper;
#[cfg(feature = "voxtral")]
mod stt_voxtral;
mod transcriber;
mod vad;
mod wake;
mod wake_sound;
mod wake_word;

use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use enigo::{Direction, Enigo, Key, Keyboard, Settings};
use ringbuf::traits::*;

use buffer::{AudioBuffer, RollingBuffer};
use state::AppState;
use transcriber::Transcriber;
use vad::Vad;
use wake::{WakeDetector, WakeResult};
use wake_sound::WakeSoundDetector;
use wake_word::WakeWordDetector;

const VAD_MODEL_PATH: &str = "models/silero_vad.onnx";

const VAD_THRESHOLD: f32 = 0.5;
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

/// Build the complete display string including in-progress streaming text.
fn build_full_display(
    finalized: &str,
    pending_raw: &[String],
    pending_corrected: &Option<String>,
    corrected_count: usize,
    stream_utterance: &str,
) -> String {
    let pending = build_pending_display(pending_raw, pending_corrected, corrected_count);
    let mut display = build_display(finalized, &pending);
    let trimmed = stream_utterance.trim();
    if !trimmed.is_empty() {
        if !display.is_empty() {
            display.push(' ');
        }
        display.push_str(trimmed);
    }
    display
}

fn print_help() {
    let bin = std::env::args().next().unwrap_or_else(|| "voicer".into());
    eprintln!("Usage: {} [OPTIONS]", bin);
    eprintln!();
    eprintln!("Options:");
    #[cfg(feature = "voxtral")]
    eprintln!("  --engine <name>   STT engine: whisper, parakeet, voxtral (default: auto-detect)");
    #[cfg(not(feature = "voxtral"))]
    eprintln!("  --engine <name>   STT engine: whisper, parakeet (default: auto-detect)");
    eprintln!("  --quant <name>    Voxtral quantization: q8, f16, etc. (default: full precision)");
    eprintln!("  --wake <mode>     Wake mode: sound, word (default: sound)");
    eprintln!("  --help, -h        Show this help message");
    eprintln!();
    eprintln!("Wake modes:");
    eprintln!("  sound   Activate by snap, clap, whistle, click, or pop (CED-Tiny)");
    eprintln!("  word    Activate by saying a wake phrase (uses STT engine)");
    eprintln!();
    eprintln!("Say \"done\" or \"stop\" while active to deactivate.");
}

fn parse_args() -> (Option<String>, Option<String>, String) {
    let args: Vec<String> = std::env::args().collect();
    let mut engine: Option<String> = None;
    let mut quant: Option<String> = None;
    let mut wake = "sound".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            "--engine" => {
                engine = args.get(i + 1).cloned();
                i += 1;
            }
            "--quant" => {
                quant = args.get(i + 1).cloned();
                i += 1;
            }
            "--wake" => {
                if let Some(val) = args.get(i + 1) {
                    wake = val.clone();
                }
                i += 1;
            }
            _ if args[i].starts_with("--engine=") => {
                engine = args[i].strip_prefix("--engine=").map(|s| s.to_string());
            }
            _ if args[i].starts_with("--quant=") => {
                quant = args[i].strip_prefix("--quant=").map(|s| s.to_string());
            }
            _ if args[i].starts_with("--wake=") => {
                if let Some(val) = args[i].strip_prefix("--wake=") {
                    wake = val.to_string();
                }
            }
            other => {
                eprintln!("Unknown option: {}", other);
                print_help();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    (engine, quant, wake)
}

fn main() -> Result<()> {
    let (engine_name, quant, wake_mode) = parse_args();

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
    let engine = stt::create_engine(engine_name.as_deref(), quant.as_deref())?;

    // Warm up STT engine with a dummy inference to pay JIT/CoreML compilation cost upfront
    let _ = engine.transcribe(&vec![0.0f32; 8000]);
    eprintln!("STT engine warmed up.");

    let transcriber = Transcriber::new(engine.clone());

    let mut wake_detector: Box<dyn WakeDetector> = match wake_mode.as_str() {
        "word" => {
            eprintln!("Wake mode: speech (say wake phrase)");
            Box::new(WakeWordDetector::new(engine.clone()))
        }
        "sound" => {
            eprintln!("Wake mode: sound (snap/clap/whistle)");
            Box::new(WakeSoundDetector::new()?)
        }
        other => anyhow::bail!("Unknown wake mode '{}'. Use 'sound' or 'word'.", other),
    };

    // Start audio capture (with reconnection support)
    let (mut _stream, mut consumer) = audio::build_input_stream()?;
    let mut last_audio_data = Instant::now();
    let mut last_reconnect_attempt = Instant::now();

    // Keyboard simulator (mac_delay=5ms: small delay per key event to avoid dropped characters)
    let mut enigo = Enigo::new(&Settings {
        mac_delay: 1,
        ..Settings::default()
    }).context("Failed to init keyboard simulator")?;

    let streaming = engine.supports_streaming();
    if streaming {
        eprintln!("Streaming mode enabled.");
    }

    // State
    let mut app_state = AppState::Sleep;
    let mut rolling = RollingBuffer::new(ROLLING_BUFFER_SECS);
    let mut audio_buf = AudioBuffer::new();
    let mut typed_any = false;
    let mut utterance_silence: u32 = 0; // consecutive silence frames in Active mode
    let mut chunk_has_speech = false; // whether current chunk contains any speech

    // LLM post-processing (optional)
    let llm = llm::LlmHandle::spawn("http://127.0.0.1:8080");
    let mut finalized = String::new();
    let mut pending_raw: Vec<String> = Vec::new();
    let mut pending_corrected: Option<String> = None;
    let mut pending_corrected_count: usize = 0;
    let mut last_sent_count: usize = 0;
    let mut displayed_text = String::new();
    let mut llm_seq: u64 = 0;

    // Streaming state: accumulated text for the current utterance
    let mut stream_utterance = String::new();

    let window_size = vad.window_size();
    let mut vad_window = vec![0.0f32; window_size];

    let wake_hint = if wake_mode == "word" {
        "say wake phrase"
    } else {
        "snap/clap/whistle"
    };
    eprintln!(
        "\n\u{1f4a4} System ready — {} to start, \"done\" to stop.\n",
        wake_hint
    );

    // Main processing loop
    while running.load(Ordering::SeqCst) {
        // Wait until a full VAD window is available before consuming
        if consumer.occupied_len() < window_size {
            // No data — check if the audio device may have gone offline
            if last_audio_data.elapsed() > Duration::from_secs(2)
                && last_reconnect_attempt.elapsed() > Duration::from_secs(3)
            {
                last_reconnect_attempt = Instant::now();
                eprintln!("\u{1f50c} Audio device appears offline, attempting reconnect...");
                match audio::build_input_stream() {
                    Ok((new_stream, new_consumer)) => {
                        _stream = new_stream;
                        consumer = new_consumer;
                        last_audio_data = Instant::now();
                        eprintln!("\u{2705} Audio device reconnected.");
                    }
                    Err(_) => {
                        eprintln!("   Reconnect failed, will retry...");
                    }
                }
            }
            thread::sleep(Duration::from_millis(5));
            continue;
        }
        last_audio_data = Instant::now();
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
                match wake_detector.feed(&vad_window, prob, &rolling)? {
                    WakeResult::Activated(desc) => {
                        eprintln!("[WAKE] {}", desc);
                        app_state = AppState::activate();
                        audio_buf.clear();
                        vad.reset();
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
                        stream_utterance.clear();
                        if streaming {
                            if let Err(e) = engine.stream_start() {
                                eprintln!("Stream start error: {:#}", e);
                            }
                        }
                        play_sound("Tink");
                        eprintln!("\n\u{1f3a4} [ACTIVATED] — Listening and transcribing...\n");
                    }
                    WakeResult::Nothing => {}
                }
            }

            AppState::Active { .. } if streaming => {
                // ── Streaming path: feed every VAD window, get tokens in real-time ──

                // Check for LLM corrections
                if let Some(ref llm_handle) = llm {
                    while let Some((seq, corrected)) = llm_handle.try_recv() {
                        if llm_seq > 0 && seq == llm_seq - 1 {
                            pending_corrected = Some(corrected);
                            pending_corrected_count = last_sent_count;
                            let display = build_full_display(
                                &finalized, &pending_raw, &pending_corrected,
                                pending_corrected_count, &stream_utterance,
                            );
                            retype(&mut enigo, &mut displayed_text, &display);
                        }
                    }
                }

                // Track speech
                if prob >= VAD_THRESHOLD {
                    app_state.touch_speech();
                    utterance_silence = 0;
                    chunk_has_speech = true;
                } else {
                    utterance_silence += 1;
                }

                // Feed audio to stream and type any tokens immediately
                match engine.stream_feed(&vad_window) {
                    Ok(text) if !text.is_empty() => {
                        stream_utterance.push_str(&text);
                        let display = build_full_display(
                            &finalized, &pending_raw, &pending_corrected,
                            pending_corrected_count, &stream_utterance,
                        );
                        retype(&mut enigo, &mut displayed_text, &display);
                        typed_any = true;
                    }
                    Err(e) => eprintln!("Stream feed error: {:#}", e),
                    _ => {}
                }

                // Global silence timeout
                if app_state.silence_timeout_exceeded(SILENCE_TIMEOUT_SECS) {
                    eprintln!(
                        "\n\u{23f8}\u{fe0f} [DEACTIVATED] — Silence timeout ({}s)\n",
                        SILENCE_TIMEOUT_SECS
                    );
                    if let Ok(tail) = engine.stream_finish() {
                        stream_utterance.push_str(&tail);
                    }
                    play_sound("Funk");
                    deactivate(&mut app_state, &mut audio_buf, &mut vad, &mut *wake_detector);
                    finalized.clear();
                    pending_raw.clear();
                    pending_corrected = None;
                    pending_corrected_count = 0;
                    last_sent_count = 0;
                    displayed_text.clear();
                    utterance_silence = 0;
                    chunk_has_speech = false;
                    stream_utterance.clear();
                    continue;
                }

                // Utterance boundary: finish+restart stream, check deactivation, send to LLM
                let pause_detected = utterance_silence >= UTTERANCE_SILENCE_FRAMES
                    && chunk_has_speech;

                if pause_detected {
                    // Finish the stream to get remaining tokens, then restart
                    // for the next utterance. Using finish+start instead of flush
                    // avoids the ~6s silence padding flush injects into the mel
                    // buffer, which delays subsequent token production and causes
                    // large bursty retype diffs.
                    if let Ok(tail) = engine.stream_finish() {
                        if !tail.is_empty() {
                            stream_utterance.push_str(&tail);
                            let display = build_full_display(
                                &finalized, &pending_raw, &pending_corrected,
                                pending_corrected_count, &stream_utterance,
                            );
                            retype(&mut enigo, &mut displayed_text, &display);
                        }
                    }
                    if let Err(e) = engine.stream_start() {
                        eprintln!("Stream restart error: {:#}", e);
                    }

                    let utterance = stream_utterance.trim().to_string();
                    stream_utterance.clear();

                    if !utterance.is_empty() {
                        // Check deactivation
                        if Transcriber::is_deactivation_command(&utterance) {
                            eprintln!(
                                "\n\u{23f8}\u{fe0f} [DEACTIVATED] — Command recognized\n"
                            );
                            // Backspace the command word that streaming already
                            // typed (batch mode never types it, but streaming does).
                            let display = build_full_display(
                                &finalized, &pending_raw, &pending_corrected,
                                pending_corrected_count, "",
                            );
                            retype(&mut enigo, &mut displayed_text, &display);
                            let _ = engine.stream_finish();
                            play_sound("Funk");
                            deactivate(
                                &mut app_state, &mut audio_buf, &mut vad, &mut *wake_detector,
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

                        // Commit utterance to pending_raw for LLM correction
                        pending_raw.push(utterance);

                        // Finalize if too large
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

                        // Update display now that utterance moved from stream into pending
                        let display = build_full_display(
                            &finalized, &pending_raw, &pending_corrected,
                            pending_corrected_count, &stream_utterance,
                        );
                        retype(&mut enigo, &mut displayed_text, &display);
                        typed_any = true;

                        // Send to LLM
                        if let Some(ref llm_handle) = llm {
                            let raw_text = pending_raw.join(" ");
                            let hint = tail_chars(&finalized, 80);
                            llm_handle.request(llm_seq, hint, &raw_text);
                            last_sent_count = pending_raw.len();
                            llm_seq += 1;
                        }
                    }

                    utterance_silence = 0;
                    chunk_has_speech = false;
                }
            }

            AppState::Active { .. } => {
                // ── Batch path: accumulate audio, transcribe on silence boundary ──

                // Check for LLM corrections
                if let Some(ref llm_handle) = llm {
                    while let Some((seq, corrected)) = llm_handle.try_recv() {
                        if llm_seq > 0 && seq == llm_seq - 1 {
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
                    deactivate(&mut app_state, &mut audio_buf, &mut vad, &mut *wake_detector);
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
                                            &mut *wake_detector,
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
    wake_detector: &mut dyn WakeDetector,
) {
    *app_state = AppState::Sleep;
    audio_buf.clear();
    vad.reset();
    wake_detector.reset();
    eprintln!("\u{1f4a4} Listening for wake event...\n");
}

/// Diff-based retype using common prefix + suffix to minimize destructive edits.
fn retype(enigo: &mut Enigo, displayed: &mut String, new_text: &str) {
    if *displayed == new_text {
        return;
    }

    let old: Vec<char> = displayed.chars().collect();
    let new: Vec<char> = new_text.chars().collect();

    // Common prefix
    let prefix = old.iter().zip(new.iter()).take_while(|(a, b)| a == b).count();

    // Fast path: pure append (streaming tokens arrive at the end)
    if prefix == old.len() {
        let byte_off: usize = new_text.chars().take(prefix).map(|c| c.len_utf8()).sum();
        let _ = enigo.text(&new_text[byte_off..]);
        *displayed = new_text.to_string();
        return;
    }

    // Common suffix (not overlapping with prefix)
    let max_suffix = (old.len() - prefix).min(new.len() - prefix);
    let suffix = (0..max_suffix)
        .take_while(|&i| old[old.len() - 1 - i] == new[new.len() - 1 - i])
        .count();

    let remove_mid = old.len() - prefix - suffix;
    let insert_mid: String = new[prefix..new.len() - suffix].iter().collect();

    // Navigate left past unchanged suffix, edit, navigate back
    for _ in 0..suffix {
        let _ = enigo.key(Key::LeftArrow, Direction::Click);
    }
    for _ in 0..remove_mid {
        let _ = enigo.key(Key::Backspace, Direction::Click);
    }
    if !insert_mid.is_empty() {
        let _ = enigo.text(&insert_mid);
    }
    for _ in 0..suffix {
        let _ = enigo.key(Key::RightArrow, Direction::Click);
    }

    *displayed = new_text.to_string();
}
