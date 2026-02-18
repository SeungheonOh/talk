use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use ringbuf::traits::*;

use crate::audio;
use crate::buffer::{AudioBuffer, RollingBuffer};
use crate::llm;
use crate::state::AppState;
use crate::stt;
use crate::transcriber::{Transcriber, VoiceCommand};
use crate::vad::Vad;
use crate::wake::{WakeDetector, WakeResult};
use crate::wake_sound::WakeSoundDetector;
use crate::wake_word::WakeWordDetector;

const VAD_MODEL_PATH: &str = "models/silero_vad.onnx";
const VAD_THRESHOLD: f32 = 0.5;
const SILENCE_TIMEOUT_SECS: u64 = 30;
const ROLLING_BUFFER_SECS: f32 = 3.0;
const UTTERANCE_SILENCE_FRAMES: u32 = 10;
const MAX_UTTERANCE_SAMPLES: usize = 16000 * 15;

/// Events sent from the processing thread to the UI.
#[derive(Debug, Clone)]
pub enum VoiceEvent {
    /// Wake event detected — show overlay, start pulsing.
    Activated,
    /// In-progress streaming text for overlay display.
    StreamUpdate(String),
    /// Utterance finalized (raw text, for display until LLM corrects).
    UtteranceFinalized(String),
    /// LLM corrected accumulated text. (corrected, number of raw utterances covered)
    LlmCorrected(String, usize),
    /// Session ended — commit text, no Enter.
    Deactivated,
    /// Session ended — commit text + press Enter.
    DeactivatedEnter,
    /// Session ended — discard all text.
    DeactivatedDiscard,
}

fn play_sound(name: &str) {
    let path = format!("/System/Library/Sounds/{}.aiff", name);
    let _ = Command::new("afplay").arg(&path).spawn();
}

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

pub fn run_processing_loop(
    engine_name: Option<String>,
    quant: Option<String>,
    wake_mode: String,
    tx: Sender<VoiceEvent>,
    running: Arc<AtomicBool>,
) -> Result<()> {
    // Load models
    eprintln!("Loading VAD model from '{}'...", VAD_MODEL_PATH);
    let mut vad = Vad::new(VAD_MODEL_PATH)?;
    eprintln!("VAD model loaded.");
    let engine = stt::create_engine(engine_name.as_deref(), quant.as_deref())?;

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

    // Audio capture
    let (mut _stream, mut consumer) = audio::build_input_stream()?;
    let mut last_audio_data = Instant::now();
    let mut last_reconnect_attempt = Instant::now();

    let streaming = engine.supports_streaming();
    if streaming {
        eprintln!("Streaming mode enabled.");
    }

    // State
    let mut app_state = AppState::Sleep;
    let mut rolling = RollingBuffer::new(ROLLING_BUFFER_SECS);
    let mut audio_buf = AudioBuffer::new();
    let mut utterance_silence: u32 = 0;
    let mut chunk_has_speech = false;

    // LLM post-processing
    let llm = llm::LlmHandle::spawn("http://127.0.0.1:8080");
    let mut all_raw: Vec<String> = Vec::new();
    let mut llm_seq: u64 = 0;
    let mut llm_sent_counts: Vec<usize> = Vec::new();

    // Streaming
    let mut stream_utterance = String::new();

    let window_size = vad.window_size();
    let mut vad_window = vec![0.0f32; window_size];

    let wake_hint = if wake_mode == "word" {
        "say wake phrase"
    } else {
        "snap/clap/whistle"
    };
    eprintln!(
        "\n\u{1f4a4} System ready — {} to start. Voice commands: presto (enter), disco (discard), apex (stop).\n",
        wake_hint
    );

    while running.load(Ordering::SeqCst) {
        // Wait for audio data
        if consumer.occupied_len() < window_size {
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
        rolling.push(&vad_window);

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
                        utterance_silence = 0;
                        chunk_has_speech = false;
                        all_raw.clear();
                        llm_seq = 0;
                        llm_sent_counts.clear();
                        stream_utterance.clear();
                        if streaming {
                            if let Err(e) = engine.stream_start() {
                                eprintln!("Stream start error: {:#}", e);
                            }
                        }
                        play_sound("Tink");
                        let _ = tx.send(VoiceEvent::Activated);
                        eprintln!("\n\u{1f3a4} [ACTIVATED] — Listening and transcribing...\n");
                    }
                    WakeResult::Nothing => {}
                }
            }

            AppState::Active { .. } if streaming => {
                // ── Streaming path ──

                // Check for LLM corrections
                if let Some(ref llm_handle) = llm {
                    while let Some((seq, corrected)) = llm_handle.try_recv() {
                        let count = llm_sent_counts.get(seq as usize).copied().unwrap_or(all_raw.len());
                        let _ = tx.send(VoiceEvent::LlmCorrected(corrected, count));
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

                // Feed audio to stream
                match engine.stream_feed(&vad_window) {
                    Ok(text) if !text.is_empty() => {
                        stream_utterance.push_str(&text);
                        let _ = tx.send(VoiceEvent::StreamUpdate(
                            stream_utterance.trim().to_string(),
                        ));
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
                    // Finalize whatever we have before deactivating
                    let utterance = stream_utterance.trim().to_string();
                    if !utterance.is_empty() {
                        let _ = tx.send(VoiceEvent::UtteranceFinalized(utterance));
                    }
                    play_sound("Funk");
                    let _ = tx.send(VoiceEvent::Deactivated);
                    deactivate(&mut app_state, &mut audio_buf, &mut vad, &mut *wake_detector);
                    all_raw.clear();
                    llm_seq = 0;
                    utterance_silence = 0;
                    chunk_has_speech = false;
                    stream_utterance.clear();
                    continue;
                }

                // Utterance boundary
                let pause_detected =
                    utterance_silence >= UTTERANCE_SILENCE_FRAMES && chunk_has_speech;

                if pause_detected {
                    if let Ok(tail) = engine.stream_finish() {
                        if !tail.is_empty() {
                            stream_utterance.push_str(&tail);
                            let _ = tx.send(VoiceEvent::StreamUpdate(
                                stream_utterance.trim().to_string(),
                            ));
                        }
                    }
                    if let Err(e) = engine.stream_start() {
                        eprintln!("Stream restart error: {:#}", e);
                    }

                    let utterance = stream_utterance.trim().to_string();
                    stream_utterance.clear();

                    if !utterance.is_empty() {
                        let (cmd, clean_text) = Transcriber::check_voice_command(&utterance);
                        if let Some(cmd) = cmd {
                            if !clean_text.is_empty() {
                                let _ = tx.send(VoiceEvent::UtteranceFinalized(clean_text));
                            }
                            let event = match cmd {
                                VoiceCommand::Presto => {
                                    eprintln!("\n\u{23f8}\u{fe0f} [DEACTIVATED] — presto\n");
                                    VoiceEvent::DeactivatedEnter
                                }
                                VoiceCommand::Disco => {
                                    eprintln!("\n\u{23f8}\u{fe0f} [DEACTIVATED] — disco\n");
                                    VoiceEvent::DeactivatedDiscard
                                }
                                VoiceCommand::Apex => {
                                    eprintln!("\n\u{23f8}\u{fe0f} [DEACTIVATED] — apex\n");
                                    VoiceEvent::Deactivated
                                }
                            };
                            let _ = engine.stream_finish();
                            play_sound("Funk");
                            let _ = tx.send(event);
                            deactivate(
                                &mut app_state,
                                &mut audio_buf,
                                &mut vad,
                                &mut *wake_detector,
                            );
                            all_raw.clear();
                            llm_seq = 0;
                            utterance_silence = 0;
                            chunk_has_speech = false;
                            continue;
                        }

                        // Commit utterance
                        let _ = tx.send(VoiceEvent::UtteranceFinalized(utterance.clone()));
                        all_raw.push(utterance);

                        // Send full accumulated text to LLM
                        if let Some(ref llm_handle) = llm {
                            let full_raw = all_raw.join(" ");
                            llm_sent_counts.push(all_raw.len());
                            llm_handle.request(llm_seq, "", &full_raw);
                            llm_seq += 1;
                        }
                    }
                    utterance_silence = 0;
                    chunk_has_speech = false;
                }
            }

            AppState::Active { .. } => {
                // ── Batch path ──

                // Check for LLM corrections
                if let Some(ref llm_handle) = llm {
                    while let Some((seq, corrected)) = llm_handle.try_recv() {
                        let count = llm_sent_counts.get(seq as usize).copied().unwrap_or(all_raw.len());
                        let _ = tx.send(VoiceEvent::LlmCorrected(corrected, count));
                    }
                }

                if prob >= VAD_THRESHOLD {
                    app_state.touch_speech();
                    utterance_silence = 0;
                    chunk_has_speech = true;
                } else {
                    utterance_silence += 1;
                }

                audio_buf.push(&vad_window);

                // Global silence timeout
                if app_state.silence_timeout_exceeded(SILENCE_TIMEOUT_SECS) {
                    eprintln!(
                        "\n\u{23f8}\u{fe0f} [DEACTIVATED] — Silence timeout ({}s)\n",
                        SILENCE_TIMEOUT_SECS
                    );
                    if chunk_has_speech && audio_buf.len() > 0 {
                        let audio = audio_buf.take();
                        if let Ok(text) = transcriber.transcribe(&audio) {
                            let trimmed = text.trim();
                            if !trimmed.is_empty() {
                                let _ = tx.send(VoiceEvent::UtteranceFinalized(
                                    trimmed.to_string(),
                                ));
                            }
                        }
                    }
                    play_sound("Funk");
                    let _ = tx.send(VoiceEvent::Deactivated);
                    deactivate(&mut app_state, &mut audio_buf, &mut vad, &mut *wake_detector);
                    all_raw.clear();
                    llm_seq = 0;
                    utterance_silence = 0;
                    chunk_has_speech = false;
                    continue;
                }

                let pause_detected =
                    utterance_silence >= UTTERANCE_SILENCE_FRAMES && audio_buf.len() > 0;
                let max_duration = audio_buf.len() >= MAX_UTTERANCE_SAMPLES;

                if pause_detected || max_duration {
                    if chunk_has_speech {
                        let chunk = audio_buf.take();
                        match transcriber.transcribe(&chunk) {
                            Ok(text) => {
                                let trimmed = text.trim();
                                if !trimmed.is_empty() {
                                    let (cmd, clean_text) =
                                        Transcriber::check_voice_command(trimmed);
                                    if let Some(cmd) = cmd {
                                        if !clean_text.is_empty() {
                                            let _ = tx.send(VoiceEvent::UtteranceFinalized(
                                                clean_text,
                                            ));
                                        }
                                        let event = match cmd {
                                            VoiceCommand::Presto => {
                                                eprintln!("\n\u{23f8}\u{fe0f} [DEACTIVATED] — presto\n");
                                                VoiceEvent::DeactivatedEnter
                                            }
                                            VoiceCommand::Disco => {
                                                eprintln!("\n\u{23f8}\u{fe0f} [DEACTIVATED] — disco\n");
                                                VoiceEvent::DeactivatedDiscard
                                            }
                                            VoiceCommand::Apex => {
                                                eprintln!("\n\u{23f8}\u{fe0f} [DEACTIVATED] — apex\n");
                                                VoiceEvent::Deactivated
                                            }
                                        };
                                        play_sound("Funk");
                                        let _ = tx.send(event);
                                        deactivate(
                                            &mut app_state,
                                            &mut audio_buf,
                                            &mut vad,
                                            &mut *wake_detector,
                                        );
                                        all_raw.clear();
                                        llm_seq = 0;
                                        utterance_silence = 0;
                                        chunk_has_speech = false;
                                        continue;
                                    }

                                    let _ = tx.send(VoiceEvent::UtteranceFinalized(
                                        trimmed.to_string(),
                                    ));
                                    all_raw.push(trimmed.to_string());

                                    if let Some(ref llm_handle) = llm {
                                        let full_raw = all_raw.join(" ");
                                        llm_sent_counts.push(all_raw.len());
                                        llm_handle.request(llm_seq, "", &full_raw);
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
