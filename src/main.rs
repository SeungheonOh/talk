mod animations;
mod app;
mod audio;
mod buffer;
mod llm;
mod overlay_ui;
mod panel;
mod processing;
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

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

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
    eprintln!("  --pos <x,y>       Overlay position in pixels (default: bottom-center)");
    eprintln!("  --help, -h        Show this help message");
    eprintln!();
    eprintln!("Wake modes:");
    eprintln!("  sound   Activate by snap, clap, whistle, click, or pop (CED-Tiny)");
    eprintln!("  word    Activate by saying a wake phrase (uses STT engine)");
    eprintln!();
    eprintln!("Say \"done\" or \"stop\" while active to deactivate.");
}

fn parse_pos(s: &str) -> (f32, f32) {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() == 2 {
        if let (Ok(x), Ok(y)) = (parts[0].trim().parse(), parts[1].trim().parse()) {
            return (x, y);
        }
    }
    eprintln!("Invalid --pos value '{}', expected x,y (e.g. 100,100)", s);
    std::process::exit(1);
}

fn default_overlay_pos() -> (f32, f32) {
    let (w, h) = panel::main_display_size();
    // Center horizontally (offset by half pill width), near the bottom
    ((w / 2.0 - 250.0), h - 120.0)
}

fn parse_args() -> (Option<String>, Option<String>, String, (f32, f32)) {
    let args: Vec<String> = std::env::args().collect();
    let mut engine: Option<String> = None;
    let mut quant: Option<String> = None;
    let mut wake = "sound".to_string();
    let mut pos: Option<(f32, f32)> = None;

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
            "--pos" => {
                if let Some(val) = args.get(i + 1) {
                    pos = Some(parse_pos(val));
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
            _ if args[i].starts_with("--pos=") => {
                if let Some(val) = args[i].strip_prefix("--pos=") {
                    pos = Some(parse_pos(val));
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

    (engine, quant, wake, pos.unwrap_or_else(default_overlay_pos))
}

fn main() -> iced::Result {
    let (engine_name, quant, wake_mode, overlay_pos) = parse_args();

    // Graceful shutdown
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        eprintln!("\nShutting down...");
        r.store(false, Ordering::SeqCst);
    })
    .expect("Failed to set Ctrl+C handler");

    // Hide from Dock before creating any windows
    panel::hide_from_dock();

    // Schedule NSPanel conversion + tray icon (runs 200ms after iced creates the window)
    panel::configure_as_overlay_panel();

    // Channel: processing thread -> UI
    let (tx, rx) = mpsc::channel();

    // Spawn processing thread
    let proc_running = running.clone();
    std::thread::spawn(move || {
        if let Err(e) =
            processing::run_processing_loop(engine_name, quant, wake_mode, tx, proc_running)
        {
            eprintln!("Processing thread error: {:#}", e);
        }
    });

    let rx = Arc::new(std::sync::Mutex::new(Some(rx)));
    iced::application(
        move || {
            let rx = rx.lock().unwrap().take().expect("boot called twice");
            app::VoicerApp::new(rx, running.clone(), overlay_pos)
        },
        app::update,
        app::view,
    )
    .title("Voicer")
    .subscription(app::subscription)
    .theme(app::theme)
    .window(app::window_settings())
    .transparent(true)
    .style(|_state, _theme| iced::theme::Style {
        background_color: iced::Color::TRANSPARENT,
        text_color: iced::Color::WHITE,
    })
    .run()
}
