use std::sync::Arc;

use anyhow::{Context, Result};
use ndarray::Array3;
use ort::session::Session;
use ort::value::Tensor;
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

const CED_MODEL_PATH: &str = "models/ced-tiny/model.onnx";

// Target AudioSet classes: (index, label)
const TARGET_CLASSES: &[(usize, &str)] = &[
    (62, "Finger snapping"),
    (63, "Clapping"),
    (40, "Whistling"),
    (491, "Clicking"),
    (434, "Burst/pop"),
];

const CONFIDENCE_THRESHOLD: f32 = 0.5;

// Mel spectrogram params (CED-Tiny expects 64-mel with center padding)
const SAMPLE_RATE: usize = 16000;
const N_FFT: usize = 512;
const HOP_LENGTH: usize = 160;
const WIN_LENGTH: usize = 512;
const N_MELS: usize = 64;
const MEL_FREQ_MIN: f32 = 0.0;
const MEL_FREQ_MAX: f32 = 8000.0;
const N_FREQ: usize = N_FFT / 2 + 1; // 257

// Energy spike detection
const RMS_SPIKE_THRESHOLD: f32 = 0.02;
const RMS_SPIKE_RATIO: f32 = 5.0;
const EMA_DECAY: f32 = 0.995;

// Cooldown: 16 VAD frames × 32ms ≈ 512ms
const COOLDOWN_FRAMES: u32 = 16;

// Classify on 1 second of audio
const CLASSIFY_SAMPLES: usize = 16000;

// AmplitudeToDB params
const TOP_DB: f32 = 120.0;

pub struct WakeSoundDetector {
    session: Session,
    hann_window: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
    mel_filters: Vec<Vec<f32>>,
    ambient_rms: f32,
    cooldown: u32,
}

impl WakeSoundDetector {
    pub fn new() -> Result<Self> {
        eprintln!("Loading CED-Tiny model from '{}'...", CED_MODEL_PATH);

        // Try CoreML EP, fallback to CPU
        let session = {
            let coreml_ep = ort::ep::CoreML::default()
                .with_compute_units(ort::ep::coreml::ComputeUnits::All)
                .build();
            Session::builder()
                .context("Failed to create CED session builder")?
                .with_execution_providers([coreml_ep])
                .context("Failed to set CED execution providers")?
                .commit_from_file(CED_MODEL_PATH)
                .or_else(|_| {
                    eprintln!("CoreML failed for CED, falling back to CPU");
                    Session::builder()?.commit_from_file(CED_MODEL_PATH)
                })
                .context("Failed to load CED-Tiny model")?
        };

        // Precompute Hann window (512 samples)
        let hann_window: Vec<f32> = (0..WIN_LENGTH)
            .map(|i| {
                0.5 * (1.0
                    - (2.0 * std::f32::consts::PI * i as f32 / WIN_LENGTH as f32).cos())
            })
            .collect();

        // Precompute FFT plan
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(N_FFT);

        // Precompute mel filterbank
        let mel_filters = mel_filterbank(N_MELS, N_FREQ, SAMPLE_RATE, MEL_FREQ_MIN, MEL_FREQ_MAX);

        eprintln!("CED-Tiny model loaded.");

        Ok(Self {
            session,
            hann_window,
            fft,
            mel_filters,
            ambient_rms: 0.0,
            cooldown: 0,
        })
    }

    /// Cheap per-frame pre-filter. Returns true if an energy spike is detected
    /// and cooldown has expired.
    pub fn check_energy_spike(&mut self, window: &[f32]) -> bool {
        let rms = rms_energy(window);

        // Update ambient EMA
        self.ambient_rms = EMA_DECAY * self.ambient_rms + (1.0 - EMA_DECAY) * rms;

        // Tick cooldown
        if self.cooldown > 0 {
            self.cooldown -= 1;
            return false;
        }

        // Spike detection: absolute threshold AND ratio above ambient
        rms >= RMS_SPIKE_THRESHOLD && rms >= self.ambient_rms * RMS_SPIKE_RATIO
    }

    /// Classify the last 1s of audio. Returns (class_index, label, probability)
    /// if a target class is detected above threshold, or None.
    pub fn classify(&mut self, audio: &[f32]) -> Result<Option<(usize, &'static str, f32)>> {
        // Always set cooldown to prevent rapid re-triggers
        self.cooldown = COOLDOWN_FRAMES;

        // Take last 1s (or pad if shorter)
        let segment: Vec<f32> = if audio.len() >= CLASSIFY_SAMPLES {
            audio[audio.len() - CLASSIFY_SAMPLES..].to_vec()
        } else {
            let mut padded = vec![0.0f32; CLASSIFY_SAMPLES - audio.len()];
            padded.extend_from_slice(audio);
            padded
        };

        // Compute mel spectrogram
        let mel = self.mel_spectrogram(&segment);

        // Run inference
        let outputs = self.session.run(
            ort::inputs![
                "feats" => Tensor::from_array(mel)?
            ],
        ).context("CED-Tiny inference failed")?;

        let probs = outputs[0]
            .try_extract_array::<f32>()
            .context("Failed to extract CED output")?;
        let probs_slice = probs.as_slice().context("CED output not contiguous")?;

        // Find best target class above threshold
        let mut best: Option<(usize, &'static str, f32)> = None;
        for &(idx, label) in TARGET_CLASSES {
            if idx < probs_slice.len() {
                let p = probs_slice[idx];
                if p >= CONFIDENCE_THRESHOLD {
                    if best.is_none() || p > best.unwrap().2 {
                        best = Some((idx, label, p));
                    }
                }
            }
        }

        Ok(best)
    }

    /// Compute 64-mel spectrogram with center padding and AmplitudeToDB.
    /// No pre-emphasis, no per-feature normalization (BatchNorm is in the ONNX model).
    fn mel_spectrogram(&self, audio: &[f32]) -> Array3<f32> {
        // Center padding with reflect (matches torchaudio center=True, pad_mode='reflect')
        let pad = N_FFT / 2;
        let padded_len = audio.len() + 2 * pad;
        let mut padded = vec![0.0f32; padded_len];
        padded[pad..pad + audio.len()].copy_from_slice(audio);
        // Reflect left: padded[pad-1-i] = audio[1+i]
        for i in 0..pad.min(audio.len() - 1) {
            padded[pad - 1 - i] = audio[1 + i];
        }
        // Reflect right: padded[pad+len+i] = audio[len-2-i]
        for i in 0..pad.min(audio.len() - 1) {
            padded[pad + audio.len() + i] = audio[audio.len() - 2 - i];
        }

        // STFT with center-padded audio
        let n_frames = (padded_len - WIN_LENGTH) / HOP_LENGTH + 1;

        let mut power_spec = vec![vec![0.0f32; n_frames]; N_FREQ];

        for frame_idx in 0..n_frames {
            let start = frame_idx * HOP_LENGTH;
            let mut fft_buf = vec![Complex::new(0.0f32, 0.0f32); N_FFT];

            for i in 0..WIN_LENGTH {
                let sample = if start + i < padded.len() {
                    padded[start + i]
                } else {
                    0.0
                };
                fft_buf[i] = Complex::new(sample * self.hann_window[i], 0.0);
            }

            self.fft.process(&mut fft_buf);

            for k in 0..N_FREQ {
                power_spec[k][frame_idx] = fft_buf[k].norm_sqr();
            }
        }

        // Apply mel filters + AmplitudeToDB
        let mut mel = Array3::<f32>::zeros((1, N_MELS, n_frames));
        let mut max_val = f32::NEG_INFINITY;

        for m in 0..N_MELS {
            for t in 0..n_frames {
                let mut sum = 0.0f32;
                for k in 0..N_FREQ {
                    sum += self.mel_filters[m][k] * power_spec[k][t];
                }
                // Power to dB: 10 * log10(max(power, 1e-10))
                let db = 10.0 * sum.max(1e-10).log10();
                mel[[0, m, t]] = db;
                if db > max_val {
                    max_val = db;
                }
            }
        }

        // top_db clamp: clamp to [max_val - TOP_DB, max_val]
        let min_val = max_val - TOP_DB;
        for m in 0..N_MELS {
            for t in 0..n_frames {
                if mel[[0, m, t]] < min_val {
                    mel[[0, m, t]] = min_val;
                }
            }
        }

        mel
    }
}

fn rms_energy(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Compute mel filterbank matrix [n_mels][n_freq].
fn mel_filterbank(
    n_mels: usize,
    n_freq: usize,
    sample_rate: usize,
    f_min: f32,
    f_max: f32,
) -> Vec<Vec<f32>> {
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    let freq_resolution = sample_rate as f32 / (2.0 * (n_freq - 1) as f32);
    let bin_points: Vec<f32> = hz_points.iter().map(|&h| h / freq_resolution).collect();

    let mut filters = vec![vec![0.0f32; n_freq]; n_mels];

    for m in 0..n_mels {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];

        for k in 0..n_freq {
            let freq = k as f32;
            if freq >= left && freq <= center && center > left {
                filters[m][k] = (freq - left) / (center - left);
            } else if freq > center && freq <= right && right > center {
                filters[m][k] = (right - freq) / (right - center);
            }
        }

        // No Slaney normalization — torchaudio MelSpectrogram defaults to norm=None
    }

    filters
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}
