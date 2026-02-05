use std::sync::{Arc, Mutex};

use anyhow::{bail, Context, Result};
use ndarray::{Array1, Array2, Array3, ArrayViewD, Ix3};
use ort::session::Session;
use ort::value::Tensor;
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use crate::stt::SttEngine;

// Audio params (match NeMo's default FastConformer preprocessing)
const SAMPLE_RATE: usize = 16000;
const N_FFT: usize = 512;
const HOP_LENGTH: usize = 160;
const WIN_LENGTH: usize = 400;
const N_MELS: usize = 80;
const PRE_EMPHASIS: f32 = 0.97;
const MEL_FREQ_MIN: f32 = 0.0;
const MEL_FREQ_MAX: f32 = 8000.0;
const N_FREQ: usize = N_FFT / 2 + 1;

// TDT decode params
const BLANK_ID: usize = 1024;
const TDT_DURATIONS: [usize; 5] = [0, 1, 2, 3, 4];

/// Parakeet TDT engine using ONNX Runtime.
pub struct ParakeetEngine {
    encoder: Mutex<Session>,
    decoder_joint: Mutex<Session>,
    vocab: Vec<String>,
    // Precomputed mel spectrogram constants
    hann_window: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
    mel_filters: Vec<Vec<f32>>,
}

impl ParakeetEngine {
    pub fn new(model_dir: &str) -> Result<Self> {
        let dir = std::path::Path::new(model_dir);

        let encoder = Session::builder()
            .context("Failed to create encoder session builder")?
            .commit_from_file(dir.join("encoder-model.onnx"))
            .context("Failed to load encoder-model.onnx")?;

        let decoder_joint = Session::builder()
            .context("Failed to create decoder_joint session builder")?
            .commit_from_file(dir.join("decoder_joint-model.onnx"))
            .context("Failed to load decoder_joint-model.onnx")?;

        let vocab_path = dir.join("vocab.txt");
        let vocab_text =
            std::fs::read_to_string(&vocab_path).context("Failed to read vocab.txt")?;
        let vocab: Vec<String> = vocab_text.lines().map(|l| l.to_string()).collect();

        if vocab.len() < BLANK_ID {
            bail!(
                "vocab.txt has {} entries, expected at least {} (blank_id)",
                vocab.len(),
                BLANK_ID
            );
        }

        // Precompute Hann window
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

        Ok(Self {
            encoder: Mutex::new(encoder),
            decoder_joint: Mutex::new(decoder_joint),
            vocab,
            hann_window,
            fft,
            mel_filters,
        })
    }

    /// Compute log-mel spectrogram from 16kHz audio.
    /// Returns shape [1, N_MELS, T].
    fn mel_spectrogram(&self, audio: &[f32]) -> Array3<f32> {
        // Pre-emphasis
        let mut emphasized = Vec::with_capacity(audio.len());
        emphasized.push(audio[0]);
        for i in 1..audio.len() {
            emphasized.push(audio[i] - PRE_EMPHASIS * audio[i - 1]);
        }

        // STFT using precomputed FFT plan and Hann window
        let n_frames = if emphasized.len() >= WIN_LENGTH {
            (emphasized.len() - WIN_LENGTH) / HOP_LENGTH + 1
        } else {
            0
        };

        let mut power_spec = vec![vec![0.0f32; n_frames]; N_FREQ];

        for frame_idx in 0..n_frames {
            let start = frame_idx * HOP_LENGTH;
            let mut fft_buf = vec![Complex::new(0.0f32, 0.0f32); N_FFT];

            for i in 0..WIN_LENGTH {
                let sample = if start + i < emphasized.len() {
                    emphasized[start + i]
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

        // Apply precomputed mel filters + log
        let mut mel = Array3::<f32>::zeros((1, N_MELS, n_frames));
        for m in 0..N_MELS {
            for t in 0..n_frames {
                let mut sum = 0.0f32;
                for k in 0..N_FREQ {
                    sum += self.mel_filters[m][k] * power_spec[k][t];
                }
                mel[[0, m, t]] = (sum.max(1e-10)).ln();
            }
        }

        // Per-feature normalization (zero mean, unit variance)
        if n_frames > 1 {
            for m in 0..N_MELS {
                let mut sum = 0.0f64;
                let mut sum_sq = 0.0f64;
                for t in 0..n_frames {
                    let v = mel[[0, m, t]] as f64;
                    sum += v;
                    sum_sq += v * v;
                }
                let mean = sum / n_frames as f64;
                let var = (sum_sq / n_frames as f64 - mean * mean).max(1e-10);
                let std = var.sqrt();
                for t in 0..n_frames {
                    mel[[0, m, t]] = ((mel[[0, m, t]] as f64 - mean) / std) as f32;
                }
            }
        }

        mel
    }

    /// Run encoder on mel spectrogram. Returns encoder output [1, D, T'].
    fn encode(&self, mel: &Array3<f32>) -> Result<Array3<f32>> {
        let n_frames = mel.shape()[2];
        let length = Array1::from_vec(vec![n_frames as i64]);

        let mut encoder = self.encoder.lock().unwrap();
        let outputs = encoder.run(
            ort::inputs![
                "audio_signal" => Tensor::from_array(mel.clone())?,
                "length" => Tensor::from_array(length)?
            ],
        )?;

        let encoded: ArrayViewD<f32> = outputs[0]
            .try_extract_array::<f32>()
            .context("Failed to extract encoder output")?;

        let encoded = encoded
            .into_dimensionality::<Ix3>()
            .context("Encoder output shape mismatch")?
            .to_owned();

        Ok(encoded)
    }

    /// TDT greedy decode on encoder output.
    /// Unlike RNNT, TDT predicts both a token and a duration at each step,
    /// allowing the decoder to skip multiple encoder frames at once.
    fn decode(&self, encoder_out: &Array3<f32>) -> Result<String> {
        let t_total = encoder_out.shape()[2];
        let enc_dim = encoder_out.shape()[1];

        // LSTM initial states: hidden and cell [num_layers, 1, 640]
        let mut hidden = Array3::<f32>::zeros((2, 1, 640));
        let mut cell = Array3::<f32>::zeros((2, 1, 640));

        let mut prev_token = BLANK_ID as i32;
        let mut tokens: Vec<usize> = Vec::new();

        let mut t = 0;
        while t < t_total {
            // Extract single encoder frame as [1, enc_dim, 1]
            let mut enc_frame = Array3::<f32>::zeros((1, enc_dim, 1));
            for d in 0..enc_dim {
                enc_frame[[0, d, 0]] = encoder_out[[0, d, t]];
            }

            let input_token = Array2::from_shape_vec((1, 1), vec![prev_token])
                .context("Failed to create token tensor")?;

            let mut decoder = self.decoder_joint.lock().unwrap();
            let outputs = decoder.run(
                ort::inputs![
                    "encoder_outputs" => Tensor::from_array(enc_frame)?,
                    "targets" => Tensor::from_array(input_token)?,
                    "target_length" => Tensor::from_array(Array1::from_vec(vec![1i32]))?,
                    "input_states_1" => Tensor::from_array(hidden.clone())?,
                    "input_states_2" => Tensor::from_array(cell.clone())?
                ],
            )?;

            // Logits shape: [1, 1, vocab_size + 1 + num_durations]
            let logits_view: ArrayViewD<f32> = outputs[0]
                .try_extract_array::<f32>()
                .context("Failed to extract joint logits")?;
            let logits_slice = logits_view.as_slice().context("Logits not contiguous")?;

            // Split: first (BLANK_ID+1) are token logits, rest are duration logits
            let token_logits = &logits_slice[..BLANK_ID + 1];
            let duration_logits = &logits_slice[BLANK_ID + 1..];

            let best_token = argmax(token_logits);
            let best_dur_idx = argmax(duration_logits);
            let best_duration = TDT_DURATIONS
                .get(best_dur_idx)
                .copied()
                .unwrap_or(1);

            if best_token != BLANK_ID {
                tokens.push(best_token);
                prev_token = best_token as i32;
            }

            // Advance by at least 1 frame to prevent infinite loops
            t += best_duration.max(1);

            // Update LSTM states (outputs[1] is prednet_lengths, skip it)
            let new_hidden: ArrayViewD<f32> = outputs[2]
                .try_extract_array::<f32>()
                .context("Failed to extract hidden state")?;
            hidden = new_hidden
                .into_dimensionality::<Ix3>()
                .context("Hidden state shape mismatch")?
                .to_owned();

            let new_cell: ArrayViewD<f32> = outputs[3]
                .try_extract_array::<f32>()
                .context("Failed to extract cell state")?;
            cell = new_cell
                .into_dimensionality::<Ix3>()
                .context("Cell state shape mismatch")?
                .to_owned();
        }

        // Detokenize: join vocab pieces, replace ▁ with space
        let text: String = tokens
            .iter()
            .filter_map(|&t| self.vocab.get(t))
            .cloned()
            .collect::<Vec<_>>()
            .join("")
            .replace('\u{2581}', " ");

        Ok(text)
    }
}

impl SttEngine for ParakeetEngine {
    fn transcribe(&self, audio: &[f32]) -> Result<String> {
        if audio.len() < WIN_LENGTH {
            return Ok(String::new());
        }
        let mel = self.mel_spectrogram(audio);
        let encoded = self.encode(&mel)?;
        self.decode(&encoded)
    }
}

/// Compute mel filterbank matrix [n_mels, n_freq].
fn mel_filterbank(
    n_mels: usize,
    n_freq: usize,
    sample_rate: usize,
    f_min: f32,
    f_max: f32,
) -> Vec<Vec<f32>> {
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    // n_mels + 2 equally spaced points in mel scale
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert Hz to FFT bin indices
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

        // Slaney normalization: scale each filter by 2/(right_hz - left_hz)
        let bandwidth = hz_points[m + 2] - hz_points[m];
        if bandwidth > 0.0 {
            let norm = 2.0 / bandwidth;
            for k in 0..n_freq {
                filters[m][k] *= norm;
            }
        }
    }

    filters
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

fn argmax(slice: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in slice.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}
