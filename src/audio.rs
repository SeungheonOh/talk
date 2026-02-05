use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream, StreamConfig};
use ringbuf::{traits::*, HeapRb};

/// Resampler that converts from native sample rate to 16kHz using linear interpolation.
struct Resampler {
    ratio: f64,
    fractional_pos: f64,
    last_sample: f32,
}

impl Resampler {
    fn new(source_rate: u32, target_rate: u32) -> Self {
        Self {
            ratio: source_rate as f64 / target_rate as f64,
            fractional_pos: 0.0,
            last_sample: 0.0,
        }
    }

    /// Returns true if resampling is needed (rates differ).
    fn is_needed(&self) -> bool {
        (self.ratio - 1.0).abs() > 0.001
    }

    /// Resample a block of mono f32 samples. Writes output into `out` buffer, returns count written.
    fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        out.clear();
        if input.is_empty() {
            return;
        }

        while (self.fractional_pos as usize) < input.len() {
            let idx = self.fractional_pos as usize;
            let frac = self.fractional_pos - idx as f64;

            let current = input[idx];
            let next = if idx + 1 < input.len() {
                input[idx + 1]
            } else {
                current
            };

            let sample = current + (next - current) * frac as f32;
            out.push(sample);
            self.fractional_pos += self.ratio;
        }

        // Keep last sample for interpolation across blocks
        self.last_sample = *input.last().unwrap_or(&0.0);
        self.fractional_pos -= input.len() as f64;
    }
}

/// Build the audio input stream and return (Stream, Consumer).
/// The consumer yields 16kHz mono f32 samples.
pub fn build_input_stream() -> Result<(Stream, ringbuf::HeapCons<f32>)> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("No input device available. Check microphone permissions.")?;

    let device_name = device
        .description()
        .map(|d| d.to_string())
        .unwrap_or_else(|_| "Unknown".into());
    let supported = device
        .default_input_config()
        .context("Failed to get default input config")?;

    let sample_format = supported.sample_format();
    let sample_rate = supported.sample_rate();
    let channels = supported.channels();

    eprintln!(
        "Audio device: {} ({}Hz, {} ch, {:?})",
        device_name, sample_rate, channels, sample_format
    );

    let config: StreamConfig = supported.into();

    // Ring buffer: 5 seconds of 16kHz mono audio
    let rb = HeapRb::<f32>::new(16000 * 5);
    let (producer, consumer) = rb.split();

    let stream = build_stream_for_format(
        &device,
        &config,
        sample_format,
        sample_rate,
        channels,
        producer,
    )?;

    stream.play().context("Failed to start audio stream")?;

    Ok((stream, consumer))
}

fn build_stream_for_format(
    device: &cpal::Device,
    config: &StreamConfig,
    format: SampleFormat,
    sample_rate: u32,
    channels: u16,
    producer: ringbuf::HeapProd<f32>,
) -> Result<Stream> {
    match format {
        SampleFormat::F32 => build_typed_stream::<f32>(
            device,
            config,
            sample_rate,
            channels,
            producer,
            |s| s,
        ),
        SampleFormat::I16 => build_typed_stream::<i16>(
            device,
            config,
            sample_rate,
            channels,
            producer,
            |s| s as f32 / 32768.0,
        ),
        SampleFormat::U16 => build_typed_stream::<u16>(
            device,
            config,
            sample_rate,
            channels,
            producer,
            |s| (s as f32 / 32768.0) - 1.0,
        ),
        other => anyhow::bail!("Unsupported sample format: {:?}", other),
    }
}

fn build_typed_stream<T: cpal::SizedSample + Send + 'static>(
    device: &cpal::Device,
    config: &StreamConfig,
    sample_rate: u32,
    channels: u16,
    mut producer: ringbuf::HeapProd<f32>,
    to_f32: fn(T) -> f32,
) -> Result<Stream> {
    // Pre-allocate buffers outside callback to avoid allocations in real-time thread.
    // We use a mutex-free approach: these buffers live in the closure and are reused.
    let mut mono_buf: Vec<f32> = Vec::with_capacity(4096);
    let mut resample_buf: Vec<f32> = Vec::with_capacity(4096);
    let mut resampler = Resampler::new(sample_rate, 16000);
    let needs_resample = resampler.is_needed();
    let ch = channels as usize;

    let stream = device.build_input_stream(
        config,
        move |data: &[T], _info: &cpal::InputCallbackInfo| {
            // Convert to mono f32
            mono_buf.clear();
            if ch == 1 {
                mono_buf.extend(data.iter().map(|&s| to_f32(s)));
            } else {
                // Average all channels per frame
                for frame in data.chunks(ch) {
                    let sum: f32 = frame.iter().map(|&s| to_f32(s)).sum();
                    mono_buf.push(sum / ch as f32);
                }
            }

            // Resample to 16kHz if needed
            let samples = if needs_resample {
                resampler.process(&mono_buf, &mut resample_buf);
                &resample_buf
            } else {
                &mono_buf
            };

            // Push to ring buffer (lock-free)
            let written = producer.push_slice(samples);
            if written < samples.len() {
                // Buffer overflow — consumer too slow. Samples dropped.
                // Don't log here (real-time thread), just accept the loss.
            }
        },
        |err| eprintln!("Audio stream error: {}", err),
        None,
    )?;

    Ok(stream)
}
