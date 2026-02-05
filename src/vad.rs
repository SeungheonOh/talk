use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array3, Ix3};
use ort::session::Session;
use ort::value::Tensor;

const SAMPLE_RATE: i64 = 16000;
const WINDOW_SIZE: usize = 512; // 32ms at 16kHz
const CONTEXT_SIZE: usize = 64; // context prepended to each window

/// Silero VAD wrapper. Runs speech probability inference on 512-sample (32ms) windows.
pub struct Vad {
    session: Session,
    state: Array3<f32>,   // RNN hidden state [2, 1, 128]
    context: Vec<f32>,    // Last CONTEXT_SIZE samples from previous window
}

impl Vad {
    /// Load Silero VAD model from an ONNX file.
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()
            .context("Failed to create ONNX session builder")?
            .commit_from_file(model_path)
            .with_context(|| format!("Failed to load VAD model from '{}'", model_path))?;

        Ok(Self {
            session,
            state: Array3::<f32>::zeros((2, 1, 128)),
            context: vec![0.0f32; CONTEXT_SIZE],
        })
    }

    /// Run inference on a 512-sample window. Returns speech probability [0.0, 1.0].
    pub fn predict(&mut self, audio_chunk: &[f32]) -> Result<f32> {
        assert_eq!(
            audio_chunk.len(),
            WINDOW_SIZE,
            "VAD expects exactly {} samples, got {}",
            WINDOW_SIZE,
            audio_chunk.len()
        );

        // Prepend context to audio chunk -> shape [1, CONTEXT_SIZE + WINDOW_SIZE]
        let mut input_data = Vec::with_capacity(CONTEXT_SIZE + WINDOW_SIZE);
        input_data.extend_from_slice(&self.context);
        input_data.extend_from_slice(audio_chunk);

        let input_len = input_data.len();
        let input = Array2::from_shape_vec((1, input_len), input_data.clone())
            .context("Failed to create input tensor")?;

        let sr = Array1::from_vec(vec![SAMPLE_RATE]);

        let outputs = self.session.run(
            ort::inputs![
                "input" => Tensor::from_array(input)?,
                "state" => Tensor::from_array(self.state.clone())?,
                "sr" => Tensor::from_array(sr)?
            ],
        )?;

        // Extract speech probability from output [1, 1]
        let prob_view = outputs["output"]
            .try_extract_array::<f32>()
            .context("Failed to extract output tensor")?;
        let prob = prob_view[[0, 0]];

        // Update RNN state from stateN output [2, 1, 128]
        let new_state_view = outputs["stateN"]
            .try_extract_array::<f32>()
            .context("Failed to extract state tensor")?;
        self.state = new_state_view
            .into_dimensionality::<Ix3>()
            .context("State shape mismatch")?
            .to_owned();

        // Update context with last CONTEXT_SIZE samples
        self.context
            .copy_from_slice(&input_data[input_len - CONTEXT_SIZE..]);

        Ok(prob)
    }

    /// Reset RNN state and context. Call between utterances.
    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.context.fill(0.0);
    }

    /// The number of samples expected per predict() call.
    pub fn window_size(&self) -> usize {
        WINDOW_SIZE
    }
}
