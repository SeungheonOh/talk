use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_void};
use std::sync::Mutex;

use anyhow::{Context, Result};

use crate::stt::SttEngine;

// ── Raw FFI ──────────────────────────────────────────────────────────

#[allow(non_camel_case_types)]
enum vox_ctx_t {}
#[allow(non_camel_case_types)]
enum vox_stream_t {}

unsafe extern "C" {
    fn vox_load(model_dir: *const c_char) -> *mut vox_ctx_t;
    fn vox_free(ctx: *mut vox_ctx_t);
    fn vox_transcribe_audio(
        ctx: *mut vox_ctx_t,
        samples: *const c_float,
        n_samples: c_int,
    ) -> *mut c_char;
    fn free(ptr: *mut c_void);

    #[cfg(target_os = "macos")]
    fn vox_metal_init() -> c_int;

    // Streaming API
    fn vox_stream_init(ctx: *mut vox_ctx_t) -> *mut vox_stream_t;
    fn vox_stream_feed(s: *mut vox_stream_t, samples: *const c_float, n: c_int) -> c_int;
    fn vox_stream_get(s: *mut vox_stream_t, out: *mut *const c_char, max: c_int) -> c_int;
    fn vox_stream_finish(s: *mut vox_stream_t) -> c_int;
    fn vox_stream_flush(s: *mut vox_stream_t) -> c_int;
    fn vox_stream_free(s: *mut vox_stream_t);
    fn vox_set_processing_interval(s: *mut vox_stream_t, seconds: c_float);
}

// ── Internal state ───────────────────────────────────────────────────

struct VoxtralInner {
    ctx: *mut vox_ctx_t,
    stream: Option<*mut vox_stream_t>,
}

/// Drain all pending tokens from a stream into a String.
fn drain_tokens(stream: *mut vox_stream_t) -> String {
    let mut out = String::new();
    let mut tokens: [*const c_char; 64] = [std::ptr::null(); 64];
    loop {
        let n = unsafe { vox_stream_get(stream, tokens.as_mut_ptr(), 64) };
        if n <= 0 {
            break;
        }
        for &tok in &tokens[..n as usize] {
            if !tok.is_null() {
                if let Ok(s) = unsafe { CStr::from_ptr(tok) }.to_str() {
                    out.push_str(s);
                }
            }
        }
    }
    out
}

// ── Safe wrapper ─────────────────────────────────────────────────────

pub struct VoxtralEngine {
    inner: Mutex<VoxtralInner>,
}

// Safety: VoxtralInner is only accessed through the Mutex.
unsafe impl Send for VoxtralEngine {}
unsafe impl Sync for VoxtralEngine {}

impl VoxtralEngine {
    pub fn new(model_dir: &str) -> Result<Self> {
        #[cfg(target_os = "macos")]
        {
            let ok = unsafe { vox_metal_init() };
            if ok != 0 {
                eprintln!("Voxtral: Metal GPU acceleration enabled.");
            } else {
                eprintln!("Voxtral: Metal unavailable, using CPU.");
            }
        }

        // Preload Q8 weights into GPU memory to avoid page-fault overhead
        // during streaming (trades higher RSS for zero-latency weight access).
        unsafe { std::env::set_var("VOX_Q8_PRELOAD", "1") };

        let c_dir =
            CString::new(model_dir).context("model_dir contains interior NUL")?;
        let ctx = unsafe { vox_load(c_dir.as_ptr()) };
        if ctx.is_null() {
            anyhow::bail!(
                "vox_load returned NULL — failed to load model from '{}'",
                model_dir
            );
        }
        Ok(Self {
            inner: Mutex::new(VoxtralInner {
                ctx,
                stream: None,
            }),
        })
    }
}

impl Drop for VoxtralEngine {
    fn drop(&mut self) {
        let inner = self.inner.lock().unwrap();
        if let Some(s) = inner.stream {
            unsafe { vox_stream_free(s) };
        }
        if !inner.ctx.is_null() {
            unsafe { vox_free(inner.ctx) };
        }
    }
}

impl SttEngine for VoxtralEngine {
    fn transcribe(&self, audio: &[f32]) -> Result<String> {
        let inner = self.inner.lock().unwrap();
        let ptr = unsafe {
            vox_transcribe_audio(inner.ctx, audio.as_ptr(), audio.len() as c_int)
        };
        if ptr.is_null() {
            anyhow::bail!("vox_transcribe_audio returned NULL");
        }
        let text = unsafe { CStr::from_ptr(ptr) }
            .to_str()
            .context("voxtral output is not valid UTF-8")?
            .to_owned();
        unsafe { free(ptr as *mut c_void) };
        Ok(text)
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn stream_start(&self) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        // Clean up any leftover stream
        if let Some(s) = inner.stream.take() {
            unsafe { vox_stream_free(s) };
        }
        let s = unsafe { vox_stream_init(inner.ctx) };
        if s.is_null() {
            anyhow::bail!("vox_stream_init returned NULL");
        }
        // Lower processing interval for responsive streaming (default 2.0s is sluggish).
        // 0.5s balances latency vs GPU overhead. Lower = more frequent encoder runs.
        unsafe { vox_set_processing_interval(s, 0.5) };
        inner.stream = Some(s);
        Ok(())
    }

    fn stream_feed(&self, audio: &[f32]) -> Result<String> {
        let inner = self.inner.lock().unwrap();
        let s = inner.stream.context("stream_feed called without active stream")?;
        let rc = unsafe { vox_stream_feed(s, audio.as_ptr(), audio.len() as c_int) };
        if rc != 0 {
            anyhow::bail!("vox_stream_feed returned {}", rc);
        }
        Ok(drain_tokens(s))
    }

    fn stream_flush(&self) -> Result<String> {
        let inner = self.inner.lock().unwrap();
        let s = inner.stream.context("stream_flush called without active stream")?;
        let _ = unsafe { vox_stream_flush(s) };
        Ok(drain_tokens(s))
    }

    fn stream_finish(&self) -> Result<String> {
        let mut inner = self.inner.lock().unwrap();
        let s = match inner.stream.take() {
            Some(s) => s,
            None => return Ok(String::new()),
        };
        let _ = unsafe { vox_stream_finish(s) };
        let text = drain_tokens(s);
        unsafe { vox_stream_free(s) };
        Ok(text)
    }
}
