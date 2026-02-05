use std::sync::mpsc;
use std::thread;

use serde_json::json;

const SYSTEM_PROMPT: &str = "\
You fix speech-to-text output from continuous dictation.\n\
Output JSON: {\"corrected\": \"<fixed text>\"}\n\
Fix capitalization, punctuation, and spacing.\n\
Remove filler words (um, uh), false starts, and ASR artifacts like \"...\".\n\
Do not paraphrase or add new content.\n\
If Previous context is given, it is read-only — only fix the Input text.\n\n\
Example 1:\n\
Input: i went to the store and bought some apples\n\
{\"corrected\": \"I went to the store and bought some apples.\"}\n\n\
Example 2:\n\
Previous: I went to the store.\n\
Input: and bought some apples um you know the red ones\n\
{\"corrected\": \"And bought some apples, you know, the red ones.\"}";

/// A request to the LLM thread: (seq, context_hint, raw_text)
type LlmRequest = (u64, String, String);

pub struct LlmHandle {
    req_tx: mpsc::Sender<LlmRequest>,
    resp_rx: mpsc::Receiver<(u64, String)>,
}

impl LlmHandle {
    /// Probe the llama.cpp server and spawn the background thread.
    /// Returns `None` if the server is not reachable.
    pub fn spawn(server_url: &str) -> Option<Self> {
        let health_url = format!("{}/health", server_url);
        match ureq::get(&health_url).call() {
            Ok(_) => eprintln!("LLM server reachable at {}", server_url),
            Err(e) => {
                eprintln!("LLM server not reachable at {} ({}) — running without LLM post-processing", server_url, e);
                return None;
            }
        }

        let (req_tx, req_rx) = mpsc::channel::<LlmRequest>();
        let (resp_tx, resp_rx) = mpsc::channel::<(u64, String)>();
        let url = format!("{}/completion", server_url);

        thread::spawn(move || {
            llm_thread(url, req_rx, resp_tx);
        });

        Some(Self { req_tx, resp_rx })
    }

    /// Send a non-blocking request to the LLM thread.
    /// `context_hint` is the tail of finalized text (read-only, for continuity).
    pub fn request(&self, seq: u64, context_hint: &str, raw_text: &str) {
        let _ = self.req_tx.send((seq, context_hint.to_string(), raw_text.to_string()));
    }

    /// Non-blocking check for a response from the LLM thread.
    pub fn try_recv(&self) -> Option<(u64, String)> {
        self.resp_rx.try_recv().ok()
    }
}

fn llm_thread(
    url: String,
    rx: mpsc::Receiver<LlmRequest>,
    tx: mpsc::Sender<(u64, String)>,
) {
    let mut pending: Option<LlmRequest> = None;

    loop {
        let mut current = match pending.take() {
            Some(p) => p,
            None => match rx.recv() {
                Ok(msg) => msg,
                Err(_) => return,
            },
        };

        // Drain to the latest request
        while let Ok(newer) = rx.try_recv() {
            current = newer;
        }

        let (seq, ref context_hint, ref raw_text) = current;

        let prompt = if context_hint.is_empty() {
            format!("{}\n\nInput: {}\nJSON:", SYSTEM_PROMPT, raw_text)
        } else {
            format!("{}\n\nPrevious: {}\nInput: {}\nJSON:", SYSTEM_PROMPT, context_hint, raw_text)
        };

        let body = json!({
            "prompt": prompt,
            "n_predict": 512,
            "temperature": 0.0,
            "top_k": 1,
            "cache_prompt": true,
            "id_slot": 0,
            "stop": ["\nInput:", "\n\n"]
        });

        let result = ureq::post(&url).send_json(&body);

        // Check if a newer request arrived while we were blocking on HTTP
        if let Ok(newer) = rx.try_recv() {
            pending = Some(newer);
            continue;
        }

        let response = match result {
            Ok(resp) => resp,
            Err(e) => {
                eprintln!("LLM request failed: {}", e);
                continue;
            }
        };

        let resp_body: serde_json::Value = match response.into_body().read_json() {
            Ok(v) => v,
            Err(e) => {
                eprintln!("LLM response parse error: {}", e);
                continue;
            }
        };

        let content = resp_body["content"]
            .as_str()
            .unwrap_or("")
            .trim();

        let corrected = extract_corrected(content);
        eprintln!("LLM raw: {}", content);
        eprintln!("LLM parsed: {}", corrected);

        if corrected.is_empty() {
            eprintln!("LLM returned empty correction, skipping");
            continue;
        }

        if corrected.len() > raw_text.len() * 2 + 50 {
            eprintln!("LLM response too long ({} vs {}), discarding", corrected.len(), raw_text.len());
            continue;
        }

        let _ = tx.send((seq, corrected));
    }
}

/// Extract the "corrected" field from LLM JSON output.
/// Handles partial JSON, missing closing brace, etc.
fn extract_corrected(content: &str) -> String {
    // Try parsing as complete JSON first
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(content) {
        if let Some(s) = v["corrected"].as_str() {
            return s.trim().to_string();
        }
    }

    // Try with closing brace appended (stop token may have eaten it)
    let with_brace = format!("{}}}", content.trim_end_matches('}'));
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&with_brace) {
        if let Some(s) = v["corrected"].as_str() {
            return s.trim().to_string();
        }
    }

    // Fallback: extract between "corrected": " and the last "
    if let Some(start) = content.find("\"corrected\"") {
        let after_key = &content[start + 11..];
        // Find the opening quote of the value
        if let Some(q1) = after_key.find('"') {
            let value_start = q1 + 1;
            let rest = &after_key[value_start..];
            // Find closing quote (handle escaped quotes)
            let mut end = 0;
            let bytes = rest.as_bytes();
            while end < bytes.len() {
                if bytes[end] == b'"' && (end == 0 || bytes[end - 1] != b'\\') {
                    break;
                }
                end += 1;
            }
            let extracted = &rest[..end];
            // Unescape basic JSON escapes
            return extracted
                .replace("\\\"", "\"")
                .replace("\\n", "\n")
                .replace("\\\\", "\\")
                .trim()
                .to_string();
        }
    }

    eprintln!("LLM: could not parse JSON from: {}", content);
    String::new()
}
