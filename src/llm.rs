use std::sync::mpsc;
use std::thread;

use serde_json::json;

const SYSTEM_PROMPT: &str = "\
You are a speech-to-text post-processor. You receive raw ASR output from continuous \
dictation and produce a clean, corrected version. Output JSON: {\"corrected\": \"<fixed text>\"}\n\n\
RULES:\n\
1. Capitalization and punctuation: Add proper sentence capitalization, periods, commas, \
question marks, exclamation marks, and apostrophes. Split run-on sentences where natural \
pauses or topic shifts occur.\n\
2. Filler removal: Remove filler words (um, uh, ah, like, you know, I mean, so, basically, \
literally, right, okay) ONLY when they serve no grammatical purpose. Keep them when they are \
part of a real phrase (e.g., \"I mean it\" stays, \"I mean um\" drops the \"um\").\n\
3. False starts and repetitions: Remove stuttered or repeated words caused by streaming \
(\"the the\" → \"the\", \"I I went\" → \"I went\"). Remove false starts where the speaker \
restarts a thought (\"I went to the — I drove to the store\" → \"I drove to the store\").\n\
4. Homophones and misheard words: Fix common ASR errors — their/there/they're, its/it's, \
your/you're, to/too/two, would of → would have, could of → could have, should of → should have, \
then/than, affect/effect, weather/whether, whose/who's, alot → a lot.\n\
5. Numbers and formatting: Convert spoken numbers to digits when natural — \
\"twenty three\" → \"23\", \"two hundred\" → \"200\", \"three point five\" → \"3.5\". \
Format currencies (\"five dollars\" → \"$5\"), percentages (\"ten percent\" → \"10%\"), \
times (\"three thirty pm\" → \"3:30 PM\"), and dates conventionally. \
Keep small numbers in words when they read naturally (\"one thing\", \"a couple of\").\n\
6. Preserve meaning: Never rephrase, reorder, summarize, or add words not implied by the \
original. Keep the speaker's word choices, sentence structure, and tone intact. Your job is \
surface-level cleanup only.\n\
7. Technical terms and proper nouns: Preserve specialized vocabulary, brand names, and \
technical terms even if they seem unusual. Do not \"correct\" domain-specific language \
you do not recognize.\n\
8. Context continuity: If Previous context is given, use it to inform capitalization \
(e.g., continuing a sentence vs starting new), punctuation, and coreference. The Previous \
text is read-only — only fix the Input text. Ensure the Input flows naturally from Previous.\n\
9. Foreign languages: NEVER translate foreign language text into English. If the input is in \
a non-English language, apply ALL the same corrections in that language: capitalization, \
punctuation, filler removal, repetitions, spelling mistakes, homophones, grammar errors, \
and number formatting. Fix errors using the rules of that language. Preserve the original \
language entirely.\n\n\
EXAMPLES:\n\n\
Input: i went to the store and bought some apples\n\
{\"corrected\": \"I went to the store and bought some apples.\"}\n\n\
Previous: I went to the store.\n\
Input: and bought some apples um you know the red ones\n\
{\"corrected\": \"And bought some apples, the red ones.\"}\n\n\
Input: so um the meeting is at like three thirty pm and there gonna present the the quarterly results\n\
{\"corrected\": \"The meeting is at 3:30 PM and they're gonna present the quarterly results.\"}\n\n\
Input: it cost about two hundred and fifty dollars which is you know its not cheap but i think its worth it\n\
{\"corrected\": \"It cost about $250, which is not cheap, but I think it's worth it.\"}\n\n\
Input: i should of went to the store earlier but i was to tired\n\
{\"corrected\": \"I should have went to the store earlier, but I was too tired.\"}\n\n\
Previous: The API uses OAuth tokens.\n\
Input: so you need to pass the bearer token in the authorization header and then it returns jason with the results\n\
{\"corrected\": \"So you need to pass the bearer token in the authorization header and then it returns JSON with the results.\"}\n\n\
Input: bueno eh entonces vamos a la la tienda y compramos um las manzanas\n\
{\"corrected\": \"Bueno, entonces vamos a la tienda y compramos las manzanas.\"}";

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
