# Screen Vocabulary Extraction via macOS Accessibility API

## Goal

Extract text from the currently focused window when dictation activates, derive a vocabulary list of technical/domain-specific terms, and inject it into the LLM prompt so speech-to-text corrections resolve ambiguous words correctly (e.g., "cube control" -> "kubectl", "tow key oh" -> "tokio", "ax you i element" -> "AXUIElement").

## Why Accessibility API over Screenshot + OCR

| | Accessibility API | Screenshot + OCR |
|---|---|---|
| Accuracy | Exact text, no OCR errors | OCR can misread code |
| Speed | ~1-50ms for focused element | ~1-2s (capture + Vision) |
| Dependencies | System framework only | Swift helper or Vision FFI |
| Permissions | Already granted for enigo | Same permission |
| Code editors | Gets full source text | Only visible pixels |

## Architecture

```
Wake word detected
       |
       v
  [Background thread]
       |
       +---> AXUIElementCreateSystemWide()
       +---> Get focused app -> focused window
       +---> Walk AX tree (depth-limited) collecting text
       +---> Extract vocabulary (filter common words, keep technical terms)
       +---> Store in Arc<Mutex<Option<String>>>
       |
  [Main thread continues immediately - dictation not blocked]
       |
       v
  [LLM requests]
       +---> If vocabulary is ready, prepend to prompt
       +---> cache_prompt: true -> vocabulary cached after first call, zero cost after
```

## New Module: `src/screen.rs`

### Dependencies

```toml
# Cargo.toml additions
accessibility-sys = "0.1"    # Complete AXUIElement FFI bindings (links ApplicationServices)
core-foundation = "0.10"     # CFString, CFArray, CFType handling
```

No new permissions needed -- enigo already requires macOS Accessibility permission, which is the same permission AXUIElement queries use.

### Public API

```rust
/// Extracted vocabulary from the screen, ready to inject into LLM prompt.
pub struct ScreenVocabulary {
    pub terms: Vec<String>,       // unique technical terms, max 150
    pub window_title: String,     // e.g. "main.rs - voicer - Visual Studio Code"
}

/// Capture vocabulary from the focused window.
/// Blocks for up to ~500ms. Call from a background thread.
pub fn capture_vocabulary() -> Option<ScreenVocabulary>
```

### Text Extraction Strategy

Three tiers, tried in order:

#### Tier 1: Focused element (fastest, ~1-5ms)

```
SystemWide -> kAXFocusedApplicationAttribute -> App
App -> kAXFocusedUIElementAttribute -> Element
Element -> kAXValueAttribute -> text (if AXTextArea / AXTextField)
```

Single IPC call. Covers ~80% of use cases: active code editor, terminal, text input.

#### Tier 2: Window title (~1ms extra)

Also grab `kAXTitleAttribute` from the focused window. Titles contain useful context:
- VS Code: `"main.rs - voicer - Visual Studio Code"`
- Chrome: `"Kubernetes Documentation - Google Chrome"`
- Terminal: `"sho@mac: ~/fun/voicer"`

#### Tier 3: Shallow tree walk (fallback, ~10-200ms)

If focused element has no `kAXValueAttribute`, walk the window's children collecting text from text-bearing roles.

```rust
const MAX_DEPTH: usize = 7;
const MAX_ELEMENTS: usize = 500;
const MAX_TEXT_BYTES: usize = 100_000; // 100KB per element, skip larger
const TEXT_ROLES: &[&str] = &["AXTextArea", "AXTextField", "AXStaticText"];
```

Use `kAXVisibleChildrenAttribute` (instead of `kAXChildrenAttribute`) where supported, to skip off-screen elements and reduce tree size.

### Implementation

#### Core: AX helper functions

```rust
use accessibility_sys::*;
use core_foundation::base::{CFRelease, CFTypeRef, TCFType};
use core_foundation::string::{CFString, CFStringRef};
use core_foundation::array::{CFArray, CFArrayRef};

unsafe fn ax_get_attr(element: AXUIElementRef, attr: &str) -> Option<CFTypeRef> {
    let mut value: CFTypeRef = std::ptr::null();
    let cf_attr = CFString::new(attr);
    let err = AXUIElementCopyAttributeValue(
        element,
        cf_attr.as_concrete_TypeRef(),
        &mut value,
    );
    if err != kAXErrorSuccess || value.is_null() {
        return None;
    }
    Some(value)
}

unsafe fn ax_get_string(element: AXUIElementRef, attr: &str) -> Option<String> {
    let value = ax_get_attr(element, attr)?;
    let cf_str = CFString::wrap_under_create_rule(value as CFStringRef);
    Some(cf_str.to_string())
}

unsafe fn ax_get_children(element: AXUIElementRef, attr: &str) -> Option<Vec<AXUIElementRef>> {
    let value = ax_get_attr(element, attr)?;
    let arr = CFArray::wrap_under_create_rule(value as CFArrayRef);
    let mut result = Vec::new();
    for i in 0..arr.len() {
        result.push(*arr.get(i).unwrap() as AXUIElementRef);
    }
    Some(result)
}
```

#### Core: capture_vocabulary()

```rust
pub fn capture_vocabulary() -> Option<ScreenVocabulary> {
    unsafe {
        // Check permission
        if !AXIsProcessTrusted() {
            eprintln!("Accessibility permission not granted, skipping screen vocabulary");
            return None;
        }

        let sys = AXUIElementCreateSystemWide();

        // Get focused app
        let app = ax_get_attr(sys, "AXFocusedApplication")? as AXUIElementRef;

        // Get focused window + title
        let window = ax_get_attr(app, "AXFocusedWindow")? as AXUIElementRef;
        let title = ax_get_string(window, "AXTitle").unwrap_or_default();

        let mut texts: Vec<String> = Vec::new();

        // Tier 1: Fast path -- focused element's value
        if let Some(focused) = ax_get_attr(app, "AXFocusedUIElement") {
            let focused = focused as AXUIElementRef;
            let role = ax_get_string(focused, "AXRole").unwrap_or_default();
            if TEXT_ROLES.contains(&role.as_str()) {
                if let Some(value) = ax_get_string(focused, "AXValue") {
                    texts.push(value);
                }
            }
        }

        // Tier 2: Add window title
        if !title.is_empty() {
            texts.push(title.clone());
        }

        // Tier 3: Tree walk if we got nothing substantial
        if texts.iter().map(|t| t.len()).sum::<usize>() < 50 {
            let mut count = 0;
            walk_ax_tree(window, &mut texts, 0, MAX_DEPTH, &mut count);
        }

        // Tier 3b: Electron apps -- enable AX tree and retry
        if texts.iter().map(|t| t.len()).sum::<usize>() < 50 {
            let key = CFString::new("AXManualAccessibility");
            AXUIElementSetAttributeValue(app, key.as_concrete_TypeRef(), kCFBooleanTrue as _);
            let mut count = 0;
            walk_ax_tree(window, &mut texts, 0, MAX_DEPTH, &mut count);
        }

        let terms = extract_terms(&texts);

        // Cleanup
        CFRelease(window as _);
        CFRelease(app as _);
        CFRelease(sys as _);

        Some(ScreenVocabulary { terms, window_title: title })
    }
}
```

#### Core: walk_ax_tree()

```rust
unsafe fn walk_ax_tree(
    element: AXUIElementRef,
    texts: &mut Vec<String>,
    depth: usize,
    max_depth: usize,
    count: &mut usize,
) {
    if depth > max_depth || *count > MAX_ELEMENTS {
        return;
    }
    *count += 1;

    let role = ax_get_string(element, "AXRole").unwrap_or_default();

    if TEXT_ROLES.contains(&role.as_str()) {
        if let Some(value) = ax_get_string(element, "AXValue") {
            if !value.is_empty() && value.len() < MAX_TEXT_BYTES {
                texts.push(value);
            }
        }
        return; // Don't recurse into text elements
    }

    // Prefer visible children
    let children = ax_get_children(element, "AXVisibleChildren")
        .or_else(|| ax_get_children(element, "AXChildren"));

    if let Some(children) = children {
        for child in children {
            walk_ax_tree(child, texts, depth + 1, max_depth, count);
        }
    }
}
```

### Vocabulary Filtering

From the raw text, extract tokens that look like code identifiers or technical terms.

#### What to keep

| Pattern | Examples |
|---------|----------|
| CamelCase / PascalCase | `useState`, `AsyncRead`, `GraphQL` |
| snake_case | `async_trait`, `read_to_string`, `pending_raw` |
| SCREAMING_CASE | `MAX_DEPTH`, `VAD_THRESHOLD` |
| Contains digits | `u32`, `f64`, `Vec3`, `h264` |
| Dot-paths / namespaces | `std::sync::Arc` -> segments `std`, `sync`, `Arc` |
| Abbreviations (3+ uppercase) | `API`, `LLM`, `ONNX`, `gRPC` |
| Uncommon words | Not in top ~500 common English words |

#### What to discard

- Top ~500 most common English words ("the", "and", "have", "with", ...)
- Single characters
- Pure numbers
- Very long tokens (> 40 chars, likely paths or URLs)

#### Implementation

```rust
use std::collections::HashSet;

const COMMON_WORDS: &[&str] = &[
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their",
    "what", "so", "up", "out", "if", "about", "who", "get", "which",
    "go", "me", "when", "make", "can", "like", "time", "no", "just",
    "him", "know", "take", "people", "into", "year", "your", "good",
    "some", "could", "them", "see", "other", "than", "then", "now",
    "look", "only", "come", "its", "over", "think", "also", "back",
    "after", "use", "two", "how", "our", "work", "first", "well",
    "way", "even", "new", "want", "because", "any", "these", "give",
    "day", "most", "us", "is", "are", "was", "were", "been", "being",
    "has", "had", "did", "does", "doing", "will", "shall", "should",
    "may", "might", "must", "need", "let", "here", "where", "why",
    "how", "each", "every", "both", "few", "more", "much", "many",
    "such", "own", "same", "very", "still", "also", "just", "should",
    // ... extend to ~500 words
];

fn is_technical(token: &str) -> bool {
    // Contains underscore (snake_case)
    if token.contains('_') { return true; }
    // Contains digit
    if token.chars().any(|c| c.is_ascii_digit()) { return true; }
    // CamelCase: has lowercase followed by uppercase
    let chars: Vec<char> = token.chars().collect();
    for w in chars.windows(2) {
        if w[0].is_lowercase() && w[1].is_uppercase() { return true; }
    }
    // ALLCAPS with 3+ letters
    if token.len() >= 3 && token.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit()) {
        return true;
    }
    false
}

pub fn extract_terms(texts: &[String]) -> Vec<String> {
    let common: HashSet<&str> = COMMON_WORDS.iter().copied().collect();
    let mut seen = HashSet::new();
    let mut terms = Vec::new();

    for text in texts {
        for token in text.split(|c: char| c.is_whitespace() || "(){}[]<>,;:\"'`#@!?/\\|=+*&^%$~".contains(c)) {
            let token = token.trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '-');
            if token.len() < 2 || token.len() > 40 { continue; }

            let lower = token.to_lowercase();
            if common.contains(lower.as_str()) && !is_technical(token) { continue; }

            if seen.insert(lower) {
                terms.push(token.to_string());
            }
        }
    }

    // Cap at 150 terms
    terms.truncate(150);
    terms
}
```

## Integration with Existing Code

### `src/main.rs`

Add shared vocabulary state and background capture on activation:

```rust
use std::sync::{Arc, Mutex};

// After LLM setup:
let screen_vocab: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
```

On activation (wake word confirmed):

```rust
Ok(true) => {
    // ... existing activation code ...

    // Capture screen vocabulary in background
    let vocab_handle = screen_vocab.clone();
    std::thread::spawn(move || {
        match screen::capture_vocabulary() {
            Some(sv) => {
                let vocab_str = sv.terms.join(", ");
                eprintln!("Screen vocab: {} terms from \"{}\"", sv.terms.len(), sv.window_title);
                *vocab_handle.lock().unwrap() = Some(vocab_str);
            }
            None => {
                eprintln!("Screen vocab: could not extract");
                *vocab_handle.lock().unwrap() = Some(String::new());
            }
        }
    });
}
```

When sending LLM requests:

```rust
if let Some(ref llm_handle) = llm {
    let raw_text = pending_raw.join(" ");
    let hint = tail_chars(&finalized, 80);
    let vocab = screen_vocab.lock().unwrap().clone().unwrap_or_default();
    llm_handle.request(llm_seq, hint, &raw_text, &vocab);
    last_sent_count = pending_raw.len();
    llm_seq += 1;
}
```

On deactivation:

```rust
*screen_vocab.lock().unwrap() = None;
```

### `src/llm.rs`

Add vocabulary to request type and prompt construction:

```rust
/// A request to the LLM thread: (seq, context_hint, raw_text, vocabulary)
type LlmRequest = (u64, String, String, String);

impl LlmHandle {
    pub fn request(&self, seq: u64, context_hint: &str, raw_text: &str, vocabulary: &str) {
        let _ = self.req_tx.send((
            seq,
            context_hint.to_string(),
            raw_text.to_string(),
            vocabulary.to_string(),
        ));
    }
}
```

In `llm_thread`, prompt construction:

```rust
let (seq, ref context_hint, ref raw_text, ref vocabulary) = current;

let mut prompt = SYSTEM_PROMPT.to_string();

if !vocabulary.is_empty() {
    prompt.push_str("\n\nVocabulary on screen: ");
    prompt.push_str(vocabulary);
}

prompt.push_str("\n\n");
if !context_hint.is_empty() {
    prompt.push_str(&format!("Previous: {}\n", context_hint));
}
prompt.push_str(&format!("Input: {}\nJSON:", raw_text));
```

The vocabulary section sits between the static system prompt and the per-request input. With `cache_prompt: true`, everything up to the `\n\n` before `Previous:`/`Input:` gets cached after the first call. Vocabulary adds ~50-100ms to the first call only.

## App-Specific Behavior

| App | AX behavior | What we get |
|-----|-------------|-------------|
| **Terminal.app** | AXTextArea with full visible buffer | Terminal output, commands |
| **iTerm2** | AXTextArea, native Cocoa app | Terminal text (visible portion) |
| **VS Code** | Electron -- AX tree disabled by default. Set `AXManualAccessibility` to enable. Lines exposed as AXStaticText children | Source code lines |
| **Xcode** | Native AppKit, excellent AX support. AXTextArea for editor | Full source code |
| **Safari** | AXWebArea root, AXStaticText children for page text | Webpage text |
| **Chrome** | AX disabled by default. Set `AXEnhancedUserInterface` on window to enable | Webpage text |
| **Slack (Electron)** | Same as VS Code -- `AXManualAccessibility` | Chat messages |
| **Finder** | AXStaticText for file names | File names in current view |

## Performance Budget

| Step | Time | Notes |
|------|------|-------|
| Tier 1: focused element | ~1-5ms | Single IPC call |
| Tier 2: window title | ~1ms | One extra attribute fetch |
| Tier 3: tree walk (native) | ~10-50ms | Up to 500 elements |
| Tier 3b: Electron enable + walk | ~50-200ms | One-time AX tree construction |
| Browser tree walk | ~100-500ms | Large DOM, depth-limited |
| Vocabulary extraction | ~1-5ms | String processing |
| **Total (typical)** | **~5-50ms** | Background thread, non-blocking |
| **Total (worst case)** | **~500ms** | Browser/Electron first time |

Runs once per activation in a background thread. First LLM request may not have vocabulary yet if capture is slow -- graceful degradation (empty vocab string is fine). All subsequent requests in the session benefit.

## Error Handling

1. **Permission denied**: `AXIsProcessTrusted()` returns false. Log warning, return None. Enigo should have already prompted.
2. **No focused window**: Return None. App works fine without vocabulary.
3. **Electron app, tree empty**: Try `AXManualAccessibility`, retry once. If still empty, return what we have (window title).
4. **Unresponsive app**: AX calls can hang. Set timeout via `AXUIElementSetMessagingTimeout(element, 0.5)` (500ms). Abandon if exceeded.
5. **Huge text**: Cap individual chunks at 100KB. Skip elements with `value.len() > 100_000`.
6. **No text anywhere**: Return empty terms vec. LLM prompt omits vocabulary line. No degradation.

## Files to Create / Modify

| File | Change |
|------|--------|
| `Cargo.toml` | Add `accessibility-sys = "0.1"`, `core-foundation = "0.10"` |
| `src/screen.rs` | **New** -- `capture_vocabulary()`, AX tree walker, `extract_terms()` |
| `src/main.rs` | Add `mod screen`, `Arc<Mutex<Option<String>>>` for vocab, background capture on activation, pass vocab to LLM, clear on deactivation |
| `src/llm.rs` | Add `vocabulary` field to `LlmRequest`, include in prompt construction |

## Example: How Vocabulary Improves Corrections

### Without vocabulary

```
Input: and then i used the ax you i element to get the text
{"corrected": "And then I used the AX UI element to get the text."}
```

### With vocabulary: `AXUIElement, CFString, kAXValueAttribute, ...`

```
Input: and then i used the ax you i element to get the text
{"corrected": "And then I used the AXUIElement to get the text."}
```

### Without vocabulary

```
Input: i need to configure the cube control cluster
{"corrected": "I need to configure the cube control cluster."}
```

### With vocabulary: `kubectl, Kubernetes, kube-system, ...`

```
Input: i need to configure the cube control cluster
{"corrected": "I need to configure the kubectl cluster."}
```

## Testing

1. **Terminal**: Activate with Terminal focused. Check stderr for extracted terms. Should include commands, paths.
2. **VS Code**: Open a Rust file, activate. Should extract function names, type names, crate names.
3. **Chrome**: Open a technical docs page, activate. Should extract headings and terms.
4. **No window**: Minimize everything, activate. Should return empty vocab gracefully, no crash.
5. **Dictation quality**: Dictate technical terms visible on screen. Compare correction with/without vocab hints.
