use crate::ProviderRuntimeOptions;
use crate::auth::AuthService;
use crate::auth::openai_oauth::extract_account_id_from_jwt;
use crate::multimodal;
use crate::traits::{
    ChatMessage, ChatRequest as ProviderChatRequest, Provider, ProviderCapabilities, StreamChunk,
    StreamError, StreamEvent, StreamOptions, StreamResult,
};
use async_trait::async_trait;
use futures_util::{StreamExt, stream};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;

const DEFAULT_CODEX_RESPONSES_URL: &str = "https://chatgpt.com/backend-api/codex/responses";
const CODEX_RESPONSES_URL_ENV: &str = "ZEROCLAW_CODEX_RESPONSES_URL";
const CODEX_BASE_URL_ENV: &str = "ZEROCLAW_CODEX_BASE_URL";
const DEFAULT_CODEX_INSTRUCTIONS: &str =
    "You are ZeroClaw, a concise and helpful coding assistant.";

pub struct OpenAiCodexProvider {
    auth: AuthService,
    auth_profile_override: Option<String>,
    responses_url: String,
    custom_endpoint: bool,
    gateway_api_key: Option<String>,
    reasoning_effort: Option<String>,
    client: Client,
}

#[derive(Debug, Serialize)]
struct ResponsesRequest {
    model: String,
    input: Vec<ResponsesInput>,
    instructions: String,
    store: bool,
    stream: bool,
    text: ResponsesTextOptions,
    reasoning: ResponsesReasoningOptions,
    include: Vec<String>,
    tool_choice: String,
    parallel_tool_calls: bool,
}

#[derive(Debug, Serialize)]
struct ResponsesInput {
    role: String,
    content: Vec<ResponsesInputContent>,
}

#[derive(Debug, Serialize)]
struct ResponsesInputContent {
    #[serde(rename = "type")]
    kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_url: Option<String>,
}

#[derive(Debug, Serialize)]
struct ResponsesTextOptions {
    verbosity: String,
}

#[derive(Debug, Serialize)]
struct ResponsesReasoningOptions {
    effort: String,
    summary: String,
}

#[derive(Debug, Deserialize)]
struct ResponsesResponse {
    #[serde(default)]
    output: Vec<ResponsesOutput>,
    #[serde(default)]
    output_text: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct ResponsesOutput {
    #[serde(default)]
    content: Vec<ResponsesContent>,
}

#[derive(Debug, Clone, Deserialize)]
struct ResponsesContent {
    #[serde(rename = "type")]
    kind: Option<String>,
    text: Option<String>,
}

impl OpenAiCodexProvider {
    pub fn new(
        options: &ProviderRuntimeOptions,
        gateway_api_key: Option<&str>,
    ) -> anyhow::Result<Self> {
        let state_dir = options
            .zeroclaw_dir
            .clone()
            .unwrap_or_else(default_zeroclaw_dir);
        let auth = AuthService::new(&state_dir, options.secrets_encrypt);
        let responses_url = resolve_responses_url(options)?;

        Ok(Self {
            auth,
            auth_profile_override: options.auth_profile_override.clone(),
            custom_endpoint: !is_default_responses_url(&responses_url),
            responses_url,
            gateway_api_key: gateway_api_key.map(ToString::to_string),
            reasoning_effort: options.reasoning_effort.clone(),
            client: Client::builder()
                .connect_timeout(std::time::Duration::from_secs(10))
                .read_timeout(std::time::Duration::from_secs(300))
                .build()
                .unwrap_or_else(|_| Client::new()),
        })
    }
}

fn default_zeroclaw_dir() -> PathBuf {
    directories::UserDirs::new().map_or_else(
        || PathBuf::from(".zeroclaw"),
        |dirs| dirs.home_dir().join(".zeroclaw"),
    )
}

fn build_responses_url(base_or_endpoint: &str) -> anyhow::Result<String> {
    let candidate = base_or_endpoint.trim();
    if candidate.is_empty() {
        anyhow::bail!("OpenAI Codex endpoint override cannot be empty");
    }

    let mut parsed = reqwest::Url::parse(candidate)
        .map_err(|_| anyhow::anyhow!("OpenAI Codex endpoint override must be a valid URL"))?;

    match parsed.scheme() {
        "http" | "https" => {}
        _ => anyhow::bail!("OpenAI Codex endpoint override must use http:// or https://"),
    }

    let path = parsed.path().trim_end_matches('/');
    if !path.ends_with("/responses") {
        let with_suffix = if path.is_empty() || path == "/" {
            "/responses".to_string()
        } else {
            format!("{path}/responses")
        };
        parsed.set_path(&with_suffix);
    }

    parsed.set_query(None);
    parsed.set_fragment(None);

    Ok(parsed.to_string())
}

fn resolve_responses_url(options: &ProviderRuntimeOptions) -> anyhow::Result<String> {
    if let Some(endpoint) = std::env::var(CODEX_RESPONSES_URL_ENV)
        .ok()
        .and_then(|value| first_nonempty(Some(&value)))
    {
        return build_responses_url(&endpoint);
    }

    if let Some(base_url) = std::env::var(CODEX_BASE_URL_ENV)
        .ok()
        .and_then(|value| first_nonempty(Some(&value)))
    {
        return build_responses_url(&base_url);
    }

    if let Some(api_url) = options
        .provider_api_url
        .as_deref()
        .and_then(|value| first_nonempty(Some(value)))
    {
        return build_responses_url(&api_url);
    }

    Ok(DEFAULT_CODEX_RESPONSES_URL.to_string())
}

fn canonical_endpoint(url: &str) -> Option<(String, String, u16, String)> {
    let parsed = reqwest::Url::parse(url).ok()?;
    let host = parsed.host_str()?.to_ascii_lowercase();
    let port = parsed.port_or_known_default()?;
    let path = parsed.path().trim_end_matches('/').to_string();
    Some((parsed.scheme().to_ascii_lowercase(), host, port, path))
}

fn is_default_responses_url(url: &str) -> bool {
    canonical_endpoint(url) == canonical_endpoint(DEFAULT_CODEX_RESPONSES_URL)
}

fn first_nonempty(text: Option<&str>) -> Option<String> {
    text.and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

#[allow(dead_code)]
fn resolve_instructions(system_prompt: Option<&str>) -> String {
    first_nonempty(system_prompt).unwrap_or_else(|| DEFAULT_CODEX_INSTRUCTIONS.to_string())
}

fn normalize_model_id(model: &str) -> &str {
    model.rsplit('/').next().unwrap_or(model)
}

fn build_responses_input(messages: &[ChatMessage]) -> (String, Vec<ResponsesInput>) {
    let mut system_parts: Vec<&str> = Vec::new();
    let mut input: Vec<ResponsesInput> = Vec::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => system_parts.push(&msg.content),
            "user" => {
                let (cleaned_text, image_refs) = multimodal::parse_image_markers(&msg.content);

                let mut content_items = Vec::new();

                // Add text if present
                if !cleaned_text.trim().is_empty() {
                    content_items.push(ResponsesInputContent {
                        kind: "input_text".to_string(),
                        text: Some(cleaned_text),
                        image_url: None,
                    });
                }

                // Add images
                for image_ref in image_refs {
                    content_items.push(ResponsesInputContent {
                        kind: "input_image".to_string(),
                        text: None,
                        image_url: Some(image_ref),
                    });
                }

                // If no content at all, add empty text
                if content_items.is_empty() {
                    content_items.push(ResponsesInputContent {
                        kind: "input_text".to_string(),
                        text: Some(String::new()),
                        image_url: None,
                    });
                }

                input.push(ResponsesInput {
                    role: "user".to_string(),
                    content: content_items,
                });
            }
            "assistant" => {
                input.push(ResponsesInput {
                    role: "assistant".to_string(),
                    content: vec![ResponsesInputContent {
                        kind: "output_text".to_string(),
                        text: Some(msg.content.clone()),
                        image_url: None,
                    }],
                });
            }
            _ => {}
        }
    }

    let instructions = if system_parts.is_empty() {
        DEFAULT_CODEX_INSTRUCTIONS.to_string()
    } else {
        system_parts.join("\n\n")
    };

    (instructions, input)
}

fn clamp_reasoning_effort(model: &str, effort: &str) -> String {
    let id = normalize_model_id(model);
    // gpt-5-codex currently supports only low|medium|high.
    if id == "gpt-5-codex" {
        return match effort {
            "low" | "medium" | "high" => effort.to_string(),
            "minimal" => "low".to_string(),
            _ => "high".to_string(),
        };
    }
    if (id.starts_with("gpt-5.2") || id.starts_with("gpt-5.3")) && effort == "minimal" {
        return "low".to_string();
    }
    if id.starts_with("gpt-5-codex") && effort == "xhigh" {
        return "high".to_string();
    }
    if id == "gpt-5.1" && effort == "xhigh" {
        return "high".to_string();
    }
    if id == "gpt-5.1-codex-mini" {
        return if effort == "high" || effort == "xhigh" {
            "high".to_string()
        } else {
            "medium".to_string()
        };
    }
    effort.to_string()
}

fn resolve_reasoning_effort(model_id: &str, configured: Option<&str>) -> String {
    let raw = configured
        .map(ToString::to_string)
        .or_else(|| std::env::var("ZEROCLAW_CODEX_REASONING_EFFORT").ok())
        .and_then(|value| first_nonempty(Some(&value)))
        .unwrap_or_else(|| "xhigh".to_string())
        .to_ascii_lowercase();
    clamp_reasoning_effort(model_id, &raw)
}

fn nonempty_preserve(text: Option<&str>) -> Option<String> {
    text.and_then(|value| {
        if value.is_empty() {
            None
        } else {
            Some(value.to_string())
        }
    })
}

fn extract_responses_text(response: &ResponsesResponse) -> Option<String> {
    if let Some(text) = first_nonempty(response.output_text.as_deref()) {
        return Some(text);
    }

    for item in &response.output {
        for content in &item.content {
            if content.kind.as_deref() == Some("output_text")
                && let Some(text) = first_nonempty(content.text.as_deref())
            {
                return Some(text);
            }
        }
    }

    for item in &response.output {
        for content in &item.content {
            if let Some(text) = first_nonempty(content.text.as_deref()) {
                return Some(text);
            }
        }
    }

    None
}

fn extract_stream_event_text(event: &Value, saw_delta: bool) -> Option<String> {
    let event_type = event.get("type").and_then(Value::as_str);
    match event_type {
        Some("response.output_text.delta") => {
            nonempty_preserve(event.get("delta").and_then(Value::as_str))
        }
        Some("response.output_text.done") if !saw_delta => {
            nonempty_preserve(event.get("text").and_then(Value::as_str))
        }
        _ => None,
    }
}

#[derive(Default)]
struct CodexStreamState {
    saw_delta: bool,
    delta_accumulator: String,
    fallback_text: Option<String>,
    accumulated_output: Vec<ResponsesOutput>,
}

async fn process_stream_event(
    event: Value,
    state: &mut CodexStreamState,
    tx: Option<&tokio::sync::mpsc::Sender<StreamResult<StreamEvent>>>,
    count_tokens: bool,
) -> Result<(), StreamError> {
    if let Some(message) = extract_stream_error_message(&event) {
        return Err(StreamError::Provider(format!(
            "OpenAI Codex stream error: {message}"
        )));
    }
    let event_type = event.get("type").and_then(Value::as_str);

    if event_type == Some("response.output_item.done") {
        if let Some(item) = event
            .get("item")
            .and_then(|value| serde_json::from_value::<ResponsesOutput>(value.clone()).ok())
        {
            state.accumulated_output.push(item);
        }
        return Ok(());
    }

    if matches!(event_type, Some("response.completed" | "response.done"))
        && let Some(mut response) = event
            .get("response")
            .and_then(|value| serde_json::from_value::<ResponsesResponse>(value.clone()).ok())
    {
        if response.output.is_empty() && !state.accumulated_output.is_empty() {
            response.output = state.accumulated_output.clone();
        }
        if let Some(text) = extract_responses_text(&response) {
            state.fallback_text.get_or_insert(text);
        }
    }

    if let Some(text) = extract_stream_event_text(&event, state.saw_delta) {
        if event_type == Some("response.output_text.delta") {
            state.saw_delta = true;
            state.delta_accumulator.push_str(&text);
            if let Some(tx) = tx {
                let mut chunk = StreamChunk::delta(text);
                if count_tokens {
                    chunk = chunk.with_token_estimate();
                }
                tx.send(Ok(StreamEvent::TextDelta(chunk)))
                    .await
                    .map_err(|_| StreamError::Provider("stream receiver dropped".to_string()))?;
            }
        } else if state.fallback_text.is_none() {
            state.fallback_text = Some(text);
        }
    }

    Ok(())
}

fn process_stream_event_sync(event: Value, state: &mut CodexStreamState) -> anyhow::Result<()> {
    if let Some(message) = extract_stream_error_message(&event) {
        anyhow::bail!("OpenAI Codex stream error: {message}");
    }
    let event_type = event.get("type").and_then(Value::as_str);

    if event_type == Some("response.output_item.done") {
        if let Some(item) = event
            .get("item")
            .and_then(|value| serde_json::from_value::<ResponsesOutput>(value.clone()).ok())
        {
            state.accumulated_output.push(item);
        }
        return Ok(());
    }

    if matches!(event_type, Some("response.completed" | "response.done"))
        && let Some(mut response) = event
            .get("response")
            .and_then(|value| serde_json::from_value::<ResponsesResponse>(value.clone()).ok())
    {
        if response.output.is_empty() && !state.accumulated_output.is_empty() {
            response.output = state.accumulated_output.clone();
        }
        if let Some(text) = extract_responses_text(&response) {
            state.fallback_text.get_or_insert(text);
        }
    }

    if let Some(text) = extract_stream_event_text(&event, state.saw_delta) {
        if event_type == Some("response.output_text.delta") {
            state.saw_delta = true;
            state.delta_accumulator.push_str(&text);
        } else if state.fallback_text.is_none() {
            state.fallback_text = Some(text);
        }
    }

    Ok(())
}

fn find_sse_separator(buffer: &str) -> Option<(usize, usize)> {
    let lf = buffer.find("\n\n").map(|idx| (idx, 2));
    let crlf = buffer.find("\r\n\r\n").map(|idx| (idx, 4));

    match (lf, crlf) {
        (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

async fn process_sse_buffer(
    buffer: &mut String,
    state: &mut CodexStreamState,
    tx: Option<&tokio::sync::mpsc::Sender<StreamResult<StreamEvent>>>,
    count_tokens: bool,
) -> Result<(), StreamError> {
    loop {
        let Some((idx, sep_len)) = find_sse_separator(buffer) else {
            break;
        };

        let chunk = buffer[..idx].to_string();
        *buffer = buffer[idx + sep_len..].to_string();
        process_sse_chunk(&chunk, state, tx, count_tokens).await?;
    }

    Ok(())
}

async fn process_sse_chunk(
    chunk: &str,
    state: &mut CodexStreamState,
    tx: Option<&tokio::sync::mpsc::Sender<StreamResult<StreamEvent>>>,
    count_tokens: bool,
) -> Result<(), StreamError> {
    let data_lines: Vec<String> = chunk
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(|line| line.trim().to_string())
        .collect();
    if data_lines.is_empty() {
        return Ok(());
    }

    let joined = data_lines.join("\n");
    let trimmed = joined.trim();
    if trimmed.is_empty() || trimmed == "[DONE]" {
        return Ok(());
    }

    if let Ok(event) = serde_json::from_str::<Value>(trimmed) {
        return process_stream_event(event, state, tx, count_tokens).await;
    }

    for line in data_lines {
        let line = line.trim();
        if line.is_empty() || line == "[DONE]" {
            continue;
        }
        if let Ok(event) = serde_json::from_str::<Value>(line) {
            process_stream_event(event, state, tx, count_tokens).await?;
        }
    }

    Ok(())
}

fn process_sse_chunk_sync(chunk: &str, state: &mut CodexStreamState) -> anyhow::Result<()> {
    let data_lines: Vec<String> = chunk
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(|line| line.trim().to_string())
        .collect();
    if data_lines.is_empty() {
        return Ok(());
    }

    let joined = data_lines.join("\n");
    let trimmed = joined.trim();
    if trimmed.is_empty() || trimmed == "[DONE]" {
        return Ok(());
    }

    if let Ok(event) = serde_json::from_str::<Value>(trimmed) {
        return process_stream_event_sync(event, state);
    }

    for line in data_lines {
        let line = line.trim();
        if line.is_empty() || line == "[DONE]" {
            continue;
        }
        if let Ok(event) = serde_json::from_str::<Value>(line) {
            process_stream_event_sync(event, state)?;
        }
    }

    Ok(())
}

fn process_sse_buffer_sync(buffer: &mut String, state: &mut CodexStreamState) -> anyhow::Result<()> {
    loop {
        let Some((idx, sep_len)) = find_sse_separator(buffer) else {
            break;
        };

        let chunk = buffer[..idx].to_string();
        *buffer = buffer[idx + sep_len..].to_string();
        process_sse_chunk_sync(&chunk, state)?;
    }

    Ok(())
}

fn parse_sse_text(body: &str) -> anyhow::Result<Option<String>> {
    let mut state = CodexStreamState::default();
    let mut buffer = body.to_string();

    process_sse_buffer_sync(&mut buffer, &mut state)?;

    if !buffer.trim().is_empty() {
        process_sse_chunk_sync(&buffer, &mut state)?;
    }

    if state.saw_delta {
        return Ok(nonempty_preserve(Some(&state.delta_accumulator)));
    }

    Ok(state.fallback_text)
}

fn extract_stream_error_message(event: &Value) -> Option<String> {
    let event_type = event.get("type").and_then(Value::as_str);

    if event_type == Some("error") {
        return first_nonempty(
            event
                .get("message")
                .and_then(Value::as_str)
                .or_else(|| event.get("code").and_then(Value::as_str))
                .or_else(|| {
                    event
                        .get("error")
                        .and_then(|error| error.get("message"))
                        .and_then(Value::as_str)
                }),
        );
    }

    if event_type == Some("response.failed") {
        return first_nonempty(
            event
                .get("response")
                .and_then(|response| response.get("error"))
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str),
        );
    }

    None
}

fn append_utf8_stream_chunk(
    body: &mut String,
    pending: &mut Vec<u8>,
    chunk: &[u8],
) -> anyhow::Result<()> {
    if pending.is_empty()
        && let Ok(text) = std::str::from_utf8(chunk)
    {
        body.push_str(text);
        return Ok(());
    }

    if !chunk.is_empty() {
        pending.extend_from_slice(chunk);
    }
    if pending.is_empty() {
        return Ok(());
    }

    match std::str::from_utf8(pending) {
        Ok(text) => {
            body.push_str(text);
            pending.clear();
            Ok(())
        }
        Err(err) => {
            let valid_up_to = err.valid_up_to();
            if valid_up_to > 0 {
                // SAFETY: `valid_up_to` always points to the end of a valid UTF-8 prefix.
                let prefix = std::str::from_utf8(&pending[..valid_up_to])
                    .expect("valid UTF-8 prefix from Utf8Error::valid_up_to");
                body.push_str(prefix);
                pending.drain(..valid_up_to);
            }

            if err.error_len().is_some() {
                return Err(anyhow::anyhow!(
                    "OpenAI Codex response contained invalid UTF-8: {err}"
                ));
            }

            // `error_len == None` means we have a valid prefix and an incomplete
            // multi-byte sequence at the end; keep it buffered until next chunk.
            Ok(())
        }
    }
}

#[allow(dead_code)]
fn decode_utf8_stream_chunks<'a, I>(chunks: I) -> anyhow::Result<String>
where
    I: IntoIterator<Item = &'a [u8]>,
{
    let mut body = String::new();
    let mut pending = Vec::new();

    for chunk in chunks {
        append_utf8_stream_chunk(&mut body, &mut pending, chunk)?;
    }

    if !pending.is_empty() {
        let err = std::str::from_utf8(&pending).expect_err("pending bytes should be invalid UTF-8");
        return Err(anyhow::anyhow!(
            "OpenAI Codex response ended with incomplete UTF-8: {err}"
        ));
    }

    Ok(body)
}

/// Read the response body incrementally via `bytes_stream()` to avoid
/// buffering the entire SSE payload in memory.  The previous implementation
/// used `response.text().await?` which holds the HTTP connection open until
/// every byte has arrived — on high-latency links the long-lived connection
/// often drops mid-read, producing the "error decoding response body" failure
/// reported in #3544.
async fn decode_responses_body(response: reqwest::Response) -> anyhow::Result<String> {
    let mut body = String::new();
    let mut pending_utf8 = Vec::new();
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let bytes = chunk
            .map_err(|err| anyhow::anyhow!("error reading OpenAI Codex response stream: {err}"))?;
        append_utf8_stream_chunk(&mut body, &mut pending_utf8, &bytes)?;
    }

    if !pending_utf8.is_empty() {
        let err = std::str::from_utf8(&pending_utf8)
            .expect_err("pending bytes should be invalid UTF-8 at end of stream");
        return Err(anyhow::anyhow!(
            "OpenAI Codex response ended with incomplete UTF-8: {err}"
        ));
    }

    if let Some(text) = parse_sse_text(&body)? {
        return Ok(text);
    }

    let body_trimmed = body.trim_start();
    let looks_like_sse = body_trimmed.starts_with("event:") || body_trimmed.starts_with("data:");
    if looks_like_sse {
        return Err(anyhow::anyhow!(
            "No response from OpenAI Codex stream payload: {}",
            super::sanitize_api_error(&body)
        ));
    }

    let parsed: ResponsesResponse = serde_json::from_str(&body).map_err(|err| {
        anyhow::anyhow!(
            "OpenAI Codex JSON parse failed: {err}. Payload: {}",
            super::sanitize_api_error(&body)
        )
    })?;
    extract_responses_text(&parsed).ok_or_else(|| anyhow::anyhow!("No response from OpenAI Codex"))
}

impl OpenAiCodexProvider {
    async fn send_responses_request(
        &self,
        input: Vec<ResponsesInput>,
        instructions: String,
        model: &str,
    ) -> anyhow::Result<String> {
        let use_gateway_api_key_auth = self.custom_endpoint && self.gateway_api_key.is_some();
        let profile = match self
            .auth
            .get_profile("openai-codex", self.auth_profile_override.as_deref())
            .await
        {
            Ok(profile) => profile,
            Err(err) if use_gateway_api_key_auth => {
                tracing::warn!(
                    error = %err,
                    "failed to load OpenAI Codex profile; continuing with custom endpoint API key mode"
                );
                None
            }
            Err(err) => return Err(err),
        };
        let oauth_access_token = match self
            .auth
            .get_valid_openai_access_token(self.auth_profile_override.as_deref())
            .await
        {
            Ok(token) => token,
            Err(err) if use_gateway_api_key_auth => {
                tracing::warn!(
                    error = %err,
                    "failed to refresh OpenAI token; continuing with custom endpoint API key mode"
                );
                None
            }
            Err(err) => return Err(err),
        };

        let account_id = profile.and_then(|profile| profile.account_id).or_else(|| {
            oauth_access_token
                .as_deref()
                .and_then(extract_account_id_from_jwt)
        });
        let access_token = if use_gateway_api_key_auth {
            oauth_access_token
        } else {
            Some(oauth_access_token.ok_or_else(|| {
                anyhow::anyhow!(
                    "OpenAI Codex auth profile not found. Run `zeroclaw auth login --provider openai-codex`."
                )
            })?)
        };
        let account_id = if use_gateway_api_key_auth {
            account_id
        } else {
            Some(account_id.ok_or_else(|| {
                anyhow::anyhow!(
                    "OpenAI Codex account id not found in auth profile/token. Run `zeroclaw auth login --provider openai-codex` again."
                )
            })?)
        };
        let normalized_model = normalize_model_id(model);

        let request = ResponsesRequest {
            model: normalized_model.to_string(),
            input,
            instructions,
            store: false,
            stream: true,
            text: ResponsesTextOptions {
                verbosity: "medium".to_string(),
            },
            reasoning: ResponsesReasoningOptions {
                effort: resolve_reasoning_effort(
                    normalized_model,
                    self.reasoning_effort.as_deref(),
                ),
                summary: "auto".to_string(),
            },
            include: vec!["reasoning.encrypted_content".to_string()],
            tool_choice: "auto".to_string(),
            parallel_tool_calls: true,
        };

        let bearer_token = if use_gateway_api_key_auth {
            self.gateway_api_key.as_deref().unwrap_or_default()
        } else {
            access_token.as_deref().unwrap_or_default()
        };

        let mut request_builder = self
            .client
            .post(&self.responses_url)
            .header("Authorization", format!("Bearer {bearer_token}"))
            .header("OpenAI-Beta", "responses=experimental")
            .header("originator", "pi")
            .header("accept", "text/event-stream")
            .header("Content-Type", "application/json");

        if let Some(account_id) = account_id.as_deref() {
            request_builder = request_builder.header("chatgpt-account-id", account_id);
        }

        if use_gateway_api_key_auth {
            if let Some(access_token) = access_token.as_deref() {
                request_builder = request_builder.header("x-openai-access-token", access_token);
            }
            if let Some(account_id) = account_id.as_deref() {
                request_builder = request_builder.header("x-openai-account-id", account_id);
            }
        }

        let response = request_builder.json(&request).send().await?;

        if !response.status().is_success() {
            return Err(super::api_error("OpenAI Codex", response).await);
        }

        decode_responses_body(response).await
    }

    async fn build_request_context(
        &self,
        messages: &[ChatMessage],
        _model: &str,
    ) -> anyhow::Result<(String, Vec<ResponsesInput>, Option<String>, Option<String>, bool)> {
        let use_gateway_api_key_auth = self.custom_endpoint && self.gateway_api_key.is_some();
        let profile = match self
            .auth
            .get_profile("openai-codex", self.auth_profile_override.as_deref())
            .await
        {
            Ok(profile) => profile,
            Err(err) if use_gateway_api_key_auth => {
                tracing::warn!(error = %err, "failed to load OpenAI Codex profile; continuing with custom endpoint API key mode");
                None
            }
            Err(err) => return Err(err),
        };
        let oauth_access_token = match self
            .auth
            .get_valid_openai_access_token(self.auth_profile_override.as_deref())
            .await
        {
            Ok(token) => token,
            Err(err) if use_gateway_api_key_auth => {
                tracing::warn!(error = %err, "failed to refresh OpenAI token; continuing with custom endpoint API key mode");
                None
            }
            Err(err) => return Err(err),
        };

        let account_id = profile.and_then(|profile| profile.account_id).or_else(|| {
            oauth_access_token
                .as_deref()
                .and_then(extract_account_id_from_jwt)
        });
        let access_token = if use_gateway_api_key_auth {
            oauth_access_token
        } else {
            Some(oauth_access_token.ok_or_else(|| {
                anyhow::anyhow!(
                    "OpenAI Codex auth profile not found. Run `zeroclaw auth login --provider openai-codex`."
                )
            })?)
        };
        let account_id = if use_gateway_api_key_auth {
            account_id
        } else {
            Some(account_id.ok_or_else(|| {
                anyhow::anyhow!(
                    "OpenAI Codex account id not found in auth profile/token. Run `zeroclaw auth login --provider openai-codex` again."
                )
            })?)
        };

        let prepared = multimodal::prepare_messages_for_provider(
            messages,
            &zeroclaw_config::schema::MultimodalConfig::default(),
        )
        .await?;
        let (instructions, input) = build_responses_input(&prepared.messages);

        Ok((
            instructions,
            input,
            access_token,
            account_id,
            use_gateway_api_key_auth,
        ))
    }
}

#[async_trait]
impl Provider for OpenAiCodexProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            native_tool_calling: false,
            vision: true,
            prompt_caching: false,
        }
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn stream_chat(
        &self,
        request: ProviderChatRequest<'_>,
        model: &str,
        _temperature: f64,
        options: StreamOptions,
    ) -> stream::BoxStream<'static, StreamResult<StreamEvent>> {
        if !options.enabled {
            return stream::once(async { Ok(StreamEvent::Final) }).boxed();
        }

        let normalized_model = normalize_model_id(model).to_string();
        let messages = request.messages.to_vec();
        let auth = self.auth.clone();
        let auth_profile_override = self.auth_profile_override.clone();
        let custom_endpoint = self.custom_endpoint;
        let gateway_api_key = self.gateway_api_key.clone();
        let reasoning_effort = self.reasoning_effort.clone();
        let client = self.client.clone();
        let responses_url = self.responses_url.clone();
        let count_tokens = options.count_tokens;
        let (tx, rx) = tokio::sync::mpsc::channel::<StreamResult<StreamEvent>>(100);

        tokio::spawn(async move {
            let helper = OpenAiCodexProvider {
                auth,
                auth_profile_override,
                responses_url: responses_url.clone(),
                custom_endpoint,
                gateway_api_key: gateway_api_key.clone(),
                reasoning_effort: reasoning_effort.clone(),
                client: client.clone(),
            };

            let (instructions, input, access_token, account_id, use_gateway_api_key_auth) =
                match helper.build_request_context(&messages, &normalized_model).await {
                    Ok(ctx) => ctx,
                    Err(error) => {
                        let _ = tx.send(Err(StreamError::Provider(error.to_string()))).await;
                        return;
                    }
                };

            let request = ResponsesRequest {
                model: normalized_model.clone(),
                input,
                instructions,
                store: false,
                stream: true,
                text: ResponsesTextOptions {
                    verbosity: "medium".to_string(),
                },
                reasoning: ResponsesReasoningOptions {
                    effort: resolve_reasoning_effort(&normalized_model, reasoning_effort.as_deref()),
                    summary: "auto".to_string(),
                },
                include: vec!["reasoning.encrypted_content".to_string()],
                tool_choice: "auto".to_string(),
                parallel_tool_calls: true,
            };

            let bearer_token = if use_gateway_api_key_auth {
                gateway_api_key.clone().unwrap_or_default()
            } else {
                access_token.clone().unwrap_or_default()
            };

            let mut request_builder = client
                .post(&responses_url)
                .header("Authorization", format!("Bearer {bearer_token}"))
                .header("OpenAI-Beta", "responses=experimental")
                .header("originator", "pi")
                .header("accept", "text/event-stream")
                .header("Content-Type", "application/json");

            if let Some(account_id) = account_id.as_deref() {
                request_builder = request_builder.header("chatgpt-account-id", account_id);
            }
            if use_gateway_api_key_auth {
                if let Some(access_token) = access_token.as_deref() {
                    request_builder =
                        request_builder.header("x-openai-access-token", access_token);
                }
                if let Some(account_id) = account_id.as_deref() {
                    request_builder = request_builder.header("x-openai-account-id", account_id);
                }
            }

            let response = match request_builder.json(&request).send().await {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.send(Err(StreamError::Http(e.to_string()))).await;
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let error = response
                    .text()
                    .await
                    .unwrap_or_else(|_| format!("HTTP error: {status}"));
                let _ = tx
                    .send(Err(StreamError::Provider(format!("{}: {}", status, error))))
                    .await;
                return;
            }

            let mut state = CodexStreamState::default();
            let mut buffer = String::new();
            let mut pending_utf8 = Vec::new();
            let mut stream = response.bytes_stream();
            let mut emitted_delta = false;

            while let Some(chunk) = stream.next().await {
                let bytes = match chunk {
                    Ok(bytes) => bytes,
                    Err(err) => {
                        if emitted_delta {
                            let _ = tx.send(Ok(StreamEvent::Final)).await;
                        } else {
                            let _ = tx
                                .send(Err(StreamError::Provider(format!(
                                    "error reading OpenAI Codex response stream: {err}"
                                ))))
                                .await;
                        }
                        return;
                    }
                };

                if let Err(err) = append_utf8_stream_chunk(&mut buffer, &mut pending_utf8, &bytes)
                {
                    if emitted_delta {
                        let _ = tx.send(Ok(StreamEvent::Final)).await;
                    } else {
                        let _ = tx.send(Err(StreamError::Provider(err.to_string()))).await;
                    }
                    return;
                }

                if let Err(err) = process_sse_buffer(&mut buffer, &mut state, Some(&tx), count_tokens).await {
                    if emitted_delta || state.saw_delta {
                        let _ = tx.send(Ok(StreamEvent::Final)).await;
                    } else {
                        let _ = tx.send(Err(err)).await;
                    }
                    return;
                }
                emitted_delta |= state.saw_delta;
            }

            if !pending_utf8.is_empty() {
                let err = std::str::from_utf8(&pending_utf8)
                    .expect_err("pending bytes should be invalid UTF-8 at end of stream");
                if emitted_delta {
                    let _ = tx.send(Ok(StreamEvent::Final)).await;
                } else {
                    let _ = tx
                        .send(Err(StreamError::Provider(format!(
                            "OpenAI Codex response ended with incomplete UTF-8: {err}"
                        ))))
                        .await;
                }
                return;
            }

            if !buffer.trim().is_empty() {
                if let Err(err) = process_sse_chunk(&buffer, &mut state, Some(&tx), count_tokens).await {
                    if emitted_delta || state.saw_delta {
                        let _ = tx.send(Ok(StreamEvent::Final)).await;
                    } else {
                        let _ = tx.send(Err(err)).await;
                    }
                    return;
                }
            }

            if !state.saw_delta {
                if let Some(text) = state.fallback_text.take() {
                    let mut chunk = StreamChunk::delta(text);
                    if count_tokens {
                        chunk = chunk.with_token_estimate();
                    }
                    let _ = tx.send(Ok(StreamEvent::TextDelta(chunk))).await;
                }
            }

            let _ = tx.send(Ok(StreamEvent::Final)).await;
        });

        stream::unfold(rx, |mut rx| async move { rx.recv().await.map(|event| (event, rx)) })
            .boxed()
    }

    async fn chat_with_system(
        &self,
        system_prompt: Option<&str>,
        message: &str,
        model: &str,
        _temperature: f64,
    ) -> anyhow::Result<String> {
        // Build temporary messages array
        let mut messages = Vec::new();
        if let Some(sys) = system_prompt {
            messages.push(ChatMessage::system(sys));
        }
        messages.push(ChatMessage::user(message));

        // Normalize images: convert file paths to data URIs
        let config = zeroclaw_config::schema::MultimodalConfig::default();
        let prepared = crate::multimodal::prepare_messages_for_provider(&messages, &config).await?;

        let (instructions, input) = build_responses_input(&prepared.messages);
        self.send_responses_request(input, instructions, model)
            .await
    }

    async fn chat_with_history(
        &self,
        messages: &[ChatMessage],
        model: &str,
        _temperature: f64,
    ) -> anyhow::Result<String> {
        // Normalize image markers: convert file paths to data URIs
        let config = zeroclaw_config::schema::MultimodalConfig::default();
        let prepared = crate::multimodal::prepare_messages_for_provider(messages, &config).await?;

        let (instructions, input) = build_responses_input(&prepared.messages);
        self.send_responses_request(input, instructions, model)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{EnvGuard, env_lock};

    #[test]
    fn extracts_output_text_first() {
        let response = ResponsesResponse {
            output: vec![],
            output_text: Some("hello".into()),
        };
        assert_eq!(extract_responses_text(&response).as_deref(), Some("hello"));
    }

    #[test]
    fn extracts_nested_output_text() {
        let response = ResponsesResponse {
            output: vec![ResponsesOutput {
                content: vec![ResponsesContent {
                    kind: Some("output_text".into()),
                    text: Some("nested".into()),
                }],
            }],
            output_text: None,
        };
        assert_eq!(extract_responses_text(&response).as_deref(), Some("nested"));
    }

    #[test]
    fn default_state_dir_is_non_empty() {
        let path = default_zeroclaw_dir();
        assert!(!path.as_os_str().is_empty());
    }

    #[test]
    fn build_responses_url_appends_suffix_for_base_url() {
        assert_eq!(
            build_responses_url("https://api.tonsof.blue/v1").unwrap(),
            "https://api.tonsof.blue/v1/responses"
        );
    }

    #[test]
    fn build_responses_url_keeps_existing_responses_endpoint() {
        assert_eq!(
            build_responses_url("https://api.tonsof.blue/v1/responses").unwrap(),
            "https://api.tonsof.blue/v1/responses"
        );
    }

    #[test]
    fn resolve_responses_url_prefers_explicit_endpoint_env() {
        let _lock = env_lock();
        let _endpoint_guard = EnvGuard::set(
            CODEX_RESPONSES_URL_ENV,
            Some("https://env.example.com/v1/responses"),
        );
        let _base_guard = EnvGuard::set(CODEX_BASE_URL_ENV, Some("https://base.example.com/v1"));

        let options = ProviderRuntimeOptions::default();
        assert_eq!(
            resolve_responses_url(&options).unwrap(),
            "https://env.example.com/v1/responses"
        );
    }

    #[test]
    fn resolve_responses_url_uses_provider_api_url_override() {
        let _lock = env_lock();
        let _endpoint_guard = EnvGuard::set(CODEX_RESPONSES_URL_ENV, None);
        let _base_guard = EnvGuard::set(CODEX_BASE_URL_ENV, None);

        let options = ProviderRuntimeOptions {
            provider_api_url: Some("https://proxy.example.com/v1".to_string()),
            ..ProviderRuntimeOptions::default()
        };

        assert_eq!(
            resolve_responses_url(&options).unwrap(),
            "https://proxy.example.com/v1/responses"
        );
    }

    #[test]
    fn default_responses_url_detector_handles_equivalent_urls() {
        assert!(is_default_responses_url(DEFAULT_CODEX_RESPONSES_URL));
        assert!(is_default_responses_url(
            "https://chatgpt.com/backend-api/codex/responses/"
        ));
        assert!(!is_default_responses_url(
            "https://api.tonsof.blue/v1/responses"
        ));
    }

    #[test]
    fn constructor_enables_custom_endpoint_key_mode() {
        let options = ProviderRuntimeOptions {
            provider_api_url: Some("https://api.tonsof.blue/v1".to_string()),
            ..ProviderRuntimeOptions::default()
        };

        let provider = OpenAiCodexProvider::new(&options, Some("test-key")).unwrap();
        assert!(provider.custom_endpoint);
        assert_eq!(provider.gateway_api_key.as_deref(), Some("test-key"));
    }

    #[test]
    fn resolve_instructions_uses_default_when_missing() {
        assert_eq!(
            resolve_instructions(None),
            DEFAULT_CODEX_INSTRUCTIONS.to_string()
        );
    }

    #[test]
    fn resolve_instructions_uses_default_when_blank() {
        assert_eq!(
            resolve_instructions(Some("   ")),
            DEFAULT_CODEX_INSTRUCTIONS.to_string()
        );
    }

    #[test]
    fn resolve_instructions_uses_system_prompt_when_present() {
        assert_eq!(
            resolve_instructions(Some("Be strict")),
            "Be strict".to_string()
        );
    }

    #[test]
    fn clamp_reasoning_effort_adjusts_known_models() {
        assert_eq!(
            clamp_reasoning_effort("gpt-5-codex", "xhigh"),
            "high".to_string()
        );
        assert_eq!(
            clamp_reasoning_effort("gpt-5-codex", "minimal"),
            "low".to_string()
        );
        assert_eq!(
            clamp_reasoning_effort("gpt-5-codex", "medium"),
            "medium".to_string()
        );
        assert_eq!(
            clamp_reasoning_effort("gpt-5.3-codex", "minimal"),
            "low".to_string()
        );
        assert_eq!(
            clamp_reasoning_effort("gpt-5.1", "xhigh"),
            "high".to_string()
        );
        assert_eq!(
            clamp_reasoning_effort("gpt-5-codex", "xhigh"),
            "high".to_string()
        );
        assert_eq!(
            clamp_reasoning_effort("gpt-5.1-codex-mini", "low"),
            "medium".to_string()
        );
        assert_eq!(
            clamp_reasoning_effort("gpt-5.1-codex-mini", "xhigh"),
            "high".to_string()
        );
        assert_eq!(
            clamp_reasoning_effort("gpt-5.3-codex", "xhigh"),
            "xhigh".to_string()
        );
    }

    #[test]
    fn resolve_reasoning_effort_prefers_configured_override() {
        let _lock = env_lock();
        let _guard = EnvGuard::set("ZEROCLAW_CODEX_REASONING_EFFORT", Some("low"));
        assert_eq!(
            resolve_reasoning_effort("gpt-5-codex", Some("high")),
            "high".to_string()
        );
    }

    #[test]
    fn resolve_reasoning_effort_uses_legacy_env_when_unconfigured() {
        let _lock = env_lock();
        let _guard = EnvGuard::set("ZEROCLAW_CODEX_REASONING_EFFORT", Some("minimal"));
        assert_eq!(
            resolve_reasoning_effort("gpt-5-codex", None),
            "low".to_string()
        );
    }

    #[test]
    fn parse_sse_text_reads_output_text_delta() {
        let payload = r#"data: {"type":"response.created","response":{"id":"resp_123"}}

data: {"type":"response.output_text.delta","delta":"Hello"}
data: {"type":"response.output_text.delta","delta":" world"}
data: {"type":"response.completed","response":{"output_text":"Hello world"}}
data: [DONE]
"#;

        assert_eq!(
            parse_sse_text(payload).unwrap().as_deref(),
            Some("Hello world")
        );
    }

    #[test]
    fn parse_sse_text_falls_back_to_completed_response() {
        let payload = r#"data: {"type":"response.completed","response":{"output_text":"Done"}}
data: [DONE]
"#;

        assert_eq!(parse_sse_text(payload).unwrap().as_deref(), Some("Done"));
    }

    #[test]
    fn parse_sse_text_backfills_completed_response_output_from_output_items() {
        let payload = r#"data: {"type":"response.output_item.done","item":{"content":[{"type":"output_text","text":"Recovered output"}]}}

data: {"type":"response.completed","response":{"output":[],"output_text":null}}
data: [DONE]
"#;

        assert_eq!(
            parse_sse_text(payload).unwrap().as_deref(),
            Some("Recovered output")
        );
    }

    #[test]
    fn decode_utf8_stream_chunks_handles_multibyte_split_across_chunks() {
        let payload = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello 世\"}\n\ndata: [DONE]\n";
        let bytes = payload.as_bytes();
        let split_at = payload.find('世').unwrap() + 1;

        let decoded = decode_utf8_stream_chunks([&bytes[..split_at], &bytes[split_at..]]).unwrap();
        assert_eq!(decoded, payload);
        assert_eq!(
            parse_sse_text(&decoded).unwrap().as_deref(),
            Some("Hello 世")
        );
    }

    #[test]
    fn build_responses_input_maps_content_types_by_role() {
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: "You are helpful.".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "Hi".into(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "Hello!".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "Thanks".into(),
            },
        ];
        let (instructions, input) = build_responses_input(&messages);
        assert_eq!(instructions, "You are helpful.");
        assert_eq!(input.len(), 3);

        let json: Vec<Value> = input
            .iter()
            .map(|item| serde_json::to_value(item).unwrap())
            .collect();
        assert_eq!(json[0]["role"], "user");
        assert_eq!(json[0]["content"][0]["type"], "input_text");
        assert_eq!(json[1]["role"], "assistant");
        assert_eq!(json[1]["content"][0]["type"], "output_text");
        assert_eq!(json[2]["role"], "user");
        assert_eq!(json[2]["content"][0]["type"], "input_text");
    }

    #[test]
    fn build_responses_input_uses_default_instructions_without_system() {
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "Hello".into(),
        }];
        let (instructions, input) = build_responses_input(&messages);
        assert_eq!(instructions, DEFAULT_CODEX_INSTRUCTIONS);
        assert_eq!(input.len(), 1);
    }

    #[test]
    fn build_responses_input_ignores_unknown_roles() {
        let messages = vec![
            ChatMessage {
                role: "tool".into(),
                content: "result".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "Go".into(),
            },
        ];
        let (instructions, input) = build_responses_input(&messages);
        assert_eq!(instructions, DEFAULT_CODEX_INSTRUCTIONS);
        assert_eq!(input.len(), 1);
        let json = serde_json::to_value(&input[0]).unwrap();
        assert_eq!(json["role"], "user");
    }

    #[test]
    fn build_responses_input_handles_image_markers() {
        let messages = vec![ChatMessage::user(
            "Describe this\n\n[IMAGE:data:image/png;base64,abc]",
        )];
        let (_, input) = build_responses_input(&messages);

        assert_eq!(input.len(), 1);
        assert_eq!(input[0].role, "user");
        assert_eq!(input[0].content.len(), 2);

        let json: Vec<Value> = input[0]
            .content
            .iter()
            .map(|item| serde_json::to_value(item).unwrap())
            .collect();

        // First content = text
        assert_eq!(json[0]["type"], "input_text");
        assert!(json[0]["text"].as_str().unwrap().contains("Describe this"));

        // Second content = image
        assert_eq!(json[1]["type"], "input_image");
        assert_eq!(json[1]["image_url"], "data:image/png;base64,abc");
    }

    #[test]
    fn build_responses_input_preserves_text_only_messages() {
        let messages = vec![ChatMessage::user("Hello without images")];
        let (_, input) = build_responses_input(&messages);

        assert_eq!(input.len(), 1);
        assert_eq!(input[0].content.len(), 1);

        let json = serde_json::to_value(&input[0].content[0]).unwrap();
        assert_eq!(json["type"], "input_text");
        assert_eq!(json["text"], "Hello without images");
    }

    #[test]
    fn build_responses_input_handles_multiple_images() {
        let messages = vec![ChatMessage::user(
            "Compare these: [IMAGE:data:image/png;base64,img1] and [IMAGE:data:image/jpeg;base64,img2]",
        )];
        let (_, input) = build_responses_input(&messages);

        assert_eq!(input.len(), 1);
        assert_eq!(input[0].content.len(), 3); // text + 2 images

        let json: Vec<Value> = input[0]
            .content
            .iter()
            .map(|item| serde_json::to_value(item).unwrap())
            .collect();

        assert_eq!(json[0]["type"], "input_text");
        assert_eq!(json[1]["type"], "input_image");
        assert_eq!(json[2]["type"], "input_image");
    }

    #[test]
    fn capabilities_includes_vision() {
        let options = ProviderRuntimeOptions {
            provider_api_url: None,
            zeroclaw_dir: None,
            secrets_encrypt: false,
            auth_profile_override: None,
            reasoning_enabled: None,
            reasoning_effort: None,
            provider_timeout_secs: None,
            extra_headers: std::collections::HashMap::new(),
            api_path: None,
            provider_max_tokens: None,
            merge_system_into_user: false,
        };
        let provider =
            OpenAiCodexProvider::new(&options, None).expect("provider should initialize");
        let caps = provider.capabilities();

        assert!(!caps.native_tool_calling);
        assert!(caps.vision);
    }

    #[tokio::test]
    async fn supports_streaming_returns_true() {
        let options = ProviderRuntimeOptions::default();
        let provider = OpenAiCodexProvider::new(&options, None).unwrap();
        assert!(provider.supports_streaming());
    }
}
