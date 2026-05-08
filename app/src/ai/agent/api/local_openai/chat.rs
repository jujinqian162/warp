use anyhow::anyhow;
use async_stream::try_stream;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::server::server_api::AIApiError;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct ChatCompletionsRequest {
    pub model: String,
    pub messages: Vec<OpenAIChatMessage>,
    pub stream: bool,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<OpenAITool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct OpenAIChatMessage {
    pub(crate) role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_calls: Option<Vec<OpenAIChatToolCall>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct OpenAIChatToolCall {
    pub(crate) id: String,
    pub(crate) r#type: &'static str,
    pub(crate) function: OpenAIChatToolCallFunction,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct OpenAIChatToolCallFunction {
    pub(crate) name: String,
    pub(crate) arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct OpenAITool {
    pub(crate) r#type: &'static str,
    pub(crate) function: OpenAIFunctionTool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct OpenAIFunctionTool {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) parameters: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum OpenAIStreamEvent {
    Content(String),
    ToolCallDelta(OpenAIToolCallDelta),
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OpenAIToolCallDelta {
    pub index: usize,
    pub id: Option<String>,
    pub function_name: Option<String>,
    pub arguments_delta: String,
}

#[derive(Deserialize)]
struct ChatCompletionChunk {
    choices: Vec<ChatCompletionChoice>,
}

#[derive(Deserialize)]
struct ChatCompletionChoice {
    delta: ChatCompletionDelta,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ChatCompletionDelta {
    content: Option<String>,
    tool_calls: Option<Vec<ChatCompletionToolCallDelta>>,
}

#[derive(Deserialize)]
struct ChatCompletionToolCallDelta {
    index: usize,
    id: Option<String>,
    function: Option<ChatCompletionToolCallFunctionDelta>,
}

#[derive(Deserialize)]
struct ChatCompletionToolCallFunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

pub(super) fn parse_sse_event(line: &str) -> Result<Option<OpenAIStreamEvent>, AIApiError> {
    let line = line.trim();
    if line.is_empty() || !line.starts_with("data:") {
        return Ok(None);
    }

    let payload = line.trim_start_matches("data:").trim();
    if payload == "[DONE]" {
        return Ok(Some(OpenAIStreamEvent::Done));
    }

    let chunk: ChatCompletionChunk = serde_json::from_str(payload)?;
    let Some(choice) = chunk.choices.into_iter().next() else {
        return Ok(None);
    };

    if let Some(content) = choice.delta.content.filter(|content| !content.is_empty()) {
        return Ok(Some(OpenAIStreamEvent::Content(content)));
    }

    if let Some(tool_call) = choice
        .delta
        .tool_calls
        .and_then(|calls| calls.into_iter().next())
    {
        let function = tool_call.function;
        return Ok(Some(OpenAIStreamEvent::ToolCallDelta(
            OpenAIToolCallDelta {
                index: tool_call.index,
                id: tool_call.id,
                function_name: function.as_ref().and_then(|f| f.name.clone()),
                arguments_delta: function.and_then(|f| f.arguments).unwrap_or_default(),
            },
        )));
    }

    if choice.finish_reason.as_deref() == Some("tool_calls") {
        return Ok(None);
    }

    Ok(None)
}

pub(super) fn parse_sse_delta(line: &str) -> Result<Option<String>, AIApiError> {
    Ok(match parse_sse_event(line)? {
        Some(OpenAIStreamEvent::Content(content)) => Some(content),
        _ => None,
    })
}

fn push_utf8_text(
    pending_bytes: &mut Vec<u8>,
    text_buffer: &mut String,
    bytes: &[u8],
) -> Result<(), AIApiError> {
    pending_bytes.extend_from_slice(bytes);

    loop {
        match std::str::from_utf8(pending_bytes) {
            Ok(text) => {
                text_buffer.push_str(text);
                pending_bytes.clear();
                return Ok(());
            }
            Err(error) => {
                let valid_up_to = error.valid_up_to();
                if valid_up_to > 0 {
                    let valid_text = std::str::from_utf8(&pending_bytes[..valid_up_to])
                        .expect("valid_up_to must split at a valid UTF-8 boundary")
                        .to_owned();
                    text_buffer.push_str(&valid_text);
                    pending_bytes.drain(..valid_up_to);
                }

                if error.error_len().is_some() {
                    return Err(AIApiError::Other(anyhow!(
                        "Local OpenAI backend received invalid UTF-8 in SSE response"
                    )));
                }

                return Ok(());
            }
        }
    }
}

fn flush_utf8_text(
    pending_bytes: &mut Vec<u8>,
    text_buffer: &mut String,
) -> Result<(), AIApiError> {
    if pending_bytes.is_empty() {
        return Ok(());
    }

    match std::str::from_utf8(pending_bytes) {
        Ok(text) => {
            text_buffer.push_str(text);
            pending_bytes.clear();
            Ok(())
        }
        Err(_) => Err(AIApiError::Other(anyhow!(
            "Local OpenAI backend received incomplete UTF-8 in SSE response"
        ))),
    }
}

fn take_sse_line(text_buffer: &mut String) -> Option<String> {
    let newline_index = text_buffer.find('\n')?;
    let mut line: String = text_buffer.drain(..=newline_index).collect();
    if line.ends_with('\n') {
        line.pop();
    }
    if line.ends_with('\r') {
        line.pop();
    }
    Some(line)
}

pub(super) fn chat_completion_event_stream(
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    request: ChatCompletionsRequest,
) -> impl futures_lite::Stream<Item = Result<OpenAIStreamEvent, AIApiError>> + Send + 'static {
    try_stream! {
        let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
        let response = client
            .post(url)
            .bearer_auth(api_key)
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response
                .text()
                .await
                .unwrap_or_else(|error| format!("(no response body: {error:#})"));
            Err(AIApiError::ErrorStatus(status, body))?;
        } else {
            let mut chunks = response.bytes_stream();
            let mut pending_bytes = Vec::new();
            let mut text_buffer = String::new();

            while let Some(chunk) = chunks.next().await {
                let chunk = chunk?;
                push_utf8_text(&mut pending_bytes, &mut text_buffer, &chunk)?;

                while let Some(line) = take_sse_line(&mut text_buffer) {
                    if let Some(event) = parse_sse_event(&line)? {
                        if matches!(event, OpenAIStreamEvent::Done) {
                            return;
                        }
                        yield event;
                    }
                }
            }

            flush_utf8_text(&mut pending_bytes, &mut text_buffer)?;
            if !text_buffer.is_empty() {
                if let Some(event) = parse_sse_event(&text_buffer)? {
                    if !matches!(event, OpenAIStreamEvent::Done) {
                        yield event;
                    }
                }
            }
        }
    }
}
