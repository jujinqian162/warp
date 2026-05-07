use std::sync::Arc;

use anyhow::anyhow;
use async_stream::stream;
use futures::channel::oneshot;
use futures_util::StreamExt;
use serde::Deserialize;
use uuid::Uuid;
use warp_multi_agent_api as api;

use crate::{
    ai::agent::{
        api::{Event, LocalOpenAITextBackendSettings, RequestParams, ResponseStream},
        AIAgentInput,
    },
    server::server_api::AIApiError,
};

const CHAT_COMPLETIONS_PATH: &str = "chat/completions";
const DEFAULT_LOCAL_MODEL: &str = "gpt-4o-mini";
const AGENT_OUTPUT_FIELD_MASK: &str = "message.agent_output.text";

pub async fn generate_text_output(
    settings: LocalOpenAITextBackendSettings,
    params: RequestParams,
    cancellation_rx: oneshot::Receiver<()>,
) -> Result<ResponseStream, ai::agent::convert::ConvertToAPITypeError> {
    log::info!("Local OpenAI text backend selected");
    let output_stream = local_text_stream(settings, params).take_until(cancellation_rx);
    Ok(Box::pin(output_stream))
}

fn require_setting(value: Option<String>, name: &'static str) -> Result<String, AIApiError> {
    value
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            AIApiError::Other(anyhow!(
                "{name} is required for the local OpenAI text backend"
            ))
        })
}

fn local_model(model: Option<String>) -> String {
    model
        .map(|model| model.trim().to_owned())
        .filter(|model| !model.is_empty())
        .unwrap_or_else(|| DEFAULT_LOCAL_MODEL.to_string())
}

fn extract_user_query(inputs: &[AIAgentInput]) -> Result<String, AIApiError> {
    let Some(input) = inputs.last() else {
        return Err(AIApiError::Other(anyhow!(
            "Local OpenAI text backend received an empty request"
        )));
    };

    match input {
        AIAgentInput::UserQuery { query, .. } if !query.trim().is_empty() => {
            Ok(query.trim().to_owned())
        }
        AIAgentInput::UserQuery { .. } => Err(AIApiError::Other(anyhow!(
            "Local OpenAI text backend received an empty user query"
        ))),
        other => Err(AIApiError::Other(anyhow!(
            "Local OpenAI text backend only supports plain user queries in this phase; received {other:?}"
        ))),
    }
}

struct ActiveTask {
    id: String,
    create_event: Option<api::ResponseEvent>,
}

fn active_task(params: &RequestParams, prompt: &str) -> ActiveTask {
    if let Some(id) = params
        .tasks
        .first()
        .map(|task| task.id.trim().to_owned())
        .filter(|id| !id.is_empty())
    {
        return ActiveTask {
            id,
            create_event: None,
        };
    }

    let id = Uuid::new_v4().to_string();
    ActiveTask {
        id: id.clone(),
        create_event: Some(create_task_event(api::Task {
            id,
            description: prompt.to_owned(),
            ..Default::default()
        })),
    }
}

fn active_task_id(params: &RequestParams) -> Result<String, AIApiError> {
    params
        .tasks
        .first()
        .map(|task| task.id.trim().to_owned())
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            AIApiError::Other(anyhow!("Local OpenAI text backend requires an active task"))
        })
}

#[derive(serde::Serialize)]
struct ChatCompletionsRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
}

#[derive(serde::Serialize)]
struct ChatMessage {
    role: &'static str,
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionChunk {
    choices: Vec<ChatCompletionChoice>,
}

#[derive(Deserialize)]
struct ChatCompletionChoice {
    delta: ChatCompletionDelta,
}

#[derive(Deserialize)]
struct ChatCompletionDelta {
    content: Option<String>,
}

fn init_event(conversation_id: String, request_id: String, run_id: String) -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::Init(
            api::response_event::StreamInit {
                conversation_id,
                request_id,
                run_id,
            },
        )),
    }
}

fn finished_event() -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::Finished(
            api::response_event::StreamFinished {
                reason: Some(api::response_event::stream_finished::Reason::Done(
                    api::response_event::stream_finished::Done {},
                )),
                ..Default::default()
            },
        )),
    }
}

fn agent_output_message(
    message_id: &str,
    task_id: &str,
    request_id: &str,
    text: String,
) -> api::Message {
    api::Message {
        id: message_id.to_owned(),
        task_id: task_id.to_owned(),
        request_id: request_id.to_owned(),
        message: Some(api::message::Message::AgentOutput(
            api::message::AgentOutput { text },
        )),
        ..Default::default()
    }
}

fn add_message_event(task_id: &str, message: api::Message) -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::ClientActions(
            api::response_event::ClientActions {
                actions: vec![api::ClientAction {
                    action: Some(api::client_action::Action::AddMessagesToTask(
                        api::client_action::AddMessagesToTask {
                            task_id: task_id.to_owned(),
                            messages: vec![message],
                        },
                    )),
                }],
            },
        )),
    }
}

fn create_task_event(task: api::Task) -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::ClientActions(
            api::response_event::ClientActions {
                actions: vec![api::ClientAction {
                    action: Some(api::client_action::Action::CreateTask(
                        api::client_action::CreateTask { task: Some(task) },
                    )),
                }],
            },
        )),
    }
}

fn append_message_event(task_id: &str, message: api::Message) -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::ClientActions(
            api::response_event::ClientActions {
                actions: vec![api::ClientAction {
                    action: Some(api::client_action::Action::AppendToMessageContent(
                        api::client_action::AppendToMessageContent {
                            task_id: task_id.to_owned(),
                            message: Some(message),
                            mask: Some(prost_types::FieldMask {
                                paths: vec![AGENT_OUTPUT_FIELD_MASK.to_owned()],
                            }),
                        },
                    )),
                }],
            },
        )),
    }
}

fn parse_sse_delta(line: &str) -> Result<Option<String>, AIApiError> {
    let line = line.trim();
    if line.is_empty() || !line.starts_with("data:") {
        return Ok(None);
    }

    let payload = line.trim_start_matches("data:").trim();
    if payload == "[DONE]" {
        return Ok(None);
    }

    let chunk: ChatCompletionChunk = serde_json::from_str(payload)?;
    Ok(chunk
        .choices
        .into_iter()
        .filter_map(|choice| choice.delta.content)
        .find(|content| !content.is_empty()))
}

async fn chat_completion_deltas(
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
    prompt: String,
) -> Result<Vec<String>, AIApiError> {
    let url = format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        CHAT_COMPLETIONS_PATH
    );
    let response = client
        .post(url)
        .bearer_auth(api_key)
        .json(&ChatCompletionsRequest {
            model,
            stream: true,
            messages: vec![ChatMessage {
                role: "user",
                content: prompt,
            }],
        })
        .send()
        .await?;

    let status = response.status();
    if !status.is_success() {
        let body = response
            .text()
            .await
            .unwrap_or_else(|error| format!("(no response body: {error:#})"));
        return Err(AIApiError::ErrorStatus(status, body));
    }

    let body = response.text().await?;
    let mut deltas = Vec::new();
    for line in body.lines() {
        if let Some(delta) = parse_sse_delta(line)? {
            deltas.push(delta);
        }
    }
    Ok(deltas)
}

fn local_text_stream(
    settings: LocalOpenAITextBackendSettings,
    params: RequestParams,
) -> impl futures_lite::Stream<Item = Event> + Send + 'static {
    stream! {
        let api_key = match require_setting(settings.api_key, "OpenAI API key") {
            Ok(value) => value,
            Err(error) => {
                yield Err(Arc::new(error));
                return;
            }
        };
        let base_url = match require_setting(settings.base_url, "OpenAI Base URL") {
            Ok(value) => value,
            Err(error) => {
                yield Err(Arc::new(error));
                return;
            }
        };
        let model = local_model(settings.model);
        let prompt = match extract_user_query(&params.input) {
            Ok(value) => value,
            Err(error) => {
                yield Err(Arc::new(error));
                return;
            }
        };
        let active_task = active_task(&params, &prompt);

        let request_id = Uuid::new_v4().to_string();
        let conversation_id = params
            .conversation_token
            .as_ref()
            .map(|token| token.as_str().to_owned())
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let run_id = format!("local-openai-text-{conversation_id}");
        let message_id = Uuid::new_v4().to_string();

        yield Ok(init_event(conversation_id, request_id.clone(), run_id));

        let deltas = match chat_completion_deltas(
            reqwest::Client::new(),
            base_url,
            api_key,
            model,
            prompt,
        )
        .await
        {
            Ok(deltas) => deltas,
            Err(error) => {
                yield Err(Arc::new(error));
                return;
            }
        };

        if let Some(create_event) = active_task.create_event {
            yield Ok(create_event);
        }

        let mut has_message = false;
        for delta in deltas {
            let message = agent_output_message(&message_id, &active_task.id, &request_id, delta);
            if has_message {
                yield Ok(append_message_event(&active_task.id, message));
            } else {
                has_message = true;
                yield Ok(add_message_event(&active_task.id, message));
            }
        }

        if !has_message {
            yield Ok(add_message_event(
                &active_task.id,
                agent_output_message(
                    &message_id,
                    &active_task.id,
                    &request_id,
                    String::new(),
                ),
            ));
        }

        yield Ok(finished_event());
    }
}

#[cfg(test)]
pub(crate) mod tests_support {
    use super::*;

    pub(crate) const AGENT_OUTPUT_FIELD_MASK: &str = super::AGENT_OUTPUT_FIELD_MASK;
    pub(crate) const DEFAULT_LOCAL_MODEL: &str = super::DEFAULT_LOCAL_MODEL;

    pub(crate) fn active_task_id(params: &RequestParams) -> Result<String, AIApiError> {
        super::active_task_id(params)
    }

    pub(crate) fn extract_user_query(inputs: &[AIAgentInput]) -> Result<String, AIApiError> {
        super::extract_user_query(inputs)
    }

    pub(crate) fn parse_sse_delta(line: &str) -> Result<Option<String>, AIApiError> {
        super::parse_sse_delta(line)
    }
}
