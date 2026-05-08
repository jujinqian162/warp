use std::sync::Arc;

use anyhow::anyhow;
use async_stream::stream;
use futures::channel::oneshot;
use futures_util::StreamExt;
use uuid::Uuid;
use warp_multi_agent_api as api;

mod chat;
mod events;
mod history;
mod tools;

use crate::{
    ai::agent::{
        api::{Event, LocalOpenAIBackendSettings, RequestParams, ResponseStream},
        AIAgentInput,
    },
    server::server_api::AIApiError,
};

const DEFAULT_LOCAL_MODEL: &str = "gpt-4o-mini";
const DEFAULT_LOCAL_MAX_TOKENS: u32 = 4096;

pub async fn generate_output(
    settings: LocalOpenAIBackendSettings,
    params: RequestParams,
    cancellation_rx: oneshot::Receiver<()>,
) -> Result<ResponseStream, ai::agent::convert::ConvertToAPITypeError> {
    log::info!("Local OpenAI backend selected");
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

fn local_model(configured_model: Option<String>, selected_model: &str) -> String {
    configured_model
        .map(|model| model.trim().to_owned())
        .filter(|model| !model.is_empty())
        .or_else(|| {
            let selected_model = selected_model.trim();
            (!selected_model.is_empty()).then(|| selected_model.to_owned())
        })
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
        create_event: Some(events::create_task_event(api::Task {
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

fn parse_sse_delta(line: &str) -> Result<Option<String>, AIApiError> {
    chat::parse_sse_delta(line)
}

fn local_text_stream(
    settings: LocalOpenAIBackendSettings,
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
        let model = local_model(settings.model, params.model.as_str());
        let prepared_history = match history::build_openai_history(&params) {
            Ok(value) => value,
            Err(error) => {
                yield Err(Arc::new(error));
                return;
            }
        };
        let active_task = active_task(&params, &prepared_history.task_description_fallback());

        let request_id = Uuid::new_v4().to_string();
        let conversation_id = String::new();
        let run_id = format!("local-openai-text-{}", Uuid::new_v4());
        let message_id = Uuid::new_v4().to_string();

        yield Ok(events::init_event(conversation_id, request_id.clone(), run_id));

        if let Some(create_event) = active_task.create_event {
            yield Ok(create_event);
        }

        if !prepared_history.messages_to_persist.is_empty() {
            yield Ok(events::add_messages_event(
                &active_task.id,
                prepared_history.messages_to_persist,
            ));
        }

        let (mcp_tools, mcp_registry) = tools::mcp_openai_tools(&params);
        let mut tools = tools::built_in_openai_tools(&params);
        tools.extend(mcp_tools);
        log::info!(
            "Local OpenAI backend selected: model={model}, tools={}, messages={}",
            tools.len(),
            prepared_history.messages.len()
        );
        let request = chat::ChatCompletionsRequest {
            model,
            messages: prepared_history.messages,
            stream: true,
            max_tokens: DEFAULT_LOCAL_MAX_TOKENS,
            tool_choice: (!tools.is_empty()).then_some("auto"),
            tools,
        };

        let mut events = Box::pin(chat::chat_completion_event_stream(
            reqwest::Client::new(),
            base_url,
            api_key,
            request,
        ));

        let mut has_message = false;
        let mut tool_calls = chat::ToolCallAccumulator::default();
        while let Some(event) = events.next().await {
            match event {
                Ok(chat::OpenAIStreamEvent::Content(delta)) => {
                    let message = events::agent_output_message(&message_id, &active_task.id, &request_id, delta);
                    if has_message {
                        yield Ok(events::append_message_event(&active_task.id, message));
                    } else {
                        has_message = true;
                        yield Ok(events::add_message_event(&active_task.id, message));
                    }
                }
                Ok(chat::OpenAIStreamEvent::ToolCallDelta(delta)) => {
                    tool_calls.push(delta);
                }
                Ok(chat::OpenAIStreamEvent::Done) => break,
                Err(error) => {
                    yield Err(Arc::new(error));
                    return;
                }
            }
        }

        let completed_tool_calls = match tool_calls.finish() {
            Ok(calls) => calls,
            Err(error) => {
                yield Err(Arc::new(error));
                return;
            }
        };
        let has_tool_calls = !completed_tool_calls.is_empty();

        if has_tool_calls {
            log::info!(
                "Local OpenAI backend emitted {} tool call(s)",
                completed_tool_calls.len()
            );
            let mut messages = Vec::new();
            for call in completed_tool_calls {
                match tools::tool_call_message_from_openai_call(
                    &active_task.id,
                    &request_id,
                    call,
                    &mcp_registry,
                ) {
                    Ok(message) => messages.push(message),
                    Err(error) => {
                        yield Err(Arc::new(error));
                        return;
                    }
                }
            }
            yield Ok(events::add_messages_event(&active_task.id, messages));
        }

        if !has_message && !has_tool_calls {
            yield Ok(events::add_message_event(
                &active_task.id,
                events::agent_output_message(
                    &message_id,
                    &active_task.id,
                    &request_id,
                    String::new(),
                ),
            ));
        }

        yield Ok(events::finished_event());
    }
}

#[cfg(test)]
pub(crate) mod tests_support {
    use super::*;

    pub(crate) const AGENT_OUTPUT_FIELD_MASK: &str = super::events::AGENT_OUTPUT_FIELD_MASK;
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

    pub(crate) use super::chat::OpenAIStreamEvent;

    pub(crate) fn parse_sse_event(line: &str) -> Result<Option<OpenAIStreamEvent>, AIApiError> {
        super::chat::parse_sse_event(line)
    }

    pub(crate) fn build_openai_history(
        params: &RequestParams,
    ) -> Result<super::history::PreparedOpenAIHistory, AIApiError> {
        super::history::build_openai_history(params)
    }

    pub(crate) use super::tools::CompletedOpenAIToolCall;

    pub(crate) fn built_in_openai_tools(params: &RequestParams) -> Vec<super::chat::OpenAITool> {
        super::tools::built_in_openai_tools(params)
    }

    pub(crate) fn tool_call_message_from_openai_call(
        task_id: &str,
        request_id: &str,
        call: CompletedOpenAIToolCall,
    ) -> Result<api::Message, AIApiError> {
        super::tools::tool_call_message_from_openai_call(
            task_id,
            request_id,
            call,
            &Default::default(),
        )
    }

    pub(crate) fn mcp_tool_function_name(server_id: &str, tool_name: &str) -> String {
        super::tools::mcp_tool_function_name(server_id, tool_name)
    }
}
