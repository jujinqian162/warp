use anyhow::anyhow;
use warp_multi_agent_api as api;

use crate::{
    ai::agent::{
        api::{convert_conversation::convert_tool_call_result_to_input, RequestParams},
        task::TaskId,
        AIAgentInput, MarkdownActionResult,
    },
    server::server_api::AIApiError,
};

use super::chat::{OpenAIChatMessage, OpenAIChatToolCall, OpenAIChatToolCallFunction};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedOpenAIHistory {
    pub(crate) messages: Vec<OpenAIChatMessage>,
    pub(crate) messages_to_persist: Vec<api::Message>,
}

impl PreparedOpenAIHistory {
    pub(super) fn task_description_fallback(&self) -> String {
        self.messages
            .iter()
            .find_map(|message| {
                (message.role == "user")
                    .then(|| message.content.clone())
                    .flatten()
            })
            .unwrap_or_else(|| "Local OpenAI conversation".to_string())
    }
}

fn message_from_task_message(
    message: &api::Message,
) -> Result<Option<OpenAIChatMessage>, AIApiError> {
    let Some(content) = message.message.as_ref() else {
        return Ok(None);
    };

    match content {
        api::message::Message::UserQuery(query) => Ok(Some(OpenAIChatMessage {
            role: "user",
            content: Some(query.query.clone()),
            tool_call_id: None,
            tool_calls: None,
        })),
        api::message::Message::AgentOutput(output) => Ok(Some(OpenAIChatMessage {
            role: "assistant",
            content: Some(output.text.clone()),
            tool_call_id: None,
            tool_calls: None,
        })),
        api::message::Message::ToolCall(tool_call) => Ok(Some(OpenAIChatMessage {
            role: "assistant",
            content: None,
            tool_call_id: None,
            tool_calls: Some(vec![openai_tool_call_from_warp_tool_call(tool_call)?]),
        })),
        api::message::Message::ToolCallResult(result) => {
            let mut document_versions = std::collections::HashMap::new();
            let task_id = TaskId::new(message.task_id.clone());
            let content = convert_tool_call_result_to_input(
                &task_id,
                result,
                &std::collections::HashMap::new(),
                &mut document_versions,
            )
            .and_then(|input| input.action_result().cloned())
            .map(|action_result| format!("{}", MarkdownActionResult(&action_result.result)))
            .unwrap_or_else(|| format!("Tool result recorded for {}", result.tool_call_id));

            Ok(Some(OpenAIChatMessage {
                role: "tool",
                content: Some(content),
                tool_call_id: Some(result.tool_call_id.clone()),
                tool_calls: None,
            }))
        }
        _ => Ok(None),
    }
}

fn openai_tool_call_from_warp_tool_call(
    tool_call: &api::message::ToolCall,
) -> Result<OpenAIChatToolCall, AIApiError> {
    let function_name = super::tools::openai_name_for_warp_tool_call(tool_call)?;
    let arguments = super::tools::openai_arguments_for_warp_tool_call(tool_call)?;
    Ok(OpenAIChatToolCall {
        id: tool_call.tool_call_id.clone(),
        r#type: "function",
        function: OpenAIChatToolCallFunction {
            name: function_name,
            arguments,
        },
    })
}

pub(super) fn build_openai_history(
    params: &RequestParams,
) -> Result<PreparedOpenAIHistory, AIApiError> {
    let mut messages = Vec::new();
    for task in &params.tasks {
        for message in &task.messages {
            if let Some(chat_message) = message_from_task_message(message)? {
                messages.push(chat_message);
            }
        }
    }

    let mut messages_to_persist = Vec::new();
    for input in &params.input {
        match input {
            AIAgentInput::UserQuery { query, .. } if !query.trim().is_empty() => {
                messages.push(OpenAIChatMessage {
                    role: "user",
                    content: Some(query.trim().to_string()),
                    tool_call_id: None,
                    tool_calls: None,
                });
            }
            AIAgentInput::ActionResult { result, .. } => {
                let content = format!("{}", MarkdownActionResult(&result.result));
                messages.push(OpenAIChatMessage {
                    role: "tool",
                    content: Some(content),
                    tool_call_id: Some(result.id.to_string()),
                    tool_calls: None,
                });
                messages_to_persist.push(tool_call_result_message_from_action_result(result)?);
            }
            other => {
                return Err(AIApiError::Other(anyhow!(
                    "Local OpenAI backend does not support input type {other:?}"
                )));
            }
        }
    }

    if messages.is_empty() {
        return Err(AIApiError::Other(anyhow!(
            "Local OpenAI backend received no chat messages"
        )));
    }

    Ok(PreparedOpenAIHistory {
        messages,
        messages_to_persist,
    })
}

fn tool_call_result_message_from_action_result(
    result: &crate::ai::agent::AIAgentActionResult,
) -> Result<api::Message, AIApiError> {
    use api::message::tool_call_result::Result as MessageResult;
    use api::request::input::tool_call_result::Result as RequestResult;
    use api::request::input::user_inputs::user_input::Input as RequestUserInput;

    let request_input: RequestUserInput = result.clone().try_into().map_err(|error| {
        AIApiError::Other(anyhow!(
            "Local OpenAI backend could not convert tool result for history: {error:?}"
        ))
    })?;
    let RequestUserInput::ToolCallResult(request_result) = request_input else {
        return Err(AIApiError::Other(anyhow!(
            "Local OpenAI backend converted an action result into a non-tool input"
        )));
    };
    let Some(request_result) = request_result.result else {
        return Err(AIApiError::Other(anyhow!(
            "Local OpenAI backend converted an empty tool result"
        )));
    };

    let message_result = match request_result {
        RequestResult::RunShellCommand(value) => MessageResult::RunShellCommand(value),
        RequestResult::ReadFiles(value) => MessageResult::ReadFiles(value),
        RequestResult::SearchCodebase(value) => MessageResult::SearchCodebase(value),
        RequestResult::ApplyFileDiffs(value) => MessageResult::ApplyFileDiffs(value),
        RequestResult::Grep(value) => MessageResult::Grep(value),
        RequestResult::FileGlobV2(value) => MessageResult::FileGlobV2(value),
        RequestResult::ReadMcpResource(value) => MessageResult::ReadMcpResource(value),
        RequestResult::CallMcpTool(value) => MessageResult::CallMcpTool(value),
        other => {
            return Err(AIApiError::Other(anyhow!(
                "Local OpenAI backend cannot persist unsupported tool result {other:?}"
            )));
        }
    };

    Ok(api::Message {
        id: uuid::Uuid::new_v4().to_string(),
        task_id: result.task_id.to_string(),
        message: Some(api::message::Message::ToolCallResult(
            api::message::ToolCallResult {
                tool_call_id: result.id.to_string(),
                result: Some(message_result),
                ..Default::default()
            },
        )),
        ..Default::default()
    })
}
