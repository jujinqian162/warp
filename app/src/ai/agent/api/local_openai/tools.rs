use anyhow::anyhow;
use serde_json::json;
use warp_multi_agent_api as api;

use crate::server::server_api::AIApiError;

pub(super) fn openai_name_for_warp_tool_call(
    tool_call: &api::message::ToolCall,
) -> Result<String, AIApiError> {
    let Some(tool) = tool_call.tool.as_ref() else {
        return Err(AIApiError::Other(anyhow!("Tool call is missing tool payload")));
    };

    match tool {
        api::message::tool_call::Tool::RunShellCommand(_) => {
            Ok("run_shell_command".to_string())
        }
        _ => Err(AIApiError::Other(anyhow!(
            "Local OpenAI backend cannot replay unsupported Warp tool call"
        ))),
    }
}

pub(super) fn openai_arguments_for_warp_tool_call(
    tool_call: &api::message::ToolCall,
) -> Result<String, AIApiError> {
    let Some(tool) = tool_call.tool.as_ref() else {
        return Err(AIApiError::Other(anyhow!("Tool call is missing tool payload")));
    };

    match tool {
        api::message::tool_call::Tool::RunShellCommand(run) => Ok(json!({
            "command": run.command.clone(),
            "is_read_only": run.is_read_only,
            "is_risky": run.is_risky,
            "uses_pager": run.uses_pager,
            "wait_until_completion": run.wait_until_complete_value.as_ref().map(|value| {
                matches!(
                    value,
                    api::message::tool_call::run_shell_command::WaitUntilCompleteValue::WaitUntilComplete(true)
                )
            })
        })
        .to_string()),
        _ => Err(AIApiError::Other(anyhow!(
            "Local OpenAI backend cannot serialize unsupported Warp tool call"
        ))),
    }
}
