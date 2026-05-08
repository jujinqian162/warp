use anyhow::anyhow;
use serde_json::{json, Value};
use warp_multi_agent_api as api;

use crate::{ai::agent::api::RequestParams, server::server_api::AIApiError};

use super::chat::{OpenAIFunctionTool, OpenAITool};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CompletedOpenAIToolCall {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) arguments: Value,
}

fn function_tool(name: &str, description: &str, parameters: Value) -> OpenAITool {
    OpenAITool {
        r#type: "function",
        function: OpenAIFunctionTool {
            name: name.to_string(),
            description: description.to_string(),
            parameters,
        },
    }
}

pub(super) fn built_in_openai_tools(params: &RequestParams) -> Vec<OpenAITool> {
    let supported = params
        .supported_tools_override
        .clone()
        .unwrap_or_else(|| super::super::r#impl::get_supported_tools(params));

    let has = |tool_type: api::ToolType| supported.contains(&tool_type);
    let mut tools = Vec::new();

    if has(api::ToolType::RunShellCommand) {
        tools.push(function_tool(
            "run_shell_command",
            "Run a shell command in the current terminal session. Use this when command output is needed.",
            json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" },
                    "is_read_only": { "type": "boolean" },
                    "is_risky": { "type": "boolean" },
                    "uses_pager": { "type": "boolean" },
                    "wait_until_completion": { "type": "boolean" }
                },
                "required": ["command"]
            }),
        ));
    }

    if has(api::ToolType::ReadFiles) {
        tools.push(function_tool(
            "read_files",
            "Read one or more files. Omit line ranges to read the entire file.",
            json!({
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": { "type": "string" },
                                "line_ranges": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "start": { "type": "integer" },
                                            "end": { "type": "integer" }
                                        },
                                        "required": ["start", "end"]
                                    }
                                }
                            },
                            "required": ["path"]
                        }
                    }
                },
                "required": ["files"]
            }),
        ));
    }

    if has(api::ToolType::SearchCodebase) {
        tools.push(function_tool(
            "search_codebase",
            "Search the current codebase for relevant files and snippets.",
            json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "path_filters": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "codebase_path": { "type": "string" }
                },
                "required": ["query"]
            }),
        ));
    }

    if has(api::ToolType::Grep) {
        tools.push(function_tool(
            "grep",
            "Search for exact text or regex patterns inside files.",
            json!({
                "type": "object",
                "properties": {
                    "queries": { "type": "array", "items": { "type": "string" } },
                    "path": { "type": "string" }
                },
                "required": ["queries", "path"]
            }),
        ));
    }

    if has(api::ToolType::FileGlobV2) {
        tools.push(function_tool(
            "file_glob",
            "Find files by glob-style filename patterns.",
            json!({
                "type": "object",
                "properties": {
                    "patterns": { "type": "array", "items": { "type": "string" } },
                    "search_dir": { "type": "string" }
                },
                "required": ["patterns"]
            }),
        ));
    }

    if has(api::ToolType::ApplyFileDiffs) {
        tools.push(function_tool(
            "apply_file_diffs",
            "Propose edits to files. The client will show the diff for review before applying it.",
            json!({
                "type": "object",
                "properties": {
                    "summary": { "type": "string" },
                    "diffs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file_path": { "type": "string" },
                                "search": { "type": "string" },
                                "replace": { "type": "string" }
                            },
                            "required": ["file_path", "search", "replace"]
                        }
                    }
                },
                "required": ["summary", "diffs"]
            }),
        ));
    }

    tools
}

pub(super) fn tool_call_message_from_openai_call(
    task_id: &str,
    request_id: &str,
    call: CompletedOpenAIToolCall,
) -> Result<api::Message, AIApiError> {
    let tool = built_in_warp_tool_from_openai_call(&call)?;
    Ok(api::Message {
        id: uuid::Uuid::new_v4().to_string(),
        task_id: task_id.to_string(),
        request_id: request_id.to_string(),
        message: Some(api::message::Message::ToolCall(api::message::ToolCall {
            tool_call_id: call.id,
            tool: Some(tool),
        })),
        ..Default::default()
    })
}

fn built_in_warp_tool_from_openai_call(
    call: &CompletedOpenAIToolCall,
) -> Result<api::message::tool_call::Tool, AIApiError> {
    match call.name.as_str() {
        "run_shell_command" => {
            let command = required_string(&call.arguments, "run_shell_command", "command")?;
            Ok(api::message::tool_call::Tool::RunShellCommand(
                api::message::tool_call::RunShellCommand {
                    command,
                    is_read_only: optional_bool(&call.arguments, "run_shell_command", "is_read_only")
                        ?.unwrap_or(false),
                    is_risky: optional_bool(&call.arguments, "run_shell_command", "is_risky")
                        ?.unwrap_or(false),
                    uses_pager: optional_bool(&call.arguments, "run_shell_command", "uses_pager")
                        ?.unwrap_or(false),
                    wait_until_complete_value: optional_bool(
                        &call.arguments,
                        "run_shell_command",
                        "wait_until_completion",
                    )?
                    .map(api::message::tool_call::run_shell_command::WaitUntilCompleteValue::WaitUntilComplete),
                    ..Default::default()
                },
            ))
        }
        "read_files" => Ok(api::message::tool_call::Tool::ReadFiles(read_files_call(&call.arguments)?)),
        "search_codebase" => Ok(api::message::tool_call::Tool::SearchCodebase(search_codebase_call(&call.arguments)?)),
        "grep" => Ok(api::message::tool_call::Tool::Grep(grep_call(&call.arguments)?)),
        "file_glob" => Ok(api::message::tool_call::Tool::FileGlobV2(file_glob_call(&call.arguments)?)),
        "apply_file_diffs" => Ok(api::message::tool_call::Tool::ApplyFileDiffs(apply_file_diffs_call(&call.arguments)?)),
        other => Err(AIApiError::Other(anyhow!(
            "Local OpenAI backend received unsupported tool call {other}"
        ))),
    }
}

fn required_string(arguments: &Value, tool: &str, field: &str) -> Result<String, AIApiError> {
    arguments
        .get(field)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| invalid_tool_field(tool, field, "non-empty string"))
}

fn optional_string(arguments: &Value, field: &str) -> Option<String> {
    arguments
        .get(field)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .filter(|value| !value.is_empty())
}

fn optional_bool(arguments: &Value, tool: &str, field: &str) -> Result<Option<bool>, AIApiError> {
    match arguments.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_bool()
            .map(Some)
            .ok_or_else(|| invalid_tool_field(tool, field, "boolean")),
    }
}

fn string_array(arguments: &Value, tool: &str, field: &str) -> Result<Vec<String>, AIApiError> {
    let Some(values) = arguments.get(field).and_then(Value::as_array) else {
        return Err(invalid_tool_field(tool, field, "array of strings"));
    };

    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(ToOwned::to_owned)
                .ok_or_else(|| invalid_tool_field(tool, field, "array of strings"))
        })
        .collect()
}

fn optional_string_array(
    arguments: &Value,
    tool: &str,
    field: &str,
) -> Result<Vec<String>, AIApiError> {
    if arguments.get(field).is_none() {
        return Ok(Vec::new());
    }
    string_array(arguments, tool, field)
}

fn invalid_tool_field(tool: &str, field: &str, expected: &str) -> AIApiError {
    AIApiError::Other(anyhow!(
        "Local OpenAI backend received invalid {tool}.{field}; expected {expected}"
    ))
}

fn read_files_call(arguments: &Value) -> Result<api::message::tool_call::ReadFiles, AIApiError> {
    let Some(files) = arguments.get("files").and_then(Value::as_array) else {
        return Err(invalid_tool_field("read_files", "files", "array"));
    };

    let files = files
        .iter()
        .map(|file| {
            let path = file
                .get("path")
                .or_else(|| file.get("name"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| invalid_tool_field("read_files", "files.path", "non-empty string"))?;
            let line_ranges = match file.get("line_ranges") {
                None => Vec::new(),
                Some(Value::Array(ranges)) => ranges
                    .iter()
                    .map(|range| {
                        let start = range
                            .get("start")
                            .and_then(Value::as_u64)
                            .ok_or_else(|| invalid_tool_field("read_files", "line_ranges.start", "integer"))?;
                        let end = range
                            .get("end")
                            .and_then(Value::as_u64)
                            .ok_or_else(|| invalid_tool_field("read_files", "line_ranges.end", "integer"))?;
                        Ok(api::FileContentLineRange {
                            start: start as u32,
                            end: end as u32,
                        })
                    })
                    .collect::<Result<Vec<_>, AIApiError>>()?,
                Some(_) => {
                    return Err(invalid_tool_field(
                        "read_files",
                        "line_ranges",
                        "array",
                    ));
                }
            };
            Ok(api::message::tool_call::read_files::File {
                name: path,
                line_ranges,
            })
        })
        .collect::<Result<Vec<_>, AIApiError>>()?;

    Ok(api::message::tool_call::ReadFiles { files })
}

fn search_codebase_call(
    arguments: &Value,
) -> Result<api::message::tool_call::SearchCodebase, AIApiError> {
    Ok(api::message::tool_call::SearchCodebase {
        query: required_string(arguments, "search_codebase", "query")?,
        path_filters: optional_string_array(arguments, "search_codebase", "path_filters")?,
        codebase_path: optional_string(arguments, "codebase_path").unwrap_or_default(),
    })
}

fn grep_call(arguments: &Value) -> Result<api::message::tool_call::Grep, AIApiError> {
    Ok(api::message::tool_call::Grep {
        queries: string_array(arguments, "grep", "queries")?,
        path: required_string(arguments, "grep", "path")?,
    })
}

fn file_glob_call(arguments: &Value) -> Result<api::message::tool_call::FileGlobV2, AIApiError> {
    Ok(api::message::tool_call::FileGlobV2 {
        patterns: string_array(arguments, "file_glob", "patterns")?,
        search_dir: optional_string(arguments, "search_dir").unwrap_or_default(),
        ..Default::default()
    })
}

fn apply_file_diffs_call(
    arguments: &Value,
) -> Result<api::message::tool_call::ApplyFileDiffs, AIApiError> {
    let Some(diffs) = arguments.get("diffs").and_then(Value::as_array) else {
        return Err(invalid_tool_field("apply_file_diffs", "diffs", "array"));
    };

    let diffs = diffs
        .iter()
        .map(|diff| {
            Ok(api::message::tool_call::apply_file_diffs::FileDiff {
                file_path: required_string(diff, "apply_file_diffs", "diffs.file_path")?,
                search: required_string(diff, "apply_file_diffs", "diffs.search")?,
                replace: required_string(diff, "apply_file_diffs", "diffs.replace")?,
            })
        })
        .collect::<Result<Vec<_>, AIApiError>>()?;

    Ok(api::message::tool_call::ApplyFileDiffs {
        summary: required_string(arguments, "apply_file_diffs", "summary")?,
        diffs,
        ..Default::default()
    })
}

pub(super) fn openai_name_for_warp_tool_call(
    tool_call: &api::message::ToolCall,
) -> Result<String, AIApiError> {
    let Some(tool) = tool_call.tool.as_ref() else {
        return Err(AIApiError::Other(anyhow!("Tool call is missing tool payload")));
    };

    match tool {
        api::message::tool_call::Tool::RunShellCommand(_) => Ok("run_shell_command".to_string()),
        api::message::tool_call::Tool::ReadFiles(_) => Ok("read_files".to_string()),
        api::message::tool_call::Tool::SearchCodebase(_) => Ok("search_codebase".to_string()),
        api::message::tool_call::Tool::Grep(_) => Ok("grep".to_string()),
        api::message::tool_call::Tool::FileGlobV2(_) => Ok("file_glob".to_string()),
        api::message::tool_call::Tool::ApplyFileDiffs(_) => Ok("apply_file_diffs".to_string()),
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

    let arguments = match tool {
        api::message::tool_call::Tool::RunShellCommand(run) => json!({
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
        }),
        api::message::tool_call::Tool::ReadFiles(read) => json!({
            "files": read.files.iter().map(|file| json!({
                "path": file.name.clone(),
                "line_ranges": file.line_ranges.iter().map(|range| json!({
                    "start": range.start,
                    "end": range.end,
                })).collect::<Vec<_>>()
            })).collect::<Vec<_>>()
        }),
        api::message::tool_call::Tool::SearchCodebase(search) => json!({
            "query": search.query.clone(),
            "path_filters": search.path_filters.clone(),
            "codebase_path": search.codebase_path.clone()
        }),
        api::message::tool_call::Tool::Grep(grep) => json!({
            "queries": grep.queries.clone(),
            "path": grep.path.clone()
        }),
        api::message::tool_call::Tool::FileGlobV2(glob) => json!({
            "patterns": glob.patterns.clone(),
            "search_dir": glob.search_dir.clone()
        }),
        api::message::tool_call::Tool::ApplyFileDiffs(diffs) => json!({
            "summary": diffs.summary.clone(),
            "diffs": diffs.diffs.iter().map(|diff| json!({
                "file_path": diff.file_path.clone(),
                "search": diff.search.clone(),
                "replace": diff.replace.clone(),
            })).collect::<Vec<_>>()
        }),
        _ => Err(AIApiError::Other(anyhow!(
            "Local OpenAI backend cannot serialize unsupported Warp tool call"
        )))?,
    };
    Ok(arguments.to_string())
}
