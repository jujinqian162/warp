use crate::ai::agent::api::{LocalOpenAIBackendSettings, MultiAgentBackend, RequestParams};
use crate::ai::blocklist::SessionContext;
use crate::ai::llms::LLMId;
use warp_core::features::FeatureFlag;
use warp_multi_agent_api as api;

use super::get_supported_tools;

fn request_params_with_ask_user_question_enabled(ask_user_question_enabled: bool) -> RequestParams {
    let model = LLMId::from("test-model");

    RequestParams {
        input: vec![],
        conversation_token: None,
        forked_from_conversation_token: None,
        ambient_agent_task_id: None,
        tasks: vec![],
        existing_suggestions: None,
        metadata: None,
        session_context: SessionContext::new_for_test(),
        model: model.clone(),
        coding_model: model.clone(),
        cli_agent_model: model.clone(),
        computer_use_model: model,
        is_memory_enabled: false,
        warp_drive_context_enabled: false,
        context_window_limit: None,
        mcp_context: None,
        planning_enabled: true,
        should_redact_secrets: false,
        api_keys: None,
        allow_use_of_warp_credits_with_byok: false,
        autonomy_level: api::AutonomyLevel::Supervised,
        isolation_level: api::IsolationLevel::None,
        web_search_enabled: false,
        computer_use_enabled: false,
        ask_user_question_enabled,
        research_agent_enabled: false,
        orchestration_enabled: false,
        supported_tools_override: None,
        parent_agent_id: None,
        agent_name: None,
        backend: MultiAgentBackend::WarpServer,
    }
}

#[test]
fn supported_tools_omits_ask_user_question_when_disabled() {
    let params = request_params_with_ask_user_question_enabled(false);
    let supported_tools = get_supported_tools(&params);

    assert!(!supported_tools.contains(&api::ToolType::AskUserQuestion));
}

#[test]
fn supported_tools_includes_ask_user_question_when_enabled_and_feature_flag_is_enabled() {
    if !FeatureFlag::AskUserQuestion.is_enabled() {
        return;
    }

    let params = request_params_with_ask_user_question_enabled(true);
    let supported_tools = get_supported_tools(&params);

    assert!(supported_tools.contains(&api::ToolType::AskUserQuestion));
}

#[test]
fn supported_tools_include_upload_artifact_when_feature_flag_is_enabled() {
    let _flag = FeatureFlag::ArtifactCommand.override_enabled(true);
    let params = request_params_with_ask_user_question_enabled(false);
    let supported_tools = get_supported_tools(&params);

    assert!(supported_tools.contains(&api::ToolType::UploadFileArtifact));
}

#[test]
fn supported_tools_omit_upload_artifact_when_feature_flag_is_disabled() {
    let _flag = FeatureFlag::ArtifactCommand.override_enabled(false);
    let params = request_params_with_ask_user_question_enabled(false);
    let supported_tools = get_supported_tools(&params);

    assert!(!supported_tools.contains(&api::ToolType::UploadFileArtifact));
}

fn user_query_input(query: &str) -> crate::ai::agent::AIAgentInput {
    crate::ai::agent::AIAgentInput::UserQuery {
        query: query.to_owned(),
        context: std::sync::Arc::from([]),
        static_query_type: None,
        referenced_attachments: Default::default(),
        user_query_mode: Default::default(),
        running_command: None,
        intended_agent: None,
    }
}

#[test]
fn local_openai_text_parses_chat_completion_delta() {
    let line = r#"data: {"choices":[{"delta":{"content":"hello"}}]}"#;

    let delta = super::super::local_openai::tests_support::parse_sse_delta(line)
        .expect("valid SSE line");

    assert_eq!(delta.as_deref(), Some("hello"));
}

#[test]
fn local_openai_text_ignores_done_delta() {
    let delta =
        super::super::local_openai::tests_support::parse_sse_delta("data: [DONE]").unwrap();

    assert_eq!(delta, None);
}

#[test]
fn local_openai_parses_content_and_tool_call_chunks() {
    let content_line = r#"data: {"choices":[{"delta":{"content":"hello"}}]}"#;
    let tool_line = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"run_shell_command","arguments":"{\"command\":\"pwd\"}"}}]},"finish_reason":"tool_calls"}]}"#;

    let content = super::super::local_openai::tests_support::parse_sse_event(content_line)
        .expect("content line parses");
    assert_eq!(
        content,
        Some(
            super::super::local_openai::tests_support::OpenAIStreamEvent::Content(
                "hello".to_string()
            )
        )
    );

    let tool = super::super::local_openai::tests_support::parse_sse_event(tool_line)
        .expect("tool line parses");
    assert!(matches!(
        tool,
        Some(
            super::super::local_openai::tests_support::OpenAIStreamEvent::ToolCallDelta(_)
        )
    ));
}

#[test]
fn local_openai_text_extracts_plain_user_query() {
    let input = vec![user_query_input(" hello ")];

    let query = super::super::local_openai::tests_support::extract_user_query(&input).unwrap();

    assert_eq!(query, "hello");
}

#[test]
fn local_openai_text_rejects_non_user_query() {
    let input = vec![crate::ai::agent::AIAgentInput::ResumeConversation {
        context: std::sync::Arc::from([]),
    }];

    let error =
        super::super::local_openai::tests_support::extract_user_query(&input).unwrap_err();

    assert!(format!("{error:?}").contains("only supports plain user queries"));
}

#[test]
fn local_openai_text_uses_default_model_constant() {
    assert_eq!(
        super::super::local_openai::tests_support::DEFAULT_LOCAL_MODEL,
        "gpt-4o-mini"
    );
}

#[test]
fn local_openai_text_extracts_active_task_id() {
    let mut params = request_params_with_ask_user_question_enabled(false);
    params.tasks = vec![api::Task {
        id: " task-1 ".to_string(),
        ..Default::default()
    }];

    let task_id = super::super::local_openai::tests_support::active_task_id(&params).unwrap();

    assert_eq!(task_id, "task-1");
}

#[test]
fn local_openai_text_posts_to_configured_base_url_and_emits_text_events() {
    let server_api = crate::server::server_api::ServerApiProvider::new_for_test().get();

    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(async move {
            local_openai_text_posts_to_configured_base_url_and_emits_text_events_async(server_api)
                .await;
        });
}

#[test]
fn local_openai_text_creates_task_when_request_has_no_active_tasks() {
    let server_api = crate::server::server_api::ServerApiProvider::new_for_test().get();

    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(async move {
            local_openai_text_creates_task_when_request_has_no_active_tasks_async(server_api).await;
        });
}

#[test]
fn local_openai_text_uses_selected_model_when_local_model_setting_is_empty() {
    let server_api = crate::server::server_api::ServerApiProvider::new_for_test().get();

    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(async move {
            local_openai_text_uses_selected_model_when_local_model_setting_is_empty_async(
                server_api,
            )
            .await;
        });
}

#[test]
fn local_openai_text_requests_default_completion_token_budget() {
    let server_api = crate::server::server_api::ServerApiProvider::new_for_test().get();

    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(async move {
            local_openai_text_requests_default_completion_token_budget_async(server_api).await;
        });
}

#[test]
fn local_openai_text_emits_first_delta_before_sse_response_finishes() {
    let server_api = crate::server::server_api::ServerApiProvider::new_for_test().get();

    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(async move {
            local_openai_text_emits_first_delta_before_sse_response_finishes_async(server_api)
                .await;
        });
}

async fn local_openai_text_posts_to_configured_base_url_and_emits_text_events_async(
    server_api: std::sync::Arc<crate::server::server_api::ServerApi>,
) {
    use futures_util::StreamExt as _;
    use mockito::{Matcher, Server};

    let mut server = Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/chat/completions")
        .match_header("authorization", "Bearer sk-local")
        .match_body(Matcher::PartialJson(serde_json::json!({
            "model": "local-model",
            "stream": true,
            "messages": [
                {
                    "role": "user",
                    "content": "hello"
                }
            ]
        })))
        .with_status(200)
        .with_body(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\n\
             data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n\
             data: [DONE]\n\n",
        )
        .create_async()
        .await;

    let mut params = request_params_with_ask_user_question_enabled(false);
    params.input = vec![user_query_input("hello")];
    params.tasks = vec![api::Task {
        id: "task-1".to_string(),
        description: "test".to_string(),
        ..Default::default()
    }];
    params.backend = MultiAgentBackend::LocalOpenAI(LocalOpenAIBackendSettings {
        api_key: Some("sk-local".to_string()),
        base_url: Some(format!("{}/v1", server.url())),
        model: Some("local-model".to_string()),
    });

    let (_tx, rx) = futures::channel::oneshot::channel();
    let mut stream = super::generate_multi_agent_output(server_api, params, rx)
        .await
        .expect("stream should be created");

    let first = stream.next().await.expect("init event").expect("init ok");
    assert!(matches!(
        first.r#type,
        Some(api::response_event::Type::Init(_))
    ));

    let second = stream.next().await.expect("add event").expect("add ok");
    let third = stream
        .next()
        .await
        .expect("append event")
        .expect("append ok");
    let fourth = stream
        .next()
        .await
        .expect("finished event")
        .expect("finish ok");

    assert!(matches!(
        fourth.r#type,
        Some(api::response_event::Type::Finished(_))
    ));

    let text_from_event = |event: api::ResponseEvent| -> String {
        let Some(api::response_event::Type::ClientActions(actions)) = event.r#type else {
            panic!("expected client actions");
        };
        let action = actions.actions.into_iter().next().unwrap().action.unwrap();
        match action {
            api::client_action::Action::AddMessagesToTask(add) => {
                let message = add.messages.into_iter().next().unwrap();
                let Some(api::message::Message::AgentOutput(output)) = message.message else {
                    panic!("expected agent output");
                };
                output.text
            }
            api::client_action::Action::AppendToMessageContent(append) => {
                assert_eq!(
                    append.mask.unwrap().paths,
                    vec![super::super::local_openai::tests_support::AGENT_OUTPUT_FIELD_MASK]
                );
                let message = append.message.unwrap();
                let Some(api::message::Message::AgentOutput(output)) = message.message else {
                    panic!("expected agent output");
                };
                output.text
            }
            other => panic!("unexpected action: {other:?}"),
        }
    };

    assert_eq!(text_from_event(second), "hel");
    assert_eq!(text_from_event(third), "lo");
    mock.assert_async().await;
}

async fn local_openai_text_emits_first_delta_before_sse_response_finishes_async(
    server_api: std::sync::Arc<crate::server::server_api::ServerApi>,
) {
    use axum::{
        body::Body,
        http::{header::CONTENT_TYPE, Response},
        routing::post,
        Router,
    };
    use bytes::Bytes;
    use futures_util::StreamExt as _;
    use std::{convert::Infallible, time::Duration};

    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            let body = async_stream::stream! {
                yield Ok::<_, Infallible>(Bytes::from_static(
                    b"data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\n",
                ));
                tokio::time::sleep(Duration::from_millis(750)).await;
                yield Ok::<_, Infallible>(Bytes::from_static(
                    b"data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n\
                      data: [DONE]\n\n",
                ));
            };

            Response::builder()
                .header(CONTENT_TYPE, "text/event-stream")
                .body(Body::from_stream(body))
                .unwrap()
        }),
    );
    let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
        .await
        .unwrap();
    let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let mut params = request_params_with_ask_user_question_enabled(false);
    params.input = vec![user_query_input("hello")];
    params.tasks = vec![api::Task {
        id: "task-1".to_string(),
        description: "test".to_string(),
        ..Default::default()
    }];
    params.backend = MultiAgentBackend::LocalOpenAI(LocalOpenAIBackendSettings {
        api_key: Some("sk-local".to_string()),
        base_url: Some(base_url),
        model: Some("local-model".to_string()),
    });

    let (_tx, rx) = futures::channel::oneshot::channel();
    let mut stream = super::generate_multi_agent_output(server_api, params, rx)
        .await
        .expect("stream should be created");

    assert!(matches!(
        stream
            .next()
            .await
            .expect("init event")
            .expect("init ok")
            .r#type,
        Some(api::response_event::Type::Init(_))
    ));

    let first_delta = tokio::time::timeout(Duration::from_millis(250), stream.next())
        .await
        .expect("first delta should arrive before the SSE response completes")
        .expect("first delta event")
        .expect("first delta ok");
    let Some(api::response_event::Type::ClientActions(actions)) = first_delta.r#type else {
        panic!("expected client actions");
    };
    let action = actions.actions.into_iter().next().unwrap().action.unwrap();
    let api::client_action::Action::AddMessagesToTask(add) = action else {
        panic!("expected add messages action");
    };
    let message = add.messages.into_iter().next().unwrap();
    let Some(api::message::Message::AgentOutput(output)) = message.message else {
        panic!("expected agent output");
    };
    assert_eq!(output.text, "hel");
}

async fn local_openai_text_requests_default_completion_token_budget_async(
    server_api: std::sync::Arc<crate::server::server_api::ServerApi>,
) {
    use futures_util::StreamExt as _;
    use mockito::{Matcher, Server};

    let mut server = Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/chat/completions")
        .match_body(Matcher::PartialJson(serde_json::json!({
            "model": "local-model",
            "max_tokens": 4096,
            "stream": true,
            "messages": [
                {
                    "role": "user",
                    "content": "hello"
                }
            ]
        })))
        .with_status(200)
        .with_body("data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n")
        .create_async()
        .await;

    let mut params = request_params_with_ask_user_question_enabled(false);
    params.input = vec![user_query_input("hello")];
    params.tasks = vec![api::Task {
        id: "task-1".to_string(),
        description: "test".to_string(),
        ..Default::default()
    }];
    params.backend = MultiAgentBackend::LocalOpenAI(LocalOpenAIBackendSettings {
        api_key: Some("sk-local".to_string()),
        base_url: Some(format!("{}/v1", server.url())),
        model: Some("local-model".to_string()),
    });

    let (_tx, rx) = futures::channel::oneshot::channel();
    let mut stream = super::generate_multi_agent_output(server_api, params, rx)
        .await
        .expect("stream should be created");

    assert!(matches!(
        stream
            .next()
            .await
            .expect("init event")
            .expect("init ok")
            .r#type,
        Some(api::response_event::Type::Init(_))
    ));
    assert!(
        stream.next().await.expect("add event").is_ok(),
        "local backend should request a multi-token completion budget"
    );

    mock.assert_async().await;
}

async fn local_openai_text_uses_selected_model_when_local_model_setting_is_empty_async(
    server_api: std::sync::Arc<crate::server::server_api::ServerApi>,
) {
    use futures_util::StreamExt as _;
    use mockito::{Matcher, Server};

    let mut server = Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/chat/completions")
        .match_header("authorization", "Bearer sk-local")
        .match_body(Matcher::PartialJson(serde_json::json!({
            "model": "selected-model",
            "stream": true,
            "messages": [
                {
                    "role": "user",
                    "content": "hello"
                }
            ]
        })))
        .with_status(200)
        .with_body("data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n")
        .create_async()
        .await;

    let mut params = request_params_with_ask_user_question_enabled(false);
    params.input = vec![user_query_input("hello")];
    params.model = LLMId::from("selected-model");
    params.tasks = vec![api::Task {
        id: "task-1".to_string(),
        description: "test".to_string(),
        ..Default::default()
    }];
    params.backend = MultiAgentBackend::LocalOpenAI(LocalOpenAIBackendSettings {
        api_key: Some("sk-local".to_string()),
        base_url: Some(format!("{}/v1", server.url())),
        model: None,
    });

    let (_tx, rx) = futures::channel::oneshot::channel();
    let mut stream = super::generate_multi_agent_output(server_api, params, rx)
        .await
        .expect("stream should be created");

    assert!(matches!(
        stream
            .next()
            .await
            .expect("init event")
            .expect("init ok")
            .r#type,
        Some(api::response_event::Type::Init(_))
    ));
    assert!(
        stream.next().await.expect("add event").is_ok(),
        "selected model request should match the local endpoint mock"
    );

    mock.assert_async().await;
}

async fn local_openai_text_creates_task_when_request_has_no_active_tasks_async(
    server_api: std::sync::Arc<crate::server::server_api::ServerApi>,
) {
    use futures_util::StreamExt as _;
    use mockito::{Matcher, Server};

    let mut server = Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/chat/completions")
        .match_header("authorization", "Bearer sk-local")
        .match_body(Matcher::PartialJson(serde_json::json!({
            "model": "local-model",
            "stream": true,
            "messages": [
                {
                    "role": "user",
                    "content": "hello"
                }
            ]
        })))
        .with_status(200)
        .with_body("data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n")
        .create_async()
        .await;

    let mut params = request_params_with_ask_user_question_enabled(false);
    params.input = vec![user_query_input("hello")];
    params.backend = MultiAgentBackend::LocalOpenAI(LocalOpenAIBackendSettings {
        api_key: Some("sk-local".to_string()),
        base_url: Some(format!("{}/v1", server.url())),
        model: Some("local-model".to_string()),
    });

    let (_tx, rx) = futures::channel::oneshot::channel();
    let mut stream = super::generate_multi_agent_output(server_api, params, rx)
        .await
        .expect("stream should be created");

    let init_event = stream.next().await.expect("init event").expect("init ok");
    let Some(api::response_event::Type::Init(init)) = init_event.r#type else {
        panic!("expected init event");
    };
    assert!(
        init.conversation_id.is_empty(),
        "new local backend conversations should not claim a cloud conversation id"
    );

    let create_event = stream
        .next()
        .await
        .expect("create task event")
        .expect("create task ok");
    let Some(api::response_event::Type::ClientActions(actions)) = create_event.r#type else {
        panic!("expected client actions");
    };
    let action = actions.actions.into_iter().next().unwrap().action.unwrap();
    let api::client_action::Action::CreateTask(create_task) = action else {
        panic!("expected create task action");
    };
    let created_task = create_task.task.expect("created task");
    assert!(!created_task.id.is_empty());

    let add_event = stream
        .next()
        .await
        .expect("add message event")
        .expect("add message ok");
    let Some(api::response_event::Type::ClientActions(actions)) = add_event.r#type else {
        panic!("expected client actions");
    };
    let action = actions.actions.into_iter().next().unwrap().action.unwrap();
    let api::client_action::Action::AddMessagesToTask(add) = action else {
        panic!("expected add messages action");
    };
    assert_eq!(add.task_id, created_task.id);

    mock.assert_async().await;
}
