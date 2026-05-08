use warp_multi_agent_api as api;

pub(super) const AGENT_OUTPUT_FIELD_MASK: &str = "agent_output.text";

pub(super) fn init_event(
    conversation_id: String,
    request_id: String,
    run_id: String,
) -> api::ResponseEvent {
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

pub(super) fn finished_event() -> api::ResponseEvent {
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

pub(super) fn agent_output_message(
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

pub(super) fn add_message_event(task_id: &str, message: api::Message) -> api::ResponseEvent {
    add_messages_event(task_id, vec![message])
}

pub(super) fn add_messages_event(task_id: &str, messages: Vec<api::Message>) -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::ClientActions(
            api::response_event::ClientActions {
                actions: vec![api::ClientAction {
                    action: Some(api::client_action::Action::AddMessagesToTask(
                        api::client_action::AddMessagesToTask {
                            task_id: task_id.to_owned(),
                            messages,
                        },
                    )),
                }],
            },
        )),
    }
}

pub(super) fn create_task_event(task: api::Task) -> api::ResponseEvent {
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

pub(super) fn append_message_event(task_id: &str, message: api::Message) -> api::ResponseEvent {
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
