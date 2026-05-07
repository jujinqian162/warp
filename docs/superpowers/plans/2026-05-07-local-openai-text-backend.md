# Local OpenAI Text Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a settings-gated local OpenAI-compatible text backend so normal Warp AI prompts can stream text from the configured OpenAI API key and base URL without calling Warp's `/ai/multi-agent` endpoint.

**Architecture:** Keep the existing Warp `ResponseEvent` UI contract. Route requests through a small backend abstraction: the existing Warp server backend remains the default, while the new local backend converts a supported `UserQuery` into a Chat Completions streaming request and synthesizes `ResponseEvent::Init`, `ResponseEvent::ClientActions`, and `ResponseEvent::Finished`. Phase 1 intentionally supports text output only and rejects tool/action/resume request types with a clear local-backend error.

**Tech Stack:** Rust, `warp_multi_agent_api`, `reqwest`, `async-stream`, `futures-util`, `serde_json`, `mockito`, existing `ApiKeyManager` secure storage, existing AI settings page UI components.

---

## Scope

This plan implements only the first phase:

- Local OpenAI-compatible backend toggle.
- Local OpenAI-compatible model name setting.
- Plain text streaming for normal `AIAgentInput::UserQuery`.
- No tool calls, no file reads, no command execution, no orchestration, no passive suggestions, no cloud/ambient task support.
- Existing Warp server backend remains the default path.

The plan deliberately does not add a separate HTTP "fake server" process. The local backend runs inside the client because the UI already consumes typed `ResponseEvent` values internally.

## File Structure

- Modify `crates/ai/src/api_keys.rs`
  - Persist local-backend enablement and OpenAI model name next to the existing locally stored API provider settings.
  - Add getters/setters used by the settings page and request routing.

- Modify `app/src/settings_view/ai_page.rs`
  - Add a local backend toggle and OpenAI model input to `ApiKeysWidget`.
  - Wire a new `AISettingsPageAction::ToggleLocalOpenAITextBackend` action to `ApiKeyManager`.
  - Update copy so the base URL/model settings are not misleadingly described as cloud Oz provider routing.

- Modify `app/src/ai/agent/api.rs`
  - Add `MultiAgentBackend` and `LocalOpenAITextBackendSettings`.
  - Add a backend field to `RequestParams`.
  - Resolve the backend from `ApiKeyManager` in `RequestParams::new`.

- Modify `app/src/ai/agent/api/impl.rs`
  - Split the current Warp-server request builder into `generate_warp_server_multi_agent_output`.
  - Dispatch to either the existing Warp server backend or the new local backend.

- Create `app/src/ai/agent/api/local_openai_text.rs`
  - Convert supported request params into a Chat Completions stream.
  - Parse OpenAI-compatible SSE `data:` chunks.
  - Synthesize Warp client actions that append streamed text into the current conversation.

- Modify `app/src/ai/agent/api/impl_tests.rs`
  - Add focused unit tests for backend routing and local backend event synthesis.

- Modify `app/src/ai/agent/api/mod.rs` if needed by module declarations.

---

### Task 1: Persist Local Backend Settings

**Files:**
- Modify: `crates/ai/src/api_keys.rs`

- [ ] **Step 1: Add failing persistence tests**

Add these tests under the existing `#[cfg(test)] mod tests` in `crates/ai/src/api_keys.rs`:

```rust
#[test]
fn local_openai_text_backend_settings_round_trip() {
    let keys: ApiKeys = serde_json::from_str(
        r#"{
            "openai":"sk-test",
            "openai_base_url":"https://proxy.example.com/v1",
            "openai_model":"gpt-local",
            "local_openai_text_backend_enabled":true
        }"#,
    )
    .unwrap();

    assert_eq!(keys.openai.as_deref(), Some("sk-test"));
    assert_eq!(
        keys.openai_base_url.as_deref(),
        Some("https://proxy.example.com/v1")
    );
    assert_eq!(keys.openai_model.as_deref(), Some("gpt-local"));
    assert!(keys.local_openai_text_backend_enabled);

    let stored = serde_json::to_value(keys).unwrap();
    assert_eq!(
        stored
            .get("local_openai_text_backend_enabled")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        stored.get("openai_model").and_then(|value| value.as_str()),
        Some("gpt-local")
    );
}

#[test]
fn openai_model_is_normalized_like_base_url() {
    assert_eq!(
        normalize_optional_string(Some("  gpt-local  ".to_string())),
        Some("gpt-local".to_string())
    );
}
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
cargo test -p ai local_openai_text_backend_settings_round_trip openai_model_is_normalized_like_base_url
```

Expected: the first test fails to compile because `ApiKeys` does not yet expose `openai_model` or `local_openai_text_backend_enabled`.

- [ ] **Step 3: Add the persisted fields and setters**

Update `ApiKeys` and `ApiKeyManager`:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ApiKeys {
    pub google: Option<String>,
    pub anthropic: Option<String>,
    pub openai: Option<String>,
    pub open_router: Option<String>,
    pub openai_base_url: Option<String>,
    #[serde(default)]
    pub openai_model: Option<String>,
    #[serde(default)]
    pub local_openai_text_backend_enabled: bool,
}
```

Add methods inside `impl ApiKeyManager`:

```rust
pub fn set_openai_model(&mut self, model: Option<String>, ctx: &mut ModelContext<Self>) {
    self.keys.openai_model = normalize_optional_string(model);
    ctx.emit(ApiKeyManagerEvent::KeysUpdated);
    self.write_keys_to_secure_storage(ctx);
}

pub fn openai_model(&self) -> Option<&str> {
    self.keys.openai_model.as_deref()
}

pub fn set_local_openai_text_backend_enabled(
    &mut self,
    enabled: bool,
    ctx: &mut ModelContext<Self>,
) {
    self.keys.local_openai_text_backend_enabled = enabled;
    ctx.emit(ApiKeyManagerEvent::KeysUpdated);
    self.write_keys_to_secure_storage(ctx);
}

pub fn is_local_openai_text_backend_enabled(&self) -> bool {
    self.keys.local_openai_text_backend_enabled
}
```

Do not include `openai_model` or `local_openai_text_backend_enabled` in `ApiKeys::has_any_key()`.

- [ ] **Step 4: Run the tests and verify they pass**

Run:

```bash
cargo test -p ai api_keys::tests
```

Expected: all `api_keys::tests` pass.

- [ ] **Step 5: Commit**

```bash
git add crates/ai/src/api_keys.rs
git commit -m "feat: persist local OpenAI text backend settings"
```

---

### Task 2: Add Settings UI Controls

**Files:**
- Modify: `app/src/settings_view/ai_page.rs`

- [ ] **Step 1: Add the new action variant**

In `AISettingsPageAction`, add:

```rust
ToggleLocalOpenAITextBackend,
```

In `TypedActionView for AISettingsPageView`, add this match arm near the other API/BYOK-related actions:

```rust
AISettingsPageAction::ToggleLocalOpenAITextBackend => {
    ApiKeyManager::handle(ctx).update(ctx, |manager, ctx| {
        let enabled = !manager.is_local_openai_text_backend_enabled();
        manager.set_local_openai_text_backend_enabled(enabled, ctx);
    });
    ctx.notify();
}
```

- [ ] **Step 2: Add editor and switch state to `ApiKeysWidget`**

Update the struct:

```rust
struct ApiKeysWidget {
    openai_api_key_editor: ViewHandle<EditorView>,
    openai_base_url_editor: ViewHandle<EditorView>,
    openai_model_editor: ViewHandle<EditorView>,
    anthropic_api_key_editor: ViewHandle<EditorView>,
    google_api_key_editor: ViewHandle<EditorView>,
    open_router_api_key_editor: ViewHandle<EditorView>,
    local_openai_text_backend_switch: SwitchStateHandle,

    can_use_warp_credits_with_byok: SwitchStateHandle,
    upgrade_highlight_index: HighlightedHyperlink,
}
```

Update the `ApiKeys` destructuring in `ApiKeysWidget::new`:

```rust
let ApiKeys {
    openai: openai_key,
    openai_base_url,
    openai_model,
    anthropic: anthropic_key,
    google: google_key,
    open_router: open_router_key,
    ..
} = ApiKeyManager::as_ref(ctx).keys().clone();
```

Create the model editor after the base URL editor:

```rust
create_api_setting_editor!(
    openai_model_editor,
    openai_model,
    set_openai_model,
    "gpt-4o-mini",
    false
);
```

Initialize the new fields in `Self`:

```rust
Self {
    openai_api_key_editor,
    openai_base_url_editor,
    openai_model_editor,
    anthropic_api_key_editor,
    google_api_key_editor,
    open_router_api_key_editor,
    local_openai_text_backend_switch: Default::default(),

    can_use_warp_credits_with_byok: Default::default(),
    upgrade_highlight_index: Default::default(),
}
```

- [ ] **Step 3: Render the local backend toggle and model input**

Inside `render_api_keys_section`, compute the local toggle state:

```rust
let local_backend_enabled =
    ApiKeyManager::as_ref(app).is_local_openai_text_backend_enabled();
```

Add a helper near `render_api_key_input`:

```rust
fn render_local_backend_toggle(
    switch_state: SwitchStateHandle,
    is_enabled: bool,
    is_toggleable: bool,
    app: &AppContext,
) -> Box<dyn Element> {
    let appearance = Appearance::as_ref(app);
    build_toggle_element(
        Text::new_inline(
            "Use local OpenAI-compatible text backend",
            appearance.ui_font_family(),
            CONTENT_FONT_SIZE,
        )
        .with_color(styles::header_font_color(is_toggleable, app).into())
        .finish(),
        render_ai_feature_switch(
            switch_state,
            is_enabled,
            is_toggleable,
            AISettingsPageAction::ToggleLocalOpenAITextBackend,
            app,
        ),
        appearance,
        None,
    )
}
```

Render it before the OpenAI inputs:

```rust
column.add_child(render_local_backend_toggle(
    self.local_openai_text_backend_switch.clone(),
    local_backend_enabled,
    is_enabled,
    app,
));
```

Render the model input after base URL:

```rust
column.add_child(render_api_key_input(
    appearance,
    "OpenAI Model",
    self.openai_model_editor.clone(),
    is_enabled,
    app,
));
```

Change the description string at the top of the section to this exact text:

```rust
"Use your own API keys from model providers. OpenAI API Key, OpenAI Base URL, and OpenAI Model are also used by the local OpenAI-compatible text backend when it is enabled. Provider keys are stored locally and never synced to the cloud."
```

- [ ] **Step 4: Run formatting**

Run:

```bash
cargo fmt -- --check
```

Expected: if formatting fails, run `cargo fmt -- app/src/settings_view/ai_page.rs crates/ai/src/api_keys.rs`, then rerun `cargo fmt -- --check`.

- [ ] **Step 5: Commit**

```bash
git add app/src/settings_view/ai_page.rs
git commit -m "feat: expose local OpenAI text backend settings"
```

---

### Task 3: Add Backend Selection Types

**Files:**
- Modify: `app/src/ai/agent/api.rs`
- Modify: `app/src/ai/agent/api/impl.rs`

- [ ] **Step 1: Add backend types to `api.rs`**

Add these types near `RequestParams`:

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultiAgentBackend {
    WarpServer,
    LocalOpenAIText(LocalOpenAITextBackendSettings),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalOpenAITextBackendSettings {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub model: Option<String>,
}
```

Add a field to `RequestParams`:

```rust
pub backend: MultiAgentBackend,
```

- [ ] **Step 2: Resolve backend in `RequestParams::new`**

After `let ai_settings = AISettings::as_ref(app);`, capture API key manager settings:

```rust
let api_key_manager = ApiKeyManager::as_ref(app);
let backend = if api_key_manager.is_local_openai_text_backend_enabled() {
    MultiAgentBackend::LocalOpenAIText(LocalOpenAITextBackendSettings {
        api_key: api_key_manager.keys().openai.clone(),
        base_url: api_key_manager.openai_base_url().map(ToOwned::to_owned),
        model: api_key_manager.openai_model().map(ToOwned::to_owned),
    })
} else {
    MultiAgentBackend::WarpServer
};
```

Use the existing `ApiKeyManager::as_ref(app)` call later for BYOK request keys or reuse `api_key_manager` before borrowing rules require a shorter scope.

Set the new field in `Self`:

```rust
backend,
```

- [ ] **Step 3: Update existing tests/build helpers that construct `RequestParams`**

Every manual `RequestParams { ... }` literal must include:

```rust
backend: MultiAgentBackend::WarpServer,
```

In `app/src/ai/agent/api/impl_tests.rs`, import the enum:

```rust
use crate::ai::agent::api::{MultiAgentBackend, RequestParams};
```

- [ ] **Step 4: Split Warp server generation**

In `app/src/ai/agent/api/impl.rs`, change the current function body into a dispatcher:

```rust
pub async fn generate_multi_agent_output(
    server_api: Arc<ServerApi>,
    params: RequestParams,
    cancellation_rx: futures::channel::oneshot::Receiver<()>,
) -> Result<ResponseStream, ConvertToAPITypeError> {
    match params.backend.clone() {
        super::MultiAgentBackend::WarpServer => {
            generate_warp_server_multi_agent_output(server_api, params, cancellation_rx).await
        }
        super::MultiAgentBackend::LocalOpenAIText(settings) => {
            super::local_openai_text::generate_text_output(settings, params, cancellation_rx).await
        }
    }
}
```

Rename the current implementation to:

```rust
async fn generate_warp_server_multi_agent_output(
    server_api: Arc<ServerApi>,
    mut params: RequestParams,
    cancellation_rx: futures::channel::oneshot::Receiver<()>,
) -> Result<ResponseStream, ConvertToAPITypeError> {
    // existing body unchanged
}
```

- [ ] **Step 5: Run compile check for the focused module**

Run:

```bash
cargo test -p warp supported_tools_ --no-run
```

Expected: compile succeeds after all manual `RequestParams` literals include `backend`.

- [ ] **Step 6: Commit**

```bash
git add app/src/ai/agent/api.rs app/src/ai/agent/api/impl.rs app/src/ai/agent/api/impl_tests.rs
git commit -m "feat: add multi-agent backend selection"
```

---

### Task 4: Implement Local OpenAI-Compatible Text Stream

**Files:**
- Create: `app/src/ai/agent/api/local_openai_text.rs`
- Modify: `app/src/ai/agent/api.rs`
- Modify: `app/src/ai/agent/api/impl.rs`

- [ ] **Step 1: Declare the module**

In `app/src/ai/agent/api.rs`, add:

```rust
mod local_openai_text;
```

Keep it private; only `impl.rs` should call it through `super::local_openai_text::generate_text_output`.

- [ ] **Step 2: Add the local backend skeleton**

Create `app/src/ai/agent/api/local_openai_text.rs` with:

```rust
use std::sync::Arc;

use anyhow::anyhow;
use async_stream::stream;
use futures::channel::oneshot;
use futures_util::{StreamExt, TryStreamExt};
use serde::Deserialize;
use uuid::Uuid;
use warp_multi_agent_api as api;

use crate::{
    ai::agent::{
        api::{
            Event, LocalOpenAITextBackendSettings, RequestParams, ResponseStream,
        },
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
    let output_stream = local_text_stream(settings, params).take_until(cancellation_rx);
    Ok(Box::pin(output_stream))
}
```

- [ ] **Step 3: Add validation and request extraction**

Add:

```rust
fn require_setting(value: Option<String>, name: &'static str) -> Result<String, AIApiError> {
    value
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| AIApiError::Other(anyhow!("{name} is required for the local OpenAI text backend")))
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

fn active_task_id(params: &RequestParams) -> Result<String, AIApiError> {
    params
        .tasks
        .first()
        .map(|task| task.id.trim().to_owned())
        .filter(|id| !id.is_empty())
        .ok_or_else(|| AIApiError::Other(anyhow!(
            "Local OpenAI text backend requires an active task"
        )))
}
```

- [ ] **Step 4: Add OpenAI-compatible request/response structs**

Add:

```rust
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
```

- [ ] **Step 5: Add event helpers**

Add:

```rust
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
```

- [ ] **Step 6: Add SSE parsing**

Add:

```rust
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
```

This first implementation parses the full response body before emitting deltas. It still uses the provider's streaming format and keeps the Warp event synthesis isolated. Replace `response.text().await?` with incremental `bytes_stream()` parsing when optimizing visible token latency.

- [ ] **Step 7: Add the response stream**

Add:

```rust
fn local_text_stream(
    settings: LocalOpenAITextBackendSettings,
    params: RequestParams,
) -> impl futures_core::Stream<Item = Event> + Send + 'static {
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
        let task_id = match active_task_id(&params) {
            Ok(value) => value,
            Err(error) => {
                yield Err(Arc::new(error));
                return;
            }
        };

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

        let mut has_message = false;
        for delta in deltas {
            let message = agent_output_message(&message_id, &task_id, &request_id, delta);
            if has_message {
                yield Ok(append_message_event(&task_id, message));
            } else {
                has_message = true;
                yield Ok(add_message_event(&task_id, message));
            }
        }

        if !has_message {
            yield Ok(add_message_event(
                &task_id,
                agent_output_message(
                    &message_id,
                    &task_id,
                    &request_id,
                    String::new(),
                ),
            ));
        }

        yield Ok(finished_event());
    }
}
```

If `futures_core` is not already available through direct dependencies, change the return type to `impl futures_util::Stream<Item = Event> + Send + 'static` or add the existing workspace dependency explicitly only if compilation requires it.

- [ ] **Step 8: Run compile check**

Run:

```bash
cargo test -p warp local_openai_text --no-run
```

Expected: compile succeeds. If the compiler rejects the stream return type, use the return type adjustment described in Step 7 and rerun.

- [ ] **Step 9: Commit**

```bash
git add app/src/ai/agent/api.rs app/src/ai/agent/api/impl.rs app/src/ai/agent/api/local_openai_text.rs
git commit -m "feat: stream local OpenAI-compatible text responses"
```

---

### Task 5: Add Local Backend Tests

**Files:**
- Modify: `app/src/ai/agent/api/impl_tests.rs`
- Modify: `app/src/ai/agent/api/local_openai_text.rs`

- [ ] **Step 1: Expose test-only helpers**

At the bottom of `local_openai_text.rs`, add:

```rust
#[cfg(test)]
pub(crate) mod tests_support {
    pub(crate) use super::{
        active_task_id, extract_user_query, parse_sse_delta, AGENT_OUTPUT_FIELD_MASK,
        DEFAULT_LOCAL_MODEL,
    };
}
```

- [ ] **Step 2: Add parser tests**

In `app/src/ai/agent/api/impl_tests.rs`, add:

```rust
#[test]
fn local_openai_text_parses_chat_completion_delta() {
    let line = r#"data: {"choices":[{"delta":{"content":"hello"}}]}"#;

    let delta =
        super::local_openai_text::tests_support::parse_sse_delta(line).expect("valid SSE line");

    assert_eq!(delta.as_deref(), Some("hello"));
}

#[test]
fn local_openai_text_ignores_done_delta() {
    let delta =
        super::local_openai_text::tests_support::parse_sse_delta("data: [DONE]").unwrap();

    assert_eq!(delta, None);
}
```

- [ ] **Step 3: Add request extraction tests**

Add a helper in `impl_tests.rs`:

```rust
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
```

Add tests:

```rust
#[test]
fn local_openai_text_extracts_plain_user_query() {
    let input = vec![user_query_input(" hello ")];

    let query =
        super::local_openai_text::tests_support::extract_user_query(&input).unwrap();

    assert_eq!(query, "hello");
}

#[test]
fn local_openai_text_rejects_non_user_query() {
    let input = vec![crate::ai::agent::AIAgentInput::ResumeConversation {
        context: std::sync::Arc::from([]),
    }];

    let error =
        super::local_openai_text::tests_support::extract_user_query(&input).unwrap_err();

    assert!(format!("{error:?}").contains("only supports plain user queries"));
}
```

- [ ] **Step 4: Add event synthesis test with mockito**

Add an async test:

```rust
#[tokio::test]
async fn local_openai_text_posts_to_configured_base_url_and_emits_text_events() {
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

    let params = RequestParams {
        input: vec![user_query_input("hello")],
        conversation_token: None,
        forked_from_conversation_token: None,
        ambient_agent_task_id: None,
        tasks: vec![api::Task {
            id: "task-1".to_string(),
            description: "test".to_string(),
            ..Default::default()
        }],
        existing_suggestions: None,
        metadata: None,
        session_context: SessionContext::new_for_test(),
        model: LLMId::from("ignored"),
        coding_model: LLMId::from("ignored"),
        cli_agent_model: LLMId::from("ignored"),
        computer_use_model: LLMId::from("ignored"),
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
        ask_user_question_enabled: false,
        research_agent_enabled: false,
        orchestration_enabled: false,
        supported_tools_override: None,
        parent_agent_id: None,
        agent_name: None,
        backend: MultiAgentBackend::LocalOpenAIText(LocalOpenAITextBackendSettings {
            api_key: Some("sk-local".to_string()),
            base_url: Some(format!("{}/v1", server.url())),
            model: Some("local-model".to_string()),
        }),
    };

    let (_tx, rx) = futures::channel::oneshot::channel();
    let mut stream = super::generate_multi_agent_output(
        crate::server::server_api::ServerApi::new_for_test().into(),
        params,
        rx,
    )
    .await
    .expect("stream should be created");

    let first = stream.next().await.expect("init event").expect("init ok");
    assert!(matches!(
        first.r#type,
        Some(api::response_event::Type::Init(_))
    ));

    let second = stream.next().await.expect("add event").expect("add ok");
    let third = stream.next().await.expect("append event").expect("append ok");
    let fourth = stream.next().await.expect("finished event").expect("finish ok");

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
                    vec![super::local_openai_text::tests_support::AGENT_OUTPUT_FIELD_MASK]
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
```

Add imports at the top of `impl_tests.rs` as needed:

```rust
use crate::ai::agent::api::{
    LocalOpenAITextBackendSettings, MultiAgentBackend, RequestParams,
};
use crate::ai::blocklist::SessionContext;
use crate::ai::llms::LLMId;
use warp_multi_agent_api as api;
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
cargo test -p warp local_openai_text
```

Expected: all local backend tests pass.

- [ ] **Step 6: Commit**

```bash
git add app/src/ai/agent/api/impl_tests.rs app/src/ai/agent/api/local_openai_text.rs
git commit -m "test: cover local OpenAI text backend"
```

---

### Task 6: Final Verification and Cleanup

**Files:**
- Review: all files changed by Tasks 1-5.

- [ ] **Step 1: Run API key tests**

Run:

```bash
cargo test -p ai api_keys::tests
```

Expected: pass.

- [ ] **Step 2: Run local backend tests**

Run:

```bash
cargo test -p warp local_openai_text
```

Expected: pass.

- [ ] **Step 3: Run existing BYOK/base URL focused tests**

Run:

```bash
cargo test -p warp openai_base_url
cargo test -p warp byo_api_key_enabled
cargo test -p warp test_has_any_ai_remaining
```

Expected: all pass.

- [ ] **Step 4: Run formatting and diff checks**

Run:

```bash
cargo fmt -- --check
git diff --check
```

Expected: both pass.

- [ ] **Step 5: Manual smoke test**

Build/run Warp using the repo's normal local run command for this checkout. In Settings -> AI/API Keys:

1. Enter `OpenAI API Key`.
2. Enter `OpenAI Base URL`.
3. Enter `OpenAI Model`.
4. Enable `Use local OpenAI-compatible text backend`.
5. Send `hello` in the normal Warp AI input.

Expected logs:

```text
Local OpenAI text backend selected
```

Expected network behavior:

```text
No request is sent to /ai/multi-agent for the prompt.
One request is sent to {OpenAI Base URL}/chat/completions.
```

Expected UI behavior:

```text
The AI response appears as streamed text in the current Warp AI conversation.
No tool call cards or command approvals appear.
```

- [ ] **Step 6: Commit final cleanup**

If Step 4 or Step 5 required changes:

```bash
git add app/src crates/ai/src
git commit -m "fix: polish local OpenAI text backend"
```

If no changes were required, do not create an empty commit.

---

## Self-Review

- Spec coverage: the plan covers local persistence, settings selection, backend routing, OpenAI-compatible text request conversion, Warp `ResponseEvent` synthesis, and focused verification.
- Placeholder scan: no task depends on an unspecified provider or an unbounded tool loop. Unsupported request types produce explicit local backend errors.
- Type consistency: `MultiAgentBackend::LocalOpenAIText(LocalOpenAITextBackendSettings)` is introduced in Task 3 and used by Tasks 4-5. The local backend returns the existing `ResponseStream` alias and existing `Event` type.
- Scope check: tool calls, command execution, file access, orchestration, and cloud ambient agent support are excluded from phase 1 and left behind the backend boundary for a separate implementation plan.
