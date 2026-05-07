pub use crate::aws_credentials::{AwsCredentials, AwsCredentialsState};
use serde::{Deserialize, Serialize};
use warp_multi_agent_api as api;
use warpui::{Entity, ModelContext, SingletonEntity};
use warpui_extras::secure_storage::{self, AppContextExt};

const SECURE_STORAGE_KEY: &str = "AiApiKeys";

/// Emitted when user-provided API keys are updated in-memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApiKeyManagerEvent {
    KeysUpdated,
}

/// User-provided API keys for AI providers.
///
/// These are used for "Bring Your Own API Key" functionality, allowing
/// users to use their own API keys instead of Warp's.
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

impl ApiKeys {
    pub fn has_any_key(&self) -> bool {
        self.openai.is_some()
            || self.anthropic.is_some()
            || self.google.is_some()
            || self.open_router.is_some()
    }
}

/// Controls how AWS credentials are refreshed by [`ApiKeyManager`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum AwsCredentialsRefreshStrategy {
    /// Load credentials from the local AWS credential chain (~/.aws). This is the default.
    #[default]
    LocalChain,
    /// Credentials are managed externally via OIDC/STS.
    /// The task ID is used to scope the STS AssumeRoleWithWebIdentity session.
    /// The role ARN is the IAM role to assume via STS.
    OidcManaged {
        task_id: Option<String>,
        role_arn: String,
    },
}

/// A structure that manages API keys for AI providers.
pub struct ApiKeyManager {
    keys: ApiKeys,
    pub(crate) aws_credentials_state: AwsCredentialsState,
    aws_credentials_refresh_strategy: AwsCredentialsRefreshStrategy,
}

impl ApiKeyManager {
    pub fn new(ctx: &mut ModelContext<Self>) -> Self {
        let keys = Self::load_keys_from_secure_storage(ctx);
        Self {
            keys,
            aws_credentials_state: AwsCredentialsState::Missing,
            aws_credentials_refresh_strategy: AwsCredentialsRefreshStrategy::default(),
        }
    }

    pub fn keys(&self) -> &ApiKeys {
        &self.keys
    }

    pub fn set_google_key(&mut self, key: Option<String>, ctx: &mut ModelContext<Self>) {
        self.keys.google = key;
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn set_anthropic_key(&mut self, key: Option<String>, ctx: &mut ModelContext<Self>) {
        self.keys.anthropic = key;
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn set_openai_key(&mut self, key: Option<String>, ctx: &mut ModelContext<Self>) {
        self.keys.openai = key;
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn set_open_router_key(&mut self, key: Option<String>, ctx: &mut ModelContext<Self>) {
        self.keys.open_router = key;
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn set_openai_base_url(&mut self, base_url: Option<String>, ctx: &mut ModelContext<Self>) {
        self.keys.openai_base_url = normalize_optional_string(base_url);
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn openai_base_url(&self) -> Option<&str> {
        self.keys.openai_base_url.as_deref()
    }

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

    pub fn set_aws_credentials_state(
        &mut self,
        state: AwsCredentialsState,
        ctx: &mut ModelContext<Self>,
    ) {
        self.aws_credentials_state = state;
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
    }

    pub fn aws_credentials_state(&self) -> &AwsCredentialsState {
        &self.aws_credentials_state
    }

    pub fn aws_credentials_refresh_strategy(&self) -> AwsCredentialsRefreshStrategy {
        self.aws_credentials_refresh_strategy.clone()
    }

    pub fn set_aws_credentials_refresh_strategy(
        &mut self,
        strategy: AwsCredentialsRefreshStrategy,
    ) {
        self.aws_credentials_refresh_strategy = strategy;
    }

    pub fn api_keys_for_request(
        &self,
        include_byo_keys: bool,
        include_aws_bedrock_credentials: bool,
    ) -> Option<api::request::settings::ApiKeys> {
        let anthropic = include_byo_keys
            .then(|| self.keys.anthropic.clone())
            .flatten()
            .unwrap_or_default();
        let openai = include_byo_keys
            .then(|| self.keys.openai.clone())
            .flatten()
            .unwrap_or_default();
        let google = include_byo_keys
            .then(|| self.keys.google.clone())
            .flatten()
            .unwrap_or_default();
        let open_router = include_byo_keys
            .then(|| self.keys.open_router.clone())
            .flatten()
            .unwrap_or_default();
        // Also include credentials when running with OIDC-managed Bedrock inference, regardless
        // of the per-user setting flag (which only applies to the local credential chain path).
        let include_aws = include_aws_bedrock_credentials
            || matches!(
                self.aws_credentials_refresh_strategy,
                AwsCredentialsRefreshStrategy::OidcManaged { .. }
            );
        let aws_credentials = include_aws
            .then(|| match self.aws_credentials_state {
                AwsCredentialsState::Loaded {
                    ref credentials, ..
                } => Some(credentials.clone().into()),
                _ => None,
            })
            .flatten();

        if anthropic.is_empty()
            && openai.is_empty()
            && google.is_empty()
            && open_router.is_empty()
            && aws_credentials.is_none()
        {
            None
        } else {
            Some(api::request::settings::ApiKeys {
                anthropic,
                openai,
                google,
                open_router,
                allow_use_of_warp_credits: false,
                aws_credentials,
            })
        }
    }

    fn load_keys_from_secure_storage(ctx: &mut ModelContext<Self>) -> ApiKeys {
        let key_json = match ctx.secure_storage().read_value(SECURE_STORAGE_KEY) {
            Ok(json) => json,
            Err(e) => {
                if !matches!(e, secure_storage::Error::NotFound) {
                    log::error!("Failed to read API keys from secure storage: {e:#}");
                }
                return ApiKeys::default();
            }
        };

        let keys = match serde_json::from_str(&key_json) {
            Ok(keys) => keys,
            Err(e) => {
                log::error!("Failed to deserialize API keys: {e:#}");
                ApiKeys::default()
            }
        };

        keys
    }

    fn write_keys_to_secure_storage(&mut self, ctx: &mut ModelContext<Self>) {
        let keys = self.keys.clone();

        let json = match serde_json::to_string(&keys) {
            Ok(json) => json,
            Err(e) => {
                log::error!("Failed to serialize API keys: {e:#}");
                return;
            }
        };

        if let Err(e) = ctx.secure_storage().write_value(SECURE_STORAGE_KEY, &json) {
            log::error!("Failed to write API keys to secure storage: {e:#}");
        }
    }
}

impl Entity for ApiKeyManager {
    type Event = ApiKeyManagerEvent;
}

impl SingletonEntity for ApiKeyManager {}

fn normalize_optional_string(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let trimmed = value.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_owned())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_keys_storage_payload_round_trips_openai_base_url() {
        let keys: ApiKeys =
            serde_json::from_str(r#"{"openai_base_url":"https://proxy.example.com/v1"}"#).unwrap();
        let stored = serde_json::to_value(keys).unwrap();

        assert_eq!(
            stored
                .get("openai_base_url")
                .and_then(|value| value.as_str()),
            Some("https://proxy.example.com/v1")
        );
    }

    #[test]
    fn openai_base_url_does_not_count_as_api_key() {
        let keys = ApiKeys {
            openai_base_url: Some("https://proxy.example.com/v1".to_string()),
            ..Default::default()
        };

        assert!(!keys.has_any_key());
    }

    #[test]
    fn normalize_optional_string_trims_and_clears_empty_values() {
        assert_eq!(
            normalize_optional_string(Some("  https://proxy.example.com/v1  ".to_string())),
            Some("https://proxy.example.com/v1".to_string())
        );
        assert_eq!(normalize_optional_string(Some("   ".to_string())), None);
        assert_eq!(normalize_optional_string(None), None);
    }

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
}
