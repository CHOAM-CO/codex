use crate::auth::SharedAuthProvider;
use crate::common::Reasoning;
use crate::common::ResponseStream;
use crate::common::ResponsesApiRequest;
use crate::common::ResponsesApiTools;
use crate::common::TextControls;
use crate::endpoint::session::EndpointSession;
use crate::error::ApiError;
use crate::provider::Provider;
use crate::requests::Compression;
use crate::requests::headers::build_session_headers;
use crate::requests::headers::insert_header;
use crate::requests::headers::subagent_header;
use crate::sse::spawn_response_stream;
use crate::telemetry::SseTelemetry;
use codex_client::EncodedJsonBody;
use codex_client::HttpTransport;
use codex_client::RequestCompression;
use codex_client::RequestTelemetry;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::SessionSource;
use http::HeaderMap;
use http::HeaderValue;
use http::Method;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use std::sync::Arc;
use std::sync::OnceLock;
use tracing::instrument;
use url::Url;

/// Count only the fields accepted by the Responses input-token count API. Base
/// instructions are already part of the prepared request's `input` items.
#[derive(Serialize)]
struct InputTokenCountRequest<'a> {
    model: &'a str,
    input: &'a [ResponseItem],
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a ResponsesApiTools>,
    tool_choice: &'a str,
    parallel_tool_calls: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<&'a Reasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<&'a TextControls>,
}

#[derive(Deserialize)]
struct InputTokenCountResponse {
    object: InputTokenCountObject,
    input_tokens: u64,
}

#[derive(Deserialize)]
enum InputTokenCountObject {
    #[serde(rename = "response.input_tokens")]
    InputTokens,
}

pub struct ResponsesClient<T: HttpTransport> {
    session: EndpointSession<T>,
    sse_telemetry: Option<Arc<dyn SseTelemetry>>,
}

#[derive(Default)]
pub struct ResponsesOptions {
    pub session_id: Option<String>,
    pub thread_id: Option<String>,
    pub session_source: Option<SessionSource>,
    pub extra_headers: HeaderMap,
    pub compression: Compression,
    pub turn_state: Option<Arc<OnceLock<String>>>,
}

impl<T: HttpTransport> ResponsesClient<T> {
    pub fn new(transport: T, provider: Provider, auth: SharedAuthProvider) -> Self {
        Self {
            session: EndpointSession::new(transport, provider, auth),
            sse_telemetry: None,
        }
    }

    pub fn with_telemetry(
        self,
        request: Option<Arc<dyn RequestTelemetry>>,
        sse: Option<Arc<dyn SseTelemetry>>,
    ) -> Self {
        Self {
            session: self.session.with_request_telemetry(request),
            sse_telemetry: sse,
        }
    }

    /// Resolve the configured endpoint using provider-relative path semantics or
    /// an absolute URL, preserving endpoint and provider query parameters.
    pub fn input_tokens_request_url(
        provider: &Provider,
        endpoint: &str,
    ) -> Result<String, ApiError> {
        let invalid_endpoint = || ApiError::InvalidRequest {
            message: "input_tokens_endpoint must be a valid URL or provider-relative path"
                .to_string(),
        };
        let mut url = match Url::parse(endpoint) {
            Ok(url) => url,
            Err(url::ParseError::RelativeUrlWithoutBase) => Url::parse(&format!(
                "{}/{}",
                provider.base_url.trim_end_matches('/'),
                endpoint.trim_start_matches('/'),
            ))
            .map_err(|_| invalid_endpoint())?,
            Err(_) => return Err(invalid_endpoint()),
        };
        if let Some(params) = &provider.query_params {
            url.query_pairs_mut().extend_pairs(params);
        }
        Ok(url.into())
    }

    /// Count a fully prepared request without generating a response. The selected
    /// endpoint uses the same provider authentication, headers, and retry policy.
    pub async fn count_input_tokens(
        &self,
        request: &ResponsesApiRequest,
        endpoint: &str,
        extra_headers: HeaderMap,
    ) -> Result<i64, ApiError> {
        let request_url = Self::input_tokens_request_url(self.session.provider(), endpoint)?;
        let body = serde_json::to_value(InputTokenCountRequest {
            model: &request.model,
            input: &request.input,
            tools: request.tools.as_ref(),
            tool_choice: &request.tool_choice,
            parallel_tool_calls: request.parallel_tool_calls,
            reasoning: request.reasoning.as_ref(),
            text: request.text.as_ref(),
        })
        .map_err(|_| ApiError::Stream("failed to encode input-token count request".to_string()))?;
        let response = self
            .session
            .execute_with(Method::POST, endpoint, extra_headers, Some(body), |req| {
                req.url.clone_from(&request_url);
            })
            .await?;
        let InputTokenCountResponse {
            object: InputTokenCountObject::InputTokens,
            input_tokens,
        } = serde_json::from_slice(&response.body).map_err(|_| {
            ApiError::Stream("invalid Responses input-token count response".to_string())
        })?;
        i64::try_from(input_tokens).map_err(|_| {
            ApiError::Stream("Responses input-token count exceeds the supported range".to_string())
        })
    }

    #[instrument(
        name = "responses.stream_request",
        level = "info",
        skip_all,
        fields(
            transport = "responses_http",
            http.method = "POST",
            api.path = "/responses"
        )
    )]
    pub async fn stream_request(
        &self,
        request: ResponsesApiRequest,
        options: ResponsesOptions,
    ) -> Result<ResponseStream, ApiError> {
        let ResponsesOptions {
            session_id,
            thread_id,
            session_source,
            extra_headers,
            compression,
            turn_state,
        } = options;
        let body = EncodedJsonBody::encode(&request)
            .map_err(|e| ApiError::Stream(format!("failed to encode responses request: {e}")))?;

        let mut headers = extra_headers;
        if let Some(ref thread_id) = thread_id {
            insert_header(&mut headers, "x-client-request-id", thread_id);
        }
        headers.extend(build_session_headers(session_id, thread_id));
        if let Some(subagent) = subagent_header(&session_source) {
            insert_header(&mut headers, "x-openai-subagent", &subagent);
        }

        self.stream_encoded(body, headers, compression, turn_state)
            .await
    }

    #[instrument(
        name = "responses.stream",
        level = "info",
        skip_all,
        fields(
            transport = "responses_http",
            http.method = "POST",
            api.path = "/responses",
            turn.has_state = turn_state.is_some()
        )
    )]
    pub async fn stream(
        &self,
        body: Value,
        extra_headers: HeaderMap,
        compression: Compression,
        turn_state: Option<Arc<OnceLock<String>>>,
    ) -> Result<ResponseStream, ApiError> {
        let body = EncodedJsonBody::encode(&body)
            .map_err(|e| ApiError::Stream(format!("failed to encode responses request: {e}")))?;
        self.stream_encoded(body, extra_headers, compression, turn_state)
            .await
    }

    async fn stream_encoded(
        &self,
        body: EncodedJsonBody,
        extra_headers: HeaderMap,
        compression: Compression,
        turn_state: Option<Arc<OnceLock<String>>>,
    ) -> Result<ResponseStream, ApiError> {
        let request_compression = match compression {
            Compression::None => RequestCompression::None,
            Compression::Zstd => RequestCompression::Zstd,
        };

        let stream_response = self
            .session
            .stream_encoded_json_with(
                Method::POST,
                "/responses",
                extra_headers,
                Some(body),
                |req| {
                    req.headers.insert(
                        http::header::ACCEPT,
                        HeaderValue::from_static("text/event-stream"),
                    );
                    req.compression = request_compression;
                },
            )
            .await?;

        Ok(spawn_response_stream(
            stream_response,
            self.session.provider().stream_idle_timeout,
            self.sse_telemetry.clone(),
            turn_state,
        ))
    }
}

#[cfg(test)]
#[path = "responses_input_tokens_tests.rs"]
mod input_tokens_tests;
