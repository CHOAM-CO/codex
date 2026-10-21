use super::*;
use crate::auth::AuthProvider;
use crate::common::OpenAiVerbosity;
use crate::common::ReasoningContext;
use crate::common::TextFormat;
use crate::provider::RetryConfig;
use codex_client::Request;
use codex_client::Response;
use codex_client::StreamResponse;
use codex_client::TransportError;
use http::StatusCode;
use pretty_assertions::assert_eq;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

#[derive(Clone)]
struct CountTransport {
    requests: Arc<Mutex<Vec<Request>>>,
    status: StatusCode,
    body: String,
}

impl CountTransport {
    fn new(status: StatusCode, body: impl Into<String>) -> Self {
        Self {
            requests: Arc::new(Mutex::new(Vec::new())),
            status,
            body: body.into(),
        }
    }
}

impl HttpTransport for CountTransport {
    async fn execute(&self, request: Request) -> Result<Response, TransportError> {
        self.requests.lock().unwrap().push(request);
        if !self.status.is_success() {
            return Err(TransportError::Http {
                status: self.status,
                url: None,
                headers: None,
                body: Some(self.body.clone()),
                retry_after: None,
            });
        }
        Ok(Response {
            status: self.status,
            headers: HeaderMap::new(),
            body: self.body.clone().into(),
        })
    }

    async fn stream(&self, _: Request) -> Result<StreamResponse, TransportError> {
        panic!("counting must not invoke the generation stream")
    }
}

struct CountAuth;

impl AuthProvider for CountAuth {
    fn add_auth_headers(&self, headers: &mut HeaderMap) {
        headers.insert(
            http::header::AUTHORIZATION,
            "Bearer example".parse().unwrap(),
        );
    }
}

fn provider() -> Provider {
    Provider {
        name: "example".to_string(),
        base_url: "https://model.example/v1/".to_string(),
        query_params: Some(HashMap::from([(
            "api-version".to_string(),
            "test 1".to_string(),
        )])),
        headers: HeaderMap::from_iter([(
            "x-provider".parse().unwrap(),
            "example".parse().unwrap(),
        )]),
        retry: RetryConfig {
            max_attempts: 1,
            base_delay: Duration::from_millis(1),
            retry_429: false,
            retry_5xx: false,
            retry_transport: false,
        },
        stream_idle_timeout: Duration::from_secs(1),
    }
}

fn generation_request() -> ResponsesApiRequest {
    let input = json!([
        {"type":"message", "role":"developer", "content":[{"type":"input_text","text":"Keep the instructions."}]},
        {"type":"message", "role":"user", "content":[{"type":"input_text","text":"Count the full conversation."}]},
        {"type":"function_call", "name":"example", "call_id":"call-example", "arguments":"{}"},
        {"type":"function_call_output", "call_id":"call-example", "output":"Keep the tool result."}
    ]);
    let tools = serde_json::value::to_raw_value(&json!([
        {"type":"function", "name":"example", "parameters":{"type":"object"}}
    ]))
    .unwrap();
    ResponsesApiRequest {
        model: "example-model".to_string(),
        input: serde_json::from_value(input).unwrap(),
        tools: Some(Arc::<serde_json::value::RawValue>::from(tools).into()),
        tool_choice: "auto".to_string(),
        parallel_tool_calls: true,
        reasoning: Some(Reasoning {
            effort: None,
            summary: None,
            context: Some(ReasoningContext::AllTurns),
        }),
        text: Some(TextControls {
            verbosity: Some(OpenAiVerbosity::High),
            format: Some(TextFormat {
                name: "answer".to_string(),
                strict: true,
                schema: json!({"type":"object"}),
                ..Default::default()
            }),
        }),
        store: false,
        stream: true,
        stream_options: None,
        include: vec!["reasoning.encrypted_content".to_string()],
        service_tier: Some("priority".to_string()),
        prompt_cache_key: Some("cache-example".to_string()),
        client_metadata: Some(HashMap::from([(
            "private".to_string(),
            "metadata".to_string(),
        )])),
        access_programs: None,
    }
}

#[tokio::test]
async fn count_preserves_prepared_input_and_tools_using_provider_auth_and_custom_endpoint() {
    for (endpoint, expected_url) in [
        (
            "/responses/input_tokens",
            "https://model.example/v1/responses/input_tokens?api-version=test+1",
        ),
        (
            "measure?deployment=example",
            "https://model.example/v1/measure?deployment=example&api-version=test+1",
        ),
        (
            "https://count.example/custom?deployment=example",
            "https://count.example/custom?deployment=example&api-version=test+1",
        ),
    ] {
        let transport = CountTransport::new(
            StatusCode::OK,
            r#"{"object":"response.input_tokens","input_tokens":1234}"#,
        );
        let client = ResponsesClient::new(transport.clone(), provider(), Arc::new(CountAuth));
        let request = generation_request();
        let original = request.clone();
        let headers =
            HeaderMap::from_iter([("x-request".parse().unwrap(), "example".parse().unwrap())]);
        assert_eq!(
            client
                .count_input_tokens(&request, endpoint, headers)
                .await
                .unwrap(),
            1234
        );
        assert_eq!(request, original);
        let requests = transport.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        let sent = &requests[0];
        assert_eq!(sent.method, Method::POST);
        assert_eq!(sent.url, expected_url);
        assert_eq!(sent.headers[http::header::AUTHORIZATION], "Bearer example");
        assert_eq!(sent.headers["x-provider"], "example");
        assert_eq!(sent.headers["x-request"], "example");
        assert_eq!(
            sent.body.as_ref().unwrap().json().unwrap(),
            &json!({
                "model": request.model,
                "input": request.input,
                "tools": request.tools,
                "tool_choice": request.tool_choice,
                "parallel_tool_calls": request.parallel_tool_calls,
                "reasoning": request.reasoning,
                "text": request.text,
            })
        );
    }
}

#[tokio::test]
async fn count_accepts_zero_and_maximum_representable_count() {
    for count in [0, i64::MAX] {
        let body = json!({"object":"response.input_tokens", "input_tokens":count}).to_string();
        let transport = CountTransport::new(StatusCode::OK, body);
        let client = ResponsesClient::new(transport, provider(), Arc::new(CountAuth));
        let mut request = generation_request();
        request.tools = None;
        request.reasoning = None;
        request.text = None;
        assert_eq!(
            client
                .count_input_tokens(&request, "count", HeaderMap::new())
                .await
                .unwrap(),
            count
        );
    }
}

#[tokio::test]
async fn count_rejects_invalid_response_contract_without_returning_a_count() {
    for body in [
        r#"{"object":"response.input_tokens","input_tokens":-1}"#,
        r#"{"object":"response.input_tokens","input_tokens":9223372036854775808}"#,
        r#"{"object":"response.input_tokens","input_tokens":18446744073709551616}"#,
        r#"{"object":"response.input_tokens","input_tokens":1.5}"#,
        r#"{"object":"response.input_tokens","input_tokens":"123"}"#,
        r#"{"object":"response","input_tokens":123}"#,
        r#"{"input_tokens":123}"#,
        r#"{"object":"response.input_tokens"}"#,
        "invalid json",
    ] {
        let transport = CountTransport::new(StatusCode::OK, body);
        let client = ResponsesClient::new(transport.clone(), provider(), Arc::new(CountAuth));
        assert!(matches!(
            client
                .count_input_tokens(&generation_request(), "count", HeaderMap::new())
                .await,
            Err(ApiError::Stream(_))
        ));
        assert_eq!(transport.requests.lock().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn count_preserves_http_errors_for_the_callers_fallback_policy() {
    for status in [
        StatusCode::NOT_FOUND,
        StatusCode::METHOD_NOT_ALLOWED,
        StatusCode::NOT_IMPLEMENTED,
        StatusCode::UNAUTHORIZED,
        StatusCode::INTERNAL_SERVER_ERROR,
    ] {
        let transport = CountTransport::new(status, "upstream error");
        let client = ResponsesClient::new(transport.clone(), provider(), Arc::new(CountAuth));
        let error = client
            .count_input_tokens(&generation_request(), "count", HeaderMap::new())
            .await
            .unwrap_err();
        let ApiError::Transport(TransportError::Http {
            status: actual,
            body,
            ..
        }) = error
        else {
            panic!("expected unchanged HTTP error");
        };
        assert_eq!(actual, status);
        assert_eq!(body.as_deref(), Some("upstream error"));
        assert_eq!(transport.requests.lock().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn count_rejects_a_malformed_endpoint_before_sending_input() {
    let transport = CountTransport::new(StatusCode::OK, "unused");
    let client = ResponsesClient::new(transport.clone(), provider(), Arc::new(CountAuth));
    let error = client
        .count_input_tokens(&generation_request(), "https://[invalid", HeaderMap::new())
        .await
        .unwrap_err();
    assert!(matches!(error, ApiError::InvalidRequest { .. }));
    assert!(transport.requests.lock().unwrap().is_empty());
}
