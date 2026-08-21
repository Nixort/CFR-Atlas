// Copyright Nixort <https://github.com/Nixort/CFR-Atlas> 2026.
//
// License: MIT
//! Typed Ollama API integration for CFR-Atlas.
//!
//! This crate discovers Ollama models, captures model metadata useful to CFR
//! validation records, and wraps documented non-streaming generation and
//! embedding endpoints. It intentionally does **not** implement
//! [`cfr_atlas::KvRegenerator`]: the standard Ollama HTTP API does not expose
//! per-layer K/V tensors or deterministic page replay required for an exact
//! CFR-Atlas adapter. Call [`OllamaClient::require_exact_kv_access`] before an
//! integration attempts to enable virtual-K/V execution; it fails closed.

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

/// Default base URL for a locally running Ollama service.
pub const DEFAULT_OLLAMA_BASE_URL: &str = "http://localhost:11434";

/// HTTP method accepted by an [`OllamaTransport`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpMethod {
    /// A retrieval request.
    Get,
    /// A JSON submission request.
    Post,
}

/// One transport request constructed by [`OllamaClient`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OllamaRequest {
    /// HTTP verb.
    pub method: HttpMethod,
    /// Absolute request URL.
    pub url: String,
    /// Optional UTF-8 JSON request body.
    pub body: Option<String>,
}

/// One transport response returned to [`OllamaClient`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OllamaResponse {
    /// HTTP status code.
    pub status: u16,
    /// UTF-8 response body.
    pub body: String,
}

/// Error returned by the Ollama integration layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OllamaError {
    /// The configured base URL is not a supported HTTP(S) URL.
    InvalidBaseUrl(String),
    /// The injected transport could not perform the request.
    Transport(String),
    /// Ollama returned a non-success HTTP status.
    HttpStatus {
        /// HTTP status code.
        status: u16,
        /// Response text returned by the service.
        body: String,
    },
    /// A response did not match the documented JSON envelope.
    Decode(String),
    /// The public Ollama API does not provide the exact K/V tensor contract.
    ExactKvAccessUnavailable,
}

impl std::fmt::Display for OllamaError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidBaseUrl(url) => write!(formatter, "invalid Ollama base URL: {url}"),
            Self::Transport(error) => write!(formatter, "Ollama transport error: {error}"),
            Self::HttpStatus { status, body } => {
                write!(formatter, "Ollama returned HTTP {status}: {body}")
            }
            Self::Decode(error) => write!(formatter, "cannot decode Ollama response: {error}"),
            Self::ExactKvAccessUnavailable => write!(
                formatter,
                "the public Ollama API does not expose exact per-layer K/V tensors or page replay"
            ),
        }
    }
}

impl std::error::Error for OllamaError {}

/// Result alias used by this crate.
pub type Result<T> = std::result::Result<T, OllamaError>;

/// Minimal HTTP boundary used by [`OllamaClient`].
///
/// Production callers receive [`StdHttpTransport`]; tests may provide a deterministic
/// in-memory implementation without starting Ollama.
pub trait OllamaTransport {
    /// Sends one fully formed request and returns the raw response.
    fn send(&self, request: &OllamaRequest) -> Result<OllamaResponse>;
}

/// Dependency-free blocking transport for a local HTTP Ollama endpoint.
///
/// It intentionally accepts only `http://` URLs. Ollama binds its local API to
/// loopback by default; callers that place Ollama behind HTTPS, a proxy, or a
/// custom authentication layer can provide their own [`OllamaTransport`].
#[derive(Debug, Clone, Copy)]
pub struct StdHttpTransport {
    timeout: Duration,
}

impl Default for StdHttpTransport {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
        }
    }
}

impl StdHttpTransport {
    /// Creates a transport with the supplied socket timeout.
    #[must_use]
    pub const fn with_timeout(timeout: Duration) -> Self {
        Self { timeout }
    }
}

impl OllamaTransport for StdHttpTransport {
    fn send(&self, request: &OllamaRequest) -> Result<OllamaResponse> {
        let endpoint = HttpEndpoint::parse(&request.url)?;
        let mut stream = endpoint.connect(self.timeout)?;
        stream
            .set_read_timeout(Some(self.timeout))
            .map_err(|error| OllamaError::Transport(error.to_string()))?;
        stream
            .set_write_timeout(Some(self.timeout))
            .map_err(|error| OllamaError::Transport(error.to_string()))?;
        let method = match request.method {
            HttpMethod::Get => "GET",
            HttpMethod::Post => "POST",
        };
        let body = request.body.as_deref().unwrap_or("");
        let headers = if request.body.is_some() {
            "content-type: application/json\r\n"
        } else {
            ""
        };
        let request_text = format!(
            "{method} {} HTTP/1.1\r\nHost: {}\r\naccept: application/json\r\n{headers}content-length: {}\r\nconnection: close\r\n\r\n{body}",
            endpoint.path, endpoint.host, body.len()
        );
        stream
            .write_all(request_text.as_bytes())
            .map_err(|error| OllamaError::Transport(error.to_string()))?;
        let mut response = String::new();
        stream
            .read_to_string(&mut response)
            .map_err(|error| OllamaError::Transport(error.to_string()))?;
        parse_http_response(&response)
    }
}

struct HttpEndpoint {
    host: String,
    addresses: Vec<std::net::SocketAddr>,
    path: String,
}

impl HttpEndpoint {
    fn parse(url: &str) -> Result<Self> {
        let rest = url
            .strip_prefix("http://")
            .ok_or_else(|| OllamaError::InvalidBaseUrl(url.to_owned()))?;
        let (authority, path) = rest
            .find('/')
            .map_or((rest, "/"), |index| (&rest[..index], &rest[index..]));
        let addresses: Vec<_> = authority
            .to_socket_addrs()
            .map_err(|_| OllamaError::InvalidBaseUrl(url.to_owned()))?
            .collect();
        if addresses.is_empty() {
            return Err(OllamaError::InvalidBaseUrl(url.to_owned()));
        }
        Ok(Self {
            host: authority.to_owned(),
            addresses,
            path: path.to_owned(),
        })
    }

    fn connect(&self, timeout: Duration) -> Result<TcpStream> {
        let mut last_error = None;
        for address in &self.addresses {
            match TcpStream::connect_timeout(address, timeout) {
                Ok(stream) => return Ok(stream),
                Err(error) => last_error = Some(error),
            }
        }
        let message = last_error.map_or_else(
            || "endpoint resolved without a reachable address".to_owned(),
            |error| error.to_string(),
        );
        Err(OllamaError::Transport(message))
    }
}

fn parse_http_response(response: &str) -> Result<OllamaResponse> {
    let (head, raw_body) = response.split_once("\r\n\r\n").ok_or_else(|| {
        OllamaError::Decode("HTTP response is missing a header boundary".to_owned())
    })?;
    let status = head
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| OllamaError::Decode("HTTP response is missing a status".to_owned()))?
        .parse::<u16>()
        .map_err(|error| OllamaError::Decode(error.to_string()))?;
    let body = if is_chunked(head) {
        decode_chunked_body(raw_body)?
    } else {
        raw_body.to_owned()
    };
    Ok(OllamaResponse { status, body })
}

fn is_chunked(head: &str) -> bool {
    head.lines().any(|line| {
        line.split_once(':').is_some_and(|(name, value)| {
            name.eq_ignore_ascii_case("transfer-encoding")
                && value
                    .split(',')
                    .any(|encoding| encoding.trim().eq_ignore_ascii_case("chunked"))
        })
    })
}

fn decode_chunked_body(raw_body: &str) -> Result<String> {
    let bytes = raw_body.as_bytes();
    let mut position = 0;
    let mut decoded = Vec::new();
    loop {
        let line_end = find_crlf(bytes, position).ok_or_else(|| {
            OllamaError::Decode("chunked response is missing a chunk-size line ending".to_owned())
        })?;
        let size_text = std::str::from_utf8(&bytes[position..line_end])
            .map_err(|error| OllamaError::Decode(error.to_string()))?;
        let size_text = size_text.split(';').next().unwrap_or_default().trim();
        let size = usize::from_str_radix(size_text, 16)
            .map_err(|error| OllamaError::Decode(error.to_string()))?;
        position = line_end + 2;
        if size == 0 {
            return String::from_utf8(decoded)
                .map_err(|error| OllamaError::Decode(error.to_string()));
        }
        let chunk_end = position.checked_add(size).ok_or_else(|| {
            OllamaError::Decode("chunked response size overflows address space".to_owned())
        })?;
        let terminator_end = chunk_end.checked_add(2).ok_or_else(|| {
            OllamaError::Decode("chunked response terminator overflows address space".to_owned())
        })?;
        if terminator_end > bytes.len() || &bytes[chunk_end..terminator_end] != b"\r\n" {
            return Err(OllamaError::Decode(
                "chunked response has an incomplete chunk or invalid terminator".to_owned(),
            ));
        }
        decoded.extend_from_slice(&bytes[position..chunk_end]);
        position = terminator_end;
    }
}

fn find_crlf(bytes: &[u8], start: usize) -> Option<usize> {
    bytes[start..]
        .windows(2)
        .position(|window| window == b"\r\n")
        .map(|offset| start + offset)
}

/// Model information reported by `GET /api/tags`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct OllamaModelSummary {
    /// Locally installed model name.
    pub name: String,
    /// Canonical model identifier, when returned by Ollama.
    #[serde(default)]
    pub model: Option<String>,
    /// SHA-256 digest reported by Ollama.
    #[serde(default)]
    pub digest: Option<String>,
    /// Model size on disk in bytes.
    #[serde(default)]
    pub size: Option<u64>,
    /// High-level model details.
    #[serde(default)]
    pub details: OllamaModelDetails,
}

/// High-level model identity returned by Ollama.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub struct OllamaModelDetails {
    /// Model file format, for example `gguf`.
    #[serde(default)]
    pub format: Option<String>,
    /// Primary model family.
    #[serde(default)]
    pub family: Option<String>,
    /// All reported model families.
    #[serde(default)]
    pub families: Vec<String>,
    /// Human-readable parameter-size label.
    #[serde(default)]
    pub parameter_size: Option<String>,
    /// Quantization label.
    #[serde(default)]
    pub quantization_level: Option<String>,
}

/// Attention topology inferred from documented `model_info` values.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OllamaAttentionTopology {
    /// Model architecture prefix, for example `llama` or `gemma4`.
    pub architecture: Option<String>,
    /// Transformer block count, when reported.
    pub block_count: Option<u64>,
    /// Query-head count, when reported.
    pub query_heads: Option<u64>,
    /// K/V-head count, when reported.
    pub kv_heads: Option<u64>,
    /// Context capacity, when reported.
    pub context_length: Option<u64>,
    /// RoPE dimension, when reported.
    pub rope_dimension_count: Option<u64>,
}

/// An auditable snapshot returned by `POST /api/show`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OllamaModelRecord {
    /// Name requested from Ollama.
    pub requested_model: String,
    /// Model name reported by the show endpoint, when available.
    pub name: Option<String>,
    /// SHA-256 model digest, when returned by the service.
    pub digest: Option<String>,
    /// Model file format, for example `gguf`.
    pub format: Option<String>,
    /// Primary model family, for example `llama`.
    pub family: Option<String>,
    /// Quantization label, when reported.
    pub quantization_level: Option<String>,
    /// Optional model capabilities such as `completion` or `vision`.
    pub capabilities: Vec<String>,
    /// High-level model identity retained without lossy normalization.
    pub details: OllamaModelDetails,
    /// Parsed topology fields used to decide whether an extension could be conformant.
    pub topology: OllamaAttentionTopology,
    /// Raw model-info map retained for validation records.
    pub model_info: BTreeMap<String, Value>,
    /// Whether the standard HTTP API can support exact CFR K/V virtualization.
    pub exact_kv_access: ExactKvAccess,
}

/// Exact K/V access status for a discovered Ollama runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactKvAccess {
    /// The documented public endpoint does not expose K/V tensors or replay.
    UnavailableThroughPublicApi,
}

/// Non-streaming documented `POST /api/generate` request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateRequest {
    /// Target Ollama model name.
    pub model: String,
    /// Prompt sent to the model.
    pub prompt: String,
    /// Optional model residency duration, for example `5m` or `0`.
    pub keep_alive: Option<String>,
    /// Optional context-window size sent as `options.num_ctx`.
    pub num_ctx: Option<u64>,
}

/// Selected fields from the final non-streaming generate response.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct GenerateResponse {
    /// Model name that handled the request.
    pub model: String,
    /// Generated response text.
    #[serde(default)]
    pub response: String,
    /// Whether Ollama marked the response complete.
    #[serde(default)]
    pub done: bool,
    /// Input-token count, when returned by Ollama.
    #[serde(default)]
    pub prompt_eval_count: Option<u64>,
    /// Generated-token count, when returned by Ollama.
    #[serde(default)]
    pub eval_count: Option<u64>,
}

/// Documented `POST /api/embed` request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedRequest {
    /// Target Ollama embedding model name.
    pub model: String,
    /// One or more texts to embed.
    pub input: Vec<String>,
    /// Whether Ollama may truncate oversized input.
    pub truncate: bool,
    /// Optional output embedding dimensions.
    pub dimensions: Option<u64>,
    /// Optional model residency duration.
    pub keep_alive: Option<String>,
}

/// Selected fields returned by `POST /api/embed`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct EmbedResponse {
    /// Model name that produced the embeddings.
    pub model: String,
    /// One embedding per input text.
    pub embeddings: Vec<Vec<f32>>,
    /// Input-token count, when returned by Ollama.
    #[serde(default)]
    pub prompt_eval_count: Option<u64>,
}

/// Typed CFR-Atlas client for a configured Ollama base URL.
#[derive(Debug)]
pub struct OllamaClient<T> {
    base_url: String,
    transport: T,
}

impl OllamaClient<StdHttpTransport> {
    /// Creates a production client using the default local blocking transport.
    ///
    /// The standard transport supports an `http://HOST:PORT` endpoint. For an
    /// HTTPS proxy or custom authentication, construct the client with a custom
    /// [`OllamaTransport`] instead.
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        Self::with_transport(base_url, StdHttpTransport::default())
    }
}

impl Default for OllamaClient<StdHttpTransport> {
    /// Connects to [`DEFAULT_OLLAMA_BASE_URL`] with the standard local transport.
    fn default() -> Self {
        Self {
            base_url: DEFAULT_OLLAMA_BASE_URL.to_owned(),
            transport: StdHttpTransport::default(),
        }
    }
}

impl<T: OllamaTransport> OllamaClient<T> {
    /// Creates a client using an injected transport.
    pub fn with_transport(base_url: impl Into<String>, transport: T) -> Result<Self> {
        let base_url = normalize_base_url(base_url.into())?;
        Ok(Self {
            base_url,
            transport,
        })
    }

    /// Returns the configured base URL without a trailing slash.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Lists models available to the Ollama server through `GET /api/tags`.
    pub fn list_models(&self) -> Result<Vec<OllamaModelSummary>> {
        #[derive(Deserialize)]
        struct Envelope {
            models: Vec<OllamaModelSummary>,
        }
        let envelope: Envelope = self.get_json("/api/tags")?;
        Ok(envelope.models)
    }

    /// Retrieves model metadata and derives CFR-relevant topology fields.
    pub fn show_model(&self, model: impl Into<String>) -> Result<OllamaModelRecord> {
        let requested_model = model.into();
        let summary = self.list_models()?.into_iter().find(|candidate| {
            candidate.name == requested_model
                || candidate.model.as_deref() == Some(requested_model.as_str())
        });
        let response: ShowResponse =
            self.post_json("/api/show", &json!({ "model": requested_model }))?;
        let details = response.details;
        let summary_details = summary.as_ref().map(|candidate| &candidate.details);
        Ok(OllamaModelRecord {
            requested_model: requested_model.clone(),
            name: response.name.or(Some(requested_model)),
            digest: response.digest.or_else(|| {
                summary
                    .as_ref()
                    .and_then(|candidate| candidate.digest.clone())
            }),
            format: details
                .format
                .clone()
                .or_else(|| summary_details.and_then(|candidate| candidate.format.clone())),
            family: details
                .family
                .clone()
                .or_else(|| summary_details.and_then(|candidate| candidate.family.clone())),
            quantization_level: details.quantization_level.clone().or_else(|| {
                summary_details.and_then(|candidate| candidate.quantization_level.clone())
            }),
            capabilities: response.capabilities,
            topology: topology_from_info(&response.model_info),
            model_info: response.model_info.into_iter().collect(),
            details,
            exact_kv_access: ExactKvAccess::UnavailableThroughPublicApi,
        })
    }

    /// Calls documented non-streaming generation and returns the final response envelope.
    pub fn generate(&self, request: &GenerateRequest) -> Result<GenerateResponse> {
        let mut body = Map::new();
        body.insert("model".to_owned(), Value::String(request.model.clone()));
        body.insert("prompt".to_owned(), Value::String(request.prompt.clone()));
        body.insert("stream".to_owned(), Value::Bool(false));
        if let Some(keep_alive) = &request.keep_alive {
            body.insert("keep_alive".to_owned(), Value::String(keep_alive.clone()));
        }
        if let Some(num_ctx) = request.num_ctx {
            body.insert("options".to_owned(), json!({ "num_ctx": num_ctx }));
        }
        self.post_json("/api/generate", &Value::Object(body))
    }

    /// Calls documented embedding generation for one or more texts.
    pub fn embed(&self, request: &EmbedRequest) -> Result<EmbedResponse> {
        let mut body = Map::new();
        body.insert("model".to_owned(), Value::String(request.model.clone()));
        body.insert(
            "input".to_owned(),
            Value::Array(request.input.iter().cloned().map(Value::String).collect()),
        );
        body.insert("truncate".to_owned(), Value::Bool(request.truncate));
        if let Some(dimensions) = request.dimensions {
            body.insert("dimensions".to_owned(), Value::from(dimensions));
        }
        if let Some(keep_alive) = &request.keep_alive {
            body.insert("keep_alive".to_owned(), Value::String(keep_alive.clone()));
        }
        self.post_json("/api/embed", &Value::Object(body))
    }

    /// Returns the exact-K/V capability provided by the standard public API.
    #[must_use]
    pub const fn exact_kv_access(&self) -> ExactKvAccess {
        ExactKvAccess::UnavailableThroughPublicApi
    }

    /// Fails closed because public Ollama endpoints do not provide exact K/V access.
    pub const fn require_exact_kv_access(&self) -> Result<()> {
        Err(OllamaError::ExactKvAccessUnavailable)
    }

    fn get_json<R: for<'de> Deserialize<'de>>(&self, path: &str) -> Result<R> {
        Self::decode_response(self.transport.send(&OllamaRequest {
            method: HttpMethod::Get,
            url: self.url(path),
            body: None,
        })?)
    }

    fn post_json<B: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<R> {
        let body =
            serde_json::to_string(body).map_err(|error| OllamaError::Decode(error.to_string()))?;
        Self::decode_response(self.transport.send(&OllamaRequest {
            method: HttpMethod::Post,
            url: self.url(path),
            body: Some(body),
        })?)
    }

    fn decode_response<R: for<'de> Deserialize<'de>>(response: OllamaResponse) -> Result<R> {
        if !(200..300).contains(&response.status) {
            return Err(OllamaError::HttpStatus {
                status: response.status,
                body: response.body,
            });
        }
        serde_json::from_str(&response.body).map_err(|error| OllamaError::Decode(error.to_string()))
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }
}

#[derive(Deserialize)]
struct ShowResponse {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    digest: Option<String>,
    #[serde(default)]
    capabilities: Vec<String>,
    #[serde(default)]
    details: OllamaModelDetails,
    #[serde(default)]
    model_info: BTreeMap<String, Value>,
}

fn normalize_base_url(base_url: String) -> Result<String> {
    let normalized = base_url.trim().trim_end_matches('/').to_owned();
    if normalized.starts_with("http://") || normalized.starts_with("https://") {
        Ok(normalized)
    } else {
        Err(OllamaError::InvalidBaseUrl(base_url))
    }
}

fn topology_from_info(model_info: &BTreeMap<String, Value>) -> OllamaAttentionTopology {
    let architecture = model_info
        .get("general.architecture")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let block_count = number_for(model_info, architecture.as_deref(), "block_count");
    let query_heads = number_for(model_info, architecture.as_deref(), "attention.head_count");
    let kv_heads = number_for(
        model_info,
        architecture.as_deref(),
        "attention.head_count_kv",
    );
    let context_length = number_for(model_info, architecture.as_deref(), "context_length");
    let rope_dimension_count =
        number_for(model_info, architecture.as_deref(), "rope.dimension_count");
    OllamaAttentionTopology {
        architecture,
        block_count,
        query_heads,
        kv_heads,
        context_length,
        rope_dimension_count,
    }
}

fn number_for(
    model_info: &BTreeMap<String, Value>,
    prefix: Option<&str>,
    suffix: &str,
) -> Option<u64> {
    let prefix = prefix?;
    model_info
        .get(&format!("{prefix}.{suffix}"))
        .and_then(Value::as_u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::VecDeque;

    #[derive(Default)]
    struct MockTransport {
        requests: RefCell<Vec<OllamaRequest>>,
        responses: RefCell<VecDeque<OllamaResponse>>,
    }

    impl MockTransport {
        fn with_response(status: u16, body: &str) -> Self {
            Self::with_responses(&[(status, body)])
        }

        fn with_responses(responses: &[(u16, &str)]) -> Self {
            Self {
                requests: RefCell::new(Vec::new()),
                responses: RefCell::new(
                    responses
                        .iter()
                        .map(|(status, body)| OllamaResponse {
                            status: *status,
                            body: (*body).to_owned(),
                        })
                        .collect(),
                ),
            }
        }
    }

    impl OllamaTransport for MockTransport {
        fn send(&self, request: &OllamaRequest) -> Result<OllamaResponse> {
            self.requests.borrow_mut().push(request.clone());
            self.responses
                .borrow_mut()
                .pop_front()
                .ok_or_else(|| OllamaError::Transport("mock has no queued response".to_owned()))
        }
    }

    #[test]
    fn chunked_http_response_is_decoded_before_json_parsing() {
        let response = parse_http_response(
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n6\r\n{\"ok\":\r\n5\r\ntrue}\r\n0\r\n\r\n",
        )
        .unwrap();
        assert_eq!(response.status, 200);
        assert_eq!(response.body, r#"{"ok":true}"#);
    }

    #[test]
    fn list_models_uses_documented_tags_endpoint() {
        let transport = MockTransport::with_response(
            200,
            r#"{"models":[{"name":"llama3.2","digest":"abc","size":42,"details":{"format":"gguf","family":"llama"}}]}"#,
        );
        let client = OllamaClient::with_transport("http://127.0.0.1:11434/", transport).unwrap();
        let models = client.list_models().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "llama3.2");
        assert_eq!(models[0].details.family.as_deref(), Some("llama"));
        assert_eq!(
            client.transport.requests.borrow()[0].url,
            "http://127.0.0.1:11434/api/tags"
        );
    }

    #[test]
    fn show_model_derives_topology_but_refuses_exact_kv() {
        let transport = MockTransport::with_responses(&[
            (
                200,
                r#"{"models":[{"name":"llama3.2","digest":"sha256:abc","details":{"format":"gguf","family":"llama","quantization_level":"Q4_K_M"}}]}"#,
            ),
            (
                200,
                r#"{"name":"llama3.2","capabilities":["completion"],"details":{"format":"gguf","family":"llama","quantization_level":"Q4_K_M"},"model_info":{"general.architecture":"llama","llama.block_count":32,"llama.attention.head_count":32,"llama.attention.head_count_kv":8,"llama.context_length":131072,"llama.rope.dimension_count":128}}"#,
            ),
        ]);
        let client = OllamaClient::with_transport("http://127.0.0.1:11434", transport).unwrap();
        let record = client.show_model("llama3.2").unwrap();
        assert_eq!(record.name.as_deref(), Some("llama3.2"));
        assert_eq!(record.digest.as_deref(), Some("sha256:abc"));
        assert_eq!(record.family.as_deref(), Some("llama"));
        assert_eq!(record.topology.query_heads, Some(32));
        assert_eq!(record.topology.kv_heads, Some(8));
        assert_eq!(record.topology.context_length, Some(131_072));
        assert_eq!(
            record.exact_kv_access,
            ExactKvAccess::UnavailableThroughPublicApi
        );
        assert_eq!(
            client.exact_kv_access(),
            ExactKvAccess::UnavailableThroughPublicApi
        );
        assert_eq!(
            client.require_exact_kv_access(),
            Err(OllamaError::ExactKvAccessUnavailable)
        );
    }

    #[test]
    fn generate_sets_non_streaming_options() {
        let transport = MockTransport::with_response(
            200,
            r#"{"model":"llama3.2","response":"ok","done":true,"prompt_eval_count":4,"eval_count":1}"#,
        );
        let client = OllamaClient::with_transport("http://127.0.0.1:11434", transport).unwrap();
        let response = client
            .generate(&GenerateRequest {
                model: "llama3.2".to_owned(),
                prompt: "hello".to_owned(),
                keep_alive: Some("5m".to_owned()),
                num_ctx: Some(8192),
            })
            .unwrap();
        assert!(response.done);
        let request = &client.transport.requests.borrow()[0];
        assert_eq!(request.method, HttpMethod::Post);
        assert_eq!(request.url, "http://127.0.0.1:11434/api/generate");
        let body: Value = serde_json::from_str(request.body.as_deref().unwrap()).unwrap();
        assert_eq!(body["stream"], Value::Bool(false));
        assert_eq!(body["options"]["num_ctx"], Value::from(8192));
    }

    #[test]
    fn embed_constructs_batch_request_and_decodes_vectors() {
        let transport = MockTransport::with_response(
            200,
            r#"{"model":"nomic-embed-text","embeddings":[[0.25,-0.5],[1.0,0.0]],"prompt_eval_count":7}"#,
        );
        let client = OllamaClient::with_transport("http://127.0.0.1:11434", transport).unwrap();
        let response = client
            .embed(&EmbedRequest {
                model: "nomic-embed-text".to_owned(),
                input: vec!["first".to_owned(), "second".to_owned()],
                truncate: false,
                dimensions: Some(2),
                keep_alive: None,
            })
            .unwrap();
        assert_eq!(response.embeddings.len(), 2);
        let request = &client.transport.requests.borrow()[0];
        assert_eq!(request.url, "http://127.0.0.1:11434/api/embed");
        let body: Value = serde_json::from_str(request.body.as_deref().unwrap()).unwrap();
        assert_eq!(body["input"], json!(["first", "second"]));
        assert_eq!(body["truncate"], Value::Bool(false));
        assert_eq!(body["dimensions"], Value::from(2));
    }

    #[test]
    fn non_success_status_is_not_silently_decoded() {
        let transport = MockTransport::with_response(404, r#"{"error":"model not found"}"#);
        let client = OllamaClient::with_transport("http://127.0.0.1:11434", transport).unwrap();
        let error = client.list_models().unwrap_err();
        assert_eq!(
            error,
            OllamaError::HttpStatus {
                status: 404,
                body: r#"{"error":"model not found"}"#.to_owned(),
            }
        );
    }
}
