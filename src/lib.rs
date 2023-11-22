use std::time::Duration;
use anyhow::{Result, Ok};
use reqwest::{Client, RequestBuilder, Response};

mod api;

pub use api::*;

const TIMEOUT: u64 = 30;

pub struct LLMSDK {
    pub(crate) token: String,
    pub(crate) client: Client,
}

pub trait IntoRequest {
    fn into_request(self, client: Client) -> RequestBuilder;
}

impl LLMSDK {
    pub fn new(token: String) -> Self {
        Self {
            token,
            client: Client::new(),
        }
    }
    
    // pub async fn chat_completion(&self, req: impl IntoRequest) -> Result<ChatCompletionResponse> {
    //     let req = self.prepare_request(req);
    //     let res = req.send().await?;
    //     Ok(res.json::<ChatCompletionResponse>().await?)
    // }
    
    pub async fn create_image(&self, req: impl IntoRequest) -> Result<CreateImageResponse> {
        let req = self.prepare_request(req);
        let res = req.send().await?;
        Ok(res.json::<CreateImageResponse>().await?)
    }
    
    fn prepare_request(&self, req: impl IntoRequest) -> RequestBuilder {
        let req = req.into_request(self.client.clone());
        let req = if self.token.is_empty() {
            req
        } else {
            req.bearer_auth(&self.token)
        };
        req.timeout(Duration::from_secs(TIMEOUT))
    }
}