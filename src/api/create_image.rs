use serde::{Deserialize, Serialize};
use reqwest::{RequestBuilder, Client};

use crate::IntoRequest;

#[derive(Debug, Clone, Default, Serialize)]
pub struct CreateImageRequest {
    /// A text description of the desired image(s). The maximum length is 4000 characters for dall-e-3.
    prompt: String,
    /// The model to use for image generation. Only support Dall-e-3
    model: ImageModel,
    /// The number of images to generate. Must be between 1 and 10. For dall-e-3, only n=1 is supported.
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<usize>,
    /// The quality of the image that will be generated. hd creates images with finer details and greater consistency across the image. This param is only supported for dall-e-3.
    #[serde(skip_serializing_if = "Option::is_none")]
    quality: Option<ImageQuality>,
    /// The format in which the generated images are returned. Must be one of url or b64_json.
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ImageResponseFormat>,
    /// The size of the generated images. Must be one of 1024x1024, 1792x1024, or 1024x1792 for dall-e-3 models.
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<ImageSize>,
    /// The style of the generated images. Must be one of vivid or natural. Vivid causes the model to lean towards generating hyper-real and dramatic images. Natural causes the model to produce more natural, less hyper-real looking images. This param is only supported for dall-e-3.
    #[serde(skip_serializing_if = "Option::is_none")]
    style: Option<ImageStyle>,
    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

impl CreateImageRequest {
    pub fn new(prompt: impl Into<String>) -> Self {
        CreateImageRequest {
            prompt: prompt.into(),
            ..Default::default()
        }
    }
}

impl IntoRequest for CreateImageRequest {
    fn into_request(self, client: Client) -> RequestBuilder {
        client.post("https://api.openai.com/v1/images/generations")
           .json(&self)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct CreateImageResponse {
    created: u64,
    data: Vec<ImageObject>
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
enum ImageModel {
    #[default]
    #[serde(rename = "dall-e-3")]
    DallE3
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ImageQuality {
    #[default]
    Standard,
    Hd,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ImageResponseFormat {
    #[default]
    Url,
    B64Json,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
enum ImageSize {
    #[default]
    #[serde(rename = "1024x1024")]
    Large,
    #[serde(rename = "1792x1024")]
    LargeWide,
    #[serde(rename = "1024x1792")]
    LargeTall,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ImageStyle {
    #[default]
    Vivid,
    Natural,
}

#[derive(Debug, Clone, Deserialize)]
struct ImageObject {
    b64_json: Option<String>,
    url: Option<String>,
    revised_prompt: String,
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::LLMSDK;

    use super::*;
    use anyhow::Result;
    use serde_json::json;

    #[test]
    fn test_image_request_serialize() -> Result<()> {
        let req = CreateImageRequest::new("draw a cute caterpillar");
        assert_eq!(
            serde_json::to_value(req)?, 
            json!({
                "prompt": "draw a cute caterpillar",
                "model": "dall-e-3",
            })
        );
        Ok(())
    }
    
    #[test]
    fn test_image_custom_request_serialize() -> Result<()> {
        let req = CreateImageRequest {
            prompt: "draw a cute caterpillar".into(),
            quality: Some(ImageQuality::Hd),
            style: Some(ImageStyle::Natural),
            ..Default::default()
        };
        assert_eq!(
            serde_json::to_value(req)?, 
            json!({
                "prompt": "draw a cute caterpillar",
                "model": "dall-e-3",
                "quality": "hd",
                "style": "natural",
            })
        );
        Ok(())
    }
    
    #[tokio::test]
    #[ignore]
    async fn test_image_response_deserialize() -> Result<()> {
        let sdk = LLMSDK::new(std::env::var("OPENAI_API_KEY")?);
        let req = CreateImageRequest::new("draw a cute caterpillar");
        let res = sdk.create_image(req).await?;
        assert_eq!(res.data.len(), 1);
        let image = &res.data[0];
        assert!(image.url.is_some());
        assert!(image.b64_json.is_none());
        println!("image: {:?}", image);
        fs::write(
            "/tmp/llm-sdk/caterpillar.png",
            reqwest::get(image.url.as_ref().unwrap())
            .await?
            .bytes()
            .await?,
        )?;
        Ok(())
    }
}