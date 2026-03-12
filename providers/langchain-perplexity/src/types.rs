use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserLocation {
    pub country: String,
    pub city: Option<String>,
    pub region: Option<String>,
}

impl UserLocation {
    pub fn new(country: impl Into<String>) -> Self {
        Self {
            country: country.into(),
            city: None,
            region: None,
        }
    }

    pub fn with_city(mut self, city: impl Into<String>) -> Self {
        self.city = Some(city.into());
        self
    }

    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebSearchOptions {
    pub search_context_size: Option<usize>,
    pub user_location: Option<UserLocation>,
}

impl WebSearchOptions {
    pub fn with_search_context_size(mut self, search_context_size: usize) -> Self {
        self.search_context_size = Some(search_context_size.max(1));
        self
    }

    pub fn with_user_location(mut self, user_location: UserLocation) -> Self {
        self.user_location = Some(user_location);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaResponse {
    pub title: String,
    pub url: String,
}

impl MediaResponse {
    pub fn new(title: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            url: url.into(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaResponseOverrides {
    pub images: Vec<MediaResponse>,
    pub videos: Vec<MediaResponse>,
}

impl MediaResponseOverrides {
    pub fn with_images(mut self, images: Vec<MediaResponse>) -> Self {
        self.images = images;
        self
    }

    pub fn with_videos(mut self, videos: Vec<MediaResponse>) -> Self {
        self.videos = videos;
        self
    }
}
