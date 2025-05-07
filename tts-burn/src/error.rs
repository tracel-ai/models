use thiserror::Error;

#[derive(Error, Debug)]
pub enum TtsError {
    #[error("Invalid input text: {0}")]
    InvalidText(String),

    #[error("Invalid audio data: {0}")]
    InvalidAudio(String),

    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Preprocessing error: {0}")]
    PreprocessingError(String),

    #[error("Vocoder error: {0}")]
    VocoderError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

pub type Result<T> = std::result::Result<T, TtsError>; 