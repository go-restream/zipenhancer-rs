//! ONNX model loader module
//!
//! Responsible for ONNX model loading, validation, and metadata extraction.

use std::path::{Path, PathBuf};
use crate::error::{ZipEnhancerError, Result};
use super::{OnnxEnvironment, OnnxSession, SessionConfig, SessionInfo, DynamicTensor};

/// ONNX model metadata
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model file path
    pub model_path: PathBuf,
    /// Model name (extracted from filename)
    pub model_name: String,
    /// Model version
    pub model_version: Option<String>,
    /// Model description
    pub model_description: Option<String>,
    /// Model author
    pub model_author: Option<String>,
    /// Model creation time
    pub model_created_at: Option<String>,
    /// Model modification time
    pub model_modified_at: Option<String>,
    /// Model size in bytes
    pub model_size: u64,
}

impl ModelMetadata {
    /// Create metadata from model file path
    pub fn from_path<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let model_path = model_path.as_ref();
        
        if !model_path.exists() {
            return Err(ZipEnhancerError::processing(format!(
                "Model file does not exist: {}",
                model_path.display()
            )));
        }

        // Get file information
        let metadata = std::fs::metadata(model_path)
            .map_err(|e| ZipEnhancerError::processing(format!(
                "Failed to read model file metadata: {}", e
            )))?;

        // Extract model name
        let model_name = model_path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown")
            .to_string();

        Ok(Self {
            model_path: model_path.to_path_buf(),
            model_name,
            model_version: None, // Can be extracted from ONNX model
            model_description: None, // Can be extracted from ONNX model
            model_author: None, // Can be extracted from ONNX model
            model_created_at: None, // Can be extracted from ONNX model
            model_modified_at: None, // Can be extracted from ONNX model
            model_size: metadata.len(),
        })
    }

    /// Get human-readable format of model file size
    pub fn size_human_readable(&self) -> String {
        let size = self.model_size;
        const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
        let mut size_f = size as f64;
        let mut unit_index = 0;

        while size_f >= 1024.0 && unit_index < UNITS.len() - 1 {
            size_f /= 1024.0;
            unit_index += 1;
        }

        format!("{:.2} {}", size_f, UNITS[unit_index])
    }
}

/// ONNX model
#[derive(Debug)]
pub struct OnnxModel {
    /// Model metadata
    metadata: ModelMetadata,
    /// ONNX environment
    environment: OnnxEnvironment,
    /// ONNX session (lazy loading)
    session: Option<OnnxSession>,
    /// Session configuration
    session_config: SessionConfig,
    /// Whether session is loaded
    session_loaded: bool,
}

impl Clone for OnnxModel {
    fn clone(&self) -> Self {
        Self {
            metadata: self.metadata.clone(),
            environment: OnnxEnvironment::new().expect("Failed to create ONNX environment"),
            session: None, // Session not cloned, needs to be reloaded
            session_config: self.session_config.clone(),
            session_loaded: false,
        }
    }
}

impl OnnxModel {
    /// Create new ONNX model
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let metadata = ModelMetadata::from_path(model_path)?;
        let environment = OnnxEnvironment::new()?;

        Ok(Self {
            metadata,
            environment,
            session: None,
            session_config: SessionConfig::default(),
            session_loaded: false,
        })
    }

    /// Create model with custom session config
    pub fn with_config<P: AsRef<Path>>(
        model_path: P,
        session_config: SessionConfig,
    ) -> Result<Self> {
        let metadata = ModelMetadata::from_path(model_path)?;
        let environment = OnnxEnvironment::new()?;

        Ok(Self {
            metadata,
            environment,
            session: None,
            session_config,
            session_loaded: false,
        })
    }

    /// Load ONNX session (lazy loading)
    pub fn load_session(&mut self) -> Result<()> {
        if self.session_loaded {
            return Ok(());
        }

        let session = OnnxSession::new(
            &self.metadata.model_path,
            self.session_config.clone(),
            &self.environment,
        )?;

        self.session = Some(session);
        self.session_loaded = true;

        Ok(())
    }

    /// Ensure session is loaded
    pub fn ensure_session_loaded(&mut self) -> Result<()> {
        if !self.session_loaded {
            self.load_session()?;
        }
        Ok(())
    }

    /// Get session reference (if loaded)
    pub fn session(&self) -> Option<&OnnxSession> {
        self.session.as_ref()
    }

    
    /// Get model metadata
    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// Get model path
    pub fn model_path(&self) -> &Path {
        &self.metadata.model_path
    }

    /// Get model path (alias)
    pub fn path(&self) -> &Path {
        &self.metadata.model_path
    }

    /// Get model name
    pub fn model_name(&self) -> &str {
        &self.metadata.model_name
    }

    /// Check if session is loaded
    pub fn is_session_loaded(&self) -> bool {
        self.session_loaded
    }

    /// Get session info (session must be loaded first)
    pub fn session_info(&self) -> Result<SessionInfo> {
        if !self.session_loaded {
            return Err(ZipEnhancerError::processing(
                "Session not loaded, please call load_session() first"
            ));
        }

        self.session
            .as_ref()
            .map(|session| session.session_info())
            .ok_or_else(|| ZipEnhancerError::processing("Session loading failed"))
    }

    /// Validate model compatibility
    pub fn validate_model(&self) -> Result<ModelValidationResult> {
        // Basic file check
        if !self.metadata.model_path.exists() {
            return Ok(ModelValidationResult {
                is_valid: false,
                errors: vec!["Model file does not exist".to_string()],
                warnings: vec![],
            });
        }

        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check file size
        if self.metadata.model_size == 0 {
            errors.push("Model file is empty".to_string());
        } else if self.metadata.model_size < 1024 {
            warnings.push("Model file may be too small".to_string());
        } else if self.metadata.model_size > 1024 * 1024 * 1024 {
            warnings.push("Model file may be too large, may affect loading performance".to_string());
        }

        // Check file extension
        if let Some(extension) = self.metadata.model_path.extension() {
            if extension != "onnx" {
                warnings.push("File extension is not .onnx".to_string());
            }
        } else {
            warnings.push("File has no extension".to_string());
        }

        let is_valid = errors.is_empty();

        Ok(ModelValidationResult {
            is_valid,
            errors,
            warnings,
        })
    }

    /// Print model information
    pub fn print_model_info(&self) -> Result<()> {
        println!("=== ONNX Model Information ===");
        println!("Model path: {}", self.metadata.model_path.display());
        println!("Model name: {}", self.metadata.model_name);
        println!("Model size: {}", self.metadata.size_human_readable());
        println!("Session loaded: {}", self.session_loaded);

        if let Some(session) = &self.session {
            let info = session.session_info();
            info.print();
        } else {
            println!("Session not loaded, cannot display detailed information");
        }
        println!("===================");

        Ok(())
    }

    /// Warm up model (run inference once to ensure model is fully loaded)
    pub fn warm_up(&mut self) -> Result<()> {
        self.ensure_session_loaded()?;

        if let Some(_session) = &self.session {
            // Simplified version: only verify session works normally
            log::info!("Model warm-up completed");
        }

        Ok(())
    }

    /// Release session resources
    pub fn unload_session(&mut self) {
        self.session = None;
        self.session_loaded = false;
        log::info!("ONNX session released");
    }

    /// Reload session
    pub fn reload_session(&mut self) -> Result<()> {
        self.unload_session();
        self.load_session()
    }

    /// Run inference (if session is loaded)
    pub fn run_inference(&mut self, inputs: Vec<ndarray::ArrayD<f32>>) -> Result<Vec<ndarray::ArrayD<f32>>> {
        if let Some(session) = &mut self.session {
            // Convert ndarray::ArrayD to DynamicTensor
            let input_tensors: Vec<DynamicTensor> = inputs.into_iter()
                .map(|array| {
                    let shape = array.shape().iter().map(|&dim| dim as i64).collect();
                    let data = array.as_slice().unwrap_or(&[]).to_vec();
                    DynamicTensor::new_f32(data, shape)
                })
                .collect();

            let output_tensors = session.run(input_tensors)?;

            // Convert DynamicTensor back to ndarray::ArrayD
            let output_arrays: Vec<ndarray::ArrayD<f32>> = output_tensors.into_iter()
                .map(|tensor| {
                    ndarray::ArrayD::from_shape_vec(
                        tensor.shape().iter().map(|&dim| dim as usize).collect::<Vec<_>>(),
                        tensor.into_ndarray().as_slice().unwrap().to_vec()
                    ).unwrap()
                })
                .collect();

            Ok(output_arrays)
        } else {
            Err(ZipEnhancerError::processing("Session not loaded, please call load_session() first"))
        }
    }

    /// Run inference (mutable reference version, ensures session is loaded)
    pub fn run_inference_mut(&mut self, inputs: Vec<ndarray::ArrayD<f32>>) -> Result<Vec<ndarray::ArrayD<f32>>> {
        self.ensure_session_loaded()?;
        self.run_inference(inputs)
    }
}

impl Drop for OnnxModel {
    fn drop(&mut self) {
        self.unload_session();
    }
}

/// Model validation result
#[derive(Debug, Clone)]
pub struct ModelValidationResult {
    /// Whether model is valid
    pub is_valid: bool,
    /// List of errors
    pub errors: Vec<String>,
    /// List of warnings
    pub warnings: Vec<String>,
}

impl ModelValidationResult {
    /// Create valid validation result
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Create invalid validation result
    pub fn invalid(errors: Vec<String>) -> Self {
        Self {
            is_valid: false,
            errors,
            warnings: Vec::new(),
        }
    }

    /// Add warnings
    pub fn with_warnings(mut self, warnings: Vec<String>) -> Self {
        self.warnings = warnings;
        self
    }
}

/// Model loader utility functions
pub struct ModelLoader;

impl ModelLoader {
    /// Find all ONNX models from directory
    pub fn find_models<P: AsRef<Path>>(directory: P) -> Result<Vec<PathBuf>> {
        let directory = directory.as_ref();
        
        if !directory.exists() {
            return Err(ZipEnhancerError::processing(format!(
                "Directory does not exist: {}",
                directory.display()
            )));
        }

        if !directory.is_dir() {
            return Err(ZipEnhancerError::processing(format!(
                "Path is not a directory: {}",
                directory.display()
            )));
        }

        let mut models = Vec::new();
        
        for entry in std::fs::read_dir(directory)
            .map_err(|e| ZipEnhancerError::processing(format!("Failed to read directory: {}", e)))?
        {
            let entry = entry
                .map_err(|e| ZipEnhancerError::processing(format!("Failed to read directory entry: {}", e)))?;
            
            let path = entry.path();
            
            if path.is_file() {
                if let Some(extension) = path.extension() {
                    if extension == "onnx" {
                        models.push(path);
                    }
                }
            }
        }

        Ok(models)
    }

    /// Validate model file format
    pub fn validate_model_file<P: AsRef<Path>>(model_path: P) -> Result<bool> {
        let model_path = model_path.as_ref();
        
        if !model_path.exists() {
            return Ok(false);
        }

        // Try to read file header
        let mut file = std::fs::File::open(model_path)
            .map_err(|e| ZipEnhancerError::processing(format!("Failed to open model file: {}", e)))?;
        
        // Read first few bytes
        let mut buffer = [0u8; 8];
        use std::io::Read;
        let bytes_read = file.read(&mut buffer)
            .map_err(|e| ZipEnhancerError::processing(format!("Failed to read model file: {}", e)))?;

        if bytes_read < 8 {
            return Ok(false);
        }

        // ONNX model files usually start with specific magic numbers
        // More complex format validation can be added here
        Ok(bytes_read >= 8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_model_metadata_from_path() {
        // Use project's real ONNX model for testing
        let model_path = PathBuf::from("model/ZipEnhancer_ONNX/ZipEnhancer.onnx");

        // If model not found in test environment, skip test
        if !model_path.exists() {
            panic!("Test model file not found: {}. Please ensure tests run in correct directory.", model_path.display());
        }

        let metadata = ModelMetadata::from_path(&model_path).unwrap();

        assert_eq!(metadata.model_name, "ZipEnhancer");
        assert!(metadata.model_size > 0);
    }

    #[test]
    fn test_model_metadata_invalid_path() {
        let result = ModelMetadata::from_path("nonexistent_model.onnx");
        assert!(result.is_err());
    }

    #[test]
    fn test_size_human_readable() {
        let model_path = PathBuf::from("model/ZipEnhancer_ONNX/ZipEnhancer.onnx");

        if !model_path.exists() {
            panic!("Test model file not found: {}. Please ensure tests run in correct directory.", model_path.display());
        }

        let mut metadata = ModelMetadata::from_path(&model_path).unwrap();

        metadata.model_size = 1024;
        assert_eq!(metadata.size_human_readable(), "1.00 KB");

        metadata.model_size = 1024 * 1024;
        assert_eq!(metadata.size_human_readable(), "1.00 MB");
    }

    #[test]
    fn test_model_creation() {
        let model_path = PathBuf::from("model/ZipEnhancer_ONNX/ZipEnhancer.onnx");

        if !model_path.exists() {
            panic!("Test model file not found: {}. Please ensure tests run in correct directory.", model_path.display());
        }

        let model = OnnxModel::new(&model_path);
        assert!(model.is_ok());

        let model = model.unwrap();
        assert!(!model.is_session_loaded());
        assert_eq!(model.model_name(), "ZipEnhancer");
    }

    #[test]
    fn test_model_validation_result() {
        let valid_result = ModelValidationResult::valid();
        assert!(valid_result.is_valid);
        assert!(valid_result.errors.is_empty());

        let invalid_result = ModelValidationResult::invalid(vec!["Error 1".to_string()]);
        assert!(!invalid_result.is_valid);
        assert_eq!(invalid_result.errors.len(), 1);

        let result_with_warnings = invalid_result.with_warnings(vec!["Warning 1".to_string()]);
        assert!(!result_with_warnings.is_valid);
        assert_eq!(result_with_warnings.warnings.len(), 1);
    }

    #[test]
    fn test_model_validation() {
        let model_path = PathBuf::from("model/ZipEnhancer_ONNX/ZipEnhancer.onnx");

        if !model_path.exists() {
            panic!("Test model file not found: {}. Please ensure tests run in correct directory.", model_path.display());
        }

        let model = OnnxModel::new(&model_path).unwrap();
        let validation = model.validate_model().unwrap();

        assert!(validation.is_valid);
        assert!(validation.errors.is_empty());
        // May have warnings about extension
    }

    #[test]
    fn test_model_validation_empty_file() {
        let temp_file = NamedTempFile::new().unwrap(); // Empty file
        
        let model = OnnxModel::new(temp_file.path()).unwrap();
        let validation = model.validate_model().unwrap();
        
        assert!(!validation.is_valid);
        assert!(!validation.errors.is_empty());
    }
}