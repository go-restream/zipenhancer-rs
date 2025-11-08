//! ONNX Runtime bindings module
//!
//! Implements ONNX model inference using onnxruntime-rs crate,
//! simplifying integration and providing better type safety.

use std::path::Path;
use ndarray::ArrayD;
use onnxruntime::{environment::Environment, session::Session, GraphOptimizationLevel, LoggingLevel};
use crate::error::{Result, ZipEnhancerError};

/// ONNX environment wrapper
#[derive(Debug)]
pub struct OnnxEnvironment {
    env: Environment,
}

impl OnnxEnvironment {
    pub fn new() -> Result<Self> {
        let env = Environment::builder()
            .with_name("zipenhancer")
            .with_log_level(LoggingLevel::Warning)
            .build()
            .map_err(|e| ZipEnhancerError::processing(format!("Failed to create ONNX environment: {}", e)))?;

        Ok(Self { env })
    }
}

impl Default for OnnxEnvironment {
    fn default() -> Self {
        Self::new().expect("Failed to create ONNX environment")
    }
}

/// ONNX inference session
pub struct OnnxSession {
    #[allow(dead_code)] // Keep environment reference to ensure correct lifetime
    environment: OnnxEnvironment,
    session: Option<Session<'static>>,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl std::fmt::Debug for OnnxSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxSession")
            .field("input_names", &self.input_names)
            .field("output_names", &self.output_names)
            .finish()
    }
}

/// Tensor data types
#[derive(Debug, Clone)]
pub enum TensorData {
    Float32(Vec<f32>),
    Int16(Vec<i16>),
}

/// Tensor data
#[derive(Debug, Clone)]
pub struct DynamicTensor {
    data: TensorData,
    shape: Vec<i64>,
}

impl DynamicTensor {
    pub fn new_f32(data: Vec<f32>, shape: Vec<i64>) -> Self {
        Self {
            data: TensorData::Float32(data),
            shape
        }
    }

    pub fn new_i16(data: Vec<i16>, shape: Vec<i64>) -> Self {
        Self {
            data: TensorData::Int16(data),
            shape
        }
    }

    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    pub fn data_type(&self) -> &str {
        match &self.data {
            TensorData::Float32(_) => "float32",
            TensorData::Int16(_) => "int16",
        }
    }

    pub fn into_ndarray(self) -> ArrayD<f32> {
        let shape: Vec<usize> = self.shape.iter().map(|&x| x as usize).collect();
        let float_data = match self.data {
            TensorData::Float32(data) => data,
            TensorData::Int16(data) => {
                // Convert int16 to normalized f32, consistent with Go implementation
                data.into_iter().map(|x| (x as f32) / 32767.0).collect()
            },
        };
        ArrayD::from_shape_vec(shape, float_data)
            .map_err(|e| ZipEnhancerError::processing(format!("Tensor shape conversion failed: {}", e)))
            .unwrap()
    }

    pub fn into_i16_ndarray(self) -> ArrayD<i16> {
        let shape: Vec<usize> = self.shape.iter().map(|&x| x as usize).collect();
        let i16_data = match self.data {
            TensorData::Int16(data) => data,
            TensorData::Float32(data) => data.into_iter().map(|x| (x * 32767.0) as i16).collect(),
        };
        ArrayD::from_shape_vec(shape, i16_data)
            .map_err(|e| ZipEnhancerError::processing(format!("Tensor shape conversion failed: {}", e)))
            .unwrap()
    }
}

/// Session configuration
#[derive(Debug)]
pub struct SessionConfig {
    pub optimization_level: GraphOptimizationLevel,
    pub intra_op_num_threads: i32,
}

impl Clone for SessionConfig {
    fn clone(&self) -> Self {
        Self {
            // Note: GraphOptimizationLevel cannot be cloned directly, so we use default value
            optimization_level: GraphOptimizationLevel::All,
            intra_op_num_threads: self.intra_op_num_threads,
        }
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            optimization_level: GraphOptimizationLevel::All,
            intra_op_num_threads: 4,
        }
    }
}

impl OnnxSession {
    /// Create new ONNX session
    pub fn new(
        model_path: &Path,
        config: SessionConfig,
        _env: &OnnxEnvironment,
    ) -> Result<Self> {

        if !model_path.exists() {
            return Err(ZipEnhancerError::processing(format!(
                "ONNX model file does not exist: {}",
                model_path.display()
            )));
        }

        // Convert path to PathBuf and leak to 'static lifetime
        let model_path_cloned = model_path.to_path_buf();
        let model_path_leaked: &'static std::path::Path = Box::leak(model_path_cloned.into_boxed_path());
        let env_leaked = Box::leak(Box::new(OnnxEnvironment::new()?));

        let session = env_leaked.env.new_session_builder()?
            .with_optimization_level(config.optimization_level)?
            .with_number_threads(config.intra_op_num_threads as i16)?
            .with_model_from_file(model_path_leaked)?;

        // Get input/output information
        let input_names = session.inputs.iter().map(|input| input.name.clone()).collect();
        let output_names = session.outputs.iter().map(|output| output.name.clone()).collect();

        log::info!("ONNX model loaded successfully");
        log::info!("Input info: {:?}", input_names);
        log::info!("Output info: {:?}", output_names);

        Ok(Self {
            environment: OnnxEnvironment::new()?,
            session: Some(session),
            input_names,
            output_names,
        })
    }

    /// Get input names
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Get output names
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// Get input count
    pub fn input_count(&self) -> usize {
        self.input_names.len()
    }

    /// Get output count
    pub fn output_count(&self) -> usize {
        self.output_names.len()
    }

    /// Run inference
    pub fn run(&mut self, inputs: Vec<DynamicTensor>) -> Result<Vec<DynamicTensor>> {
        if inputs.len() != self.input_count() {
            return Err(ZipEnhancerError::processing(
                format!("Input count mismatch: expected {}, actual {}", self.input_count(), inputs.len())
            ));
        }

        // Check input data types and log
        for (i, input) in inputs.iter().enumerate() {
            log::debug!("Input {}: type={}, shape={:?}", i, input.data_type(), input.shape());
        }

        // Convert inputs to ndarray Vec - use i16 conversion since model expects int16
        let input_arrays: Vec<ndarray::ArrayD<i16>> = inputs.into_iter()
            .enumerate()
            .map(|(i, input)| {
                log::info!("Converting input {} to int16 ndarray", i);
                input.into_i16_ndarray()
            })
            .collect();

        log::info!("Executing ONNX model inference...");

        // Run inference
        let outputs = self.session.as_mut().unwrap().run(input_arrays)
            .map_err(|e| ZipEnhancerError::processing(format!("ONNX inference failed: {}", e)))?;

        // Convert outputs to DynamicTensor - convert int16 outputs to f32 for subsequent processing
        let mut result = Vec::new();
        for (i, output_tensor) in outputs.into_iter().enumerate() {
            let shape: Vec<i64> = output_tensor.shape().iter().map(|&x| x as i64).collect();
            let i16_data: &[i16] = output_tensor.as_slice().unwrap();
            let f32_data: Vec<f32> = i16_data.iter().map(|&x| x as f32 / 32767.0).collect();
            let data_len = f32_data.len();
            result.push(DynamicTensor::new_f32(f32_data, shape.clone()));
            log::debug!("Output {}: shape {:?}, data length {}", i, shape, data_len);
        }

        log::info!("ONNX inference completed, output count: {}", result.len());
        Ok(result)
    }

    /// Get session information
    pub fn session_info(&self) -> SessionInfo {
        SessionInfo {
            input_count: self.input_names.len(),
            output_count: self.output_names.len(),
            input_names: self.input_names.clone(),
            output_names: self.output_names.clone(),
            input_shapes: vec![], // onnxruntime-rs doesn't directly provide shape information
            output_shapes: vec![],
        }
    }
}

/// Session information
#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub input_count: usize,
    pub output_count: usize,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub input_shapes: Vec<Vec<i64>>,
    pub output_shapes: Vec<Vec<i64>>,
}

impl SessionInfo {
    pub fn print(&self) {
        println!("=== ONNX Session Information ===");
        println!("Input count: {}", self.input_count);
        println!("Output count: {}", self.output_count);

        println!("\nInput information:");
        for (i, name) in self.input_names.iter().enumerate() {
            println!("  [{}] {} - shape: {:?}", i, name, self.input_shapes[i]);
        }

        println!("\nOutput information:");
        for (i, name) in self.output_names.iter().enumerate() {
            println!("  [{}] {} - shape: {:?}", i, name, self.output_shapes[i]);
        }
        println!("==================");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_tensor() {
        let f32_data = vec![1.0, 2.0, 3.0, 4.0];
        let i16_data = vec![32767, 16384, 0, -16384];
        let shape = vec![2, 2];

        let f32_tensor = DynamicTensor::new_f32(f32_data.clone(), shape.clone());
        let i16_tensor = DynamicTensor::new_i16(i16_data.clone(), shape.clone());

        assert_eq!(f32_tensor.shape(), &shape);
        assert_eq!(f32_tensor.data_type(), "float32");

        assert_eq!(i16_tensor.shape(), &shape);
        assert_eq!(i16_tensor.data_type(), "int16");
    }

    #[test]
    fn test_session_config_default() {
        let config = SessionConfig::default();
        assert_eq!(config.intra_op_num_threads, 4);
        // Since GraphOptimizationLevel doesn't implement PartialEq, we check the variant directly
        match config.optimization_level {
            GraphOptimizationLevel::All => {}, // Expected value
            _ => panic!("Expected GraphOptimizationLevel::All"),
        }
    }
}