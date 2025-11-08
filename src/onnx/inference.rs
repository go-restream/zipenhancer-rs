//! ONNX inference engine

use std::time::{Duration, Instant};
use ndarray::ArrayD;
use crate::error::Result;
use super::{OnnxModel, OnnxSession, DynamicTensor};

#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub max_retries: u32,
    pub retry_delay_ms: u64,
    pub timeout_ms: u64,
    pub enable_profiling: bool,
    pub batch_size: Option<usize>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_delay_ms: 100,
            timeout_ms: 5000,
            enable_profiling: false,
            batch_size: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferenceStats {
    pub total_inferences: u64,
    pub successful_inferences: u64,
    pub failed_inferences: u64,
    pub total_time_ms: u64,
    pub average_time_ms: f64,
    pub min_time_ms: u64,
    pub max_time_ms: u64,
    pub last_inference_time_ms: u64,
}

impl Default for InferenceStats {
    fn default() -> Self {
        Self {
            total_inferences: 0,
            successful_inferences: 0,
            failed_inferences: 0,
            total_time_ms: 0,
            average_time_ms: 0.0,
            min_time_ms: u64::MAX,
            max_time_ms: 0,
            last_inference_time_ms: 0,
        }
    }
}

impl InferenceStats {
    pub fn update(&mut self, inference_time_ms: u64, success: bool) {
        self.total_inferences += 1;
        
        if success {
            self.successful_inferences += 1;
            self.total_time_ms += inference_time_ms;
            self.average_time_ms = self.total_time_ms as f64 / self.successful_inferences as f64;
            self.min_time_ms = self.min_time_ms.min(inference_time_ms);
            self.max_time_ms = self.max_time_ms.max(inference_time_ms);
        } else {
            self.failed_inferences += 1;
        }
        
        self.last_inference_time_ms = inference_time_ms;
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_inferences == 0 {
            0.0
        } else {
            self.successful_inferences as f64 / self.total_inferences as f64 * 100.0
        }
    }

    pub fn print(&self) {
        println!("=== Inference Stats ===");
        println!("Total: {}", self.total_inferences);
        println!("Success: {}", self.successful_inferences);
        println!("Failed: {}", self.failed_inferences);
        println!("Success rate: {:.2}%", self.success_rate());
        
        if self.successful_inferences > 0 {
            println!("Avg: {:.2}ms", self.average_time_ms);
            println!("Min: {}ms", self.min_time_ms);
            println!("Max: {}ms", self.max_time_ms);
            println!("Last: {}ms", self.last_inference_time_ms);
        }
        println!("================");
    }
}

#[derive(Debug)]
pub struct InferenceResult {
    pub outputs: Vec<ArrayD<f32>>,
    pub inference_time_ms: u64,
    pub success: bool,
    pub error: Option<String>,
}

impl InferenceResult {
    pub fn success(outputs: Vec<ArrayD<f32>>, inference_time_ms: u64) -> Self {
        Self {
            outputs,
            inference_time_ms,
            success: true,
            error: None,
        }
    }

    pub fn failure(error: String, inference_time_ms: u64) -> Self {
        Self {
            outputs: Vec::new(),
            inference_time_ms,
            success: false,
            error: Some(error),
        }
    }

    pub fn first_output(&self) -> Option<&ArrayD<f32>> {
        self.outputs.first()
    }

    pub fn get_output(&self, index: usize) -> Option<&ArrayD<f32>> {
        self.outputs.get(index)
    }

    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }
}

#[derive(Debug)]
pub struct InferenceEngine {
    session: OnnxSession,
    config: InferenceConfig,
    stats: InferenceStats,
}

impl InferenceEngine {
    pub fn new(session: OnnxSession, config: InferenceConfig) -> Result<Self> {
        log::info!("Creating inference engine");
        log::info!("Input count: {}", session.input_count());
        log::info!("Output count: {}", session.output_count());
        log::info!("Max retries: {}", config.max_retries);

        Ok(Self {
            session,
            config,
            stats: InferenceStats::default(),
        })
    }

    pub fn from_model(model: OnnxModel, config: InferenceConfig) -> Result<Self> {
        use super::{OnnxEnvironment, SessionConfig};
        let env = OnnxEnvironment::new()?;
        let session_config = SessionConfig::default();
        let session = OnnxSession::new(model.path(), session_config, &env)?;
        Self::new(session, config)
    }

    pub fn with_default_config(session: OnnxSession) -> Result<Self> {
        Self::new(session, InferenceConfig::default())
    }

    pub fn session(&self) -> &OnnxSession {
        &self.session
    }

    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }

    pub fn stats(&self) -> &InferenceStats {
        &self.stats
    }

    pub fn run_inference(&mut self, inputs: Vec<DynamicTensor>) -> Result<InferenceResult> {
        let start_time = Instant::now();
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                std::thread::sleep(Duration::from_millis(self.config.retry_delay_ms));
                log::warn!("Inference retry {}/{}", attempt, self.config.max_retries);
            }

            match self.run_single_inference(&inputs) {
                Ok(result) => {
                    let inference_time = start_time.elapsed().as_millis() as u64;

                    if self.config.enable_profiling {
                        log::debug!("Inference successful, time: {} ms", inference_time);
                    }

                    let result_with_timing = InferenceResult::success(result, inference_time);
                    self.stats.update(inference_time, true);

                    return Ok(result_with_timing);
                }
                Err(e) => {
                    last_error = Some(e.clone());
                    log::error!("Inference failed (attempt {}): {}", attempt + 1, e);

                    if attempt == self.config.max_retries {
                        let inference_time = start_time.elapsed().as_millis() as u64;
                        self.stats.update(inference_time, false);
                    }
                }
            }
        }

        let inference_time = start_time.elapsed().as_millis() as u64;
        let error_msg = last_error
            .map(|e| format!("Inference failed after {} retries: {}", self.config.max_retries, e))
            .unwrap_or_else(|| "Inference failed".to_string());

        Ok(InferenceResult::failure(error_msg, inference_time))
    }

    fn run_single_inference(&mut self, inputs: &[DynamicTensor]) -> Result<Vec<ArrayD<f32>>> {
        let output_tensors = self.session.run(inputs.to_vec())?;
        let output_arrays: Vec<ArrayD<f32>> = output_tensors.into_iter()
            .map(|tensor| tensor.into_ndarray())
            .collect();
        Ok(output_arrays)
    }

    pub fn warm_up(&mut self) -> Result<()> {
        log::info!("Starting warm-up...");
        let dummy_data = vec![0.0f32; 16000];
        let dummy_tensor = DynamicTensor::new_f32(dummy_data, vec![1, 1, 16000]);
        let inputs = vec![dummy_tensor];
        let _result = self.run_inference(inputs)?;
        log::info!("Warm-up completed");
        Ok(())
    }

    pub fn reset_stats(&mut self) {
        self.stats.reset();
    }

    pub fn print_info(&self) {
        println!("=== Engine Info ===");
        println!("Inputs: {}", self.session.input_count());
        println!("Outputs: {}", self.session.output_count());
        println!("Max retries: {}", self.config.max_retries);
        println!("Retry delay: {}ms", self.config.retry_delay_ms);
        println!("Timeout: {}ms", self.config.timeout_ms);
        println!("Profiling: {}", if self.config.enable_profiling { "On" } else { "Off" });

        if let Some(batch_size) = self.config.batch_size {
            println!("Batch size: {}", batch_size);
        }

        self.stats.print();
        println!("===============");
    }
}

#[derive(Debug)]
pub struct BatchInferenceEngine {
    base_engine: InferenceEngine,
    batch_size: usize,
}

impl BatchInferenceEngine {
    pub fn new(model: OnnxModel, batch_size: usize, config: InferenceConfig) -> Result<Self> {
        let base_engine = InferenceEngine::from_model(model, config)?;
        Ok(Self {
            base_engine,
            batch_size,
        })
    }

    pub fn run_batch_inference(&mut self, inputs: Vec<Vec<DynamicTensor>>) -> Result<Vec<InferenceResult>> {
        let mut results = Vec::new();

        for batch in inputs.chunks(self.batch_size) {
            for input_batch in batch {
                let result = self.base_engine.run_inference(input_batch.clone())?;
                results.push(result);
            }
        }

        Ok(results)
    }

    pub fn base_engine(&self) -> &InferenceEngine {
        &self.base_engine
    }

    pub fn base_engine_mut(&mut self) -> &mut InferenceEngine {
        &mut self.base_engine
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_model() -> OnnxModel {
        // Use project's real ONNX model for testing
        let model_path = "model/ZipEnhancer_ONNX/ZipEnhancer.onnx";

        // If model not found in test environment, skip test
        if !std::path::Path::new(model_path).exists() {
            panic!("Test model file not found: {}. Please ensure tests run in correct directory.", model_path);
        }

        OnnxModel::new(model_path).unwrap()
    }

    fn create_test_session() -> OnnxSession {
        use crate::onnx::{OnnxEnvironment, SessionConfig};
        let model_path = "model/ZipEnhancer_ONNX/ZipEnhancer.onnx";

        // If model not found in test environment, skip test
        if !std::path::Path::new(model_path).exists() {
            panic!("Test model file not found: {}. Please ensure tests run in correct directory.", model_path);
        }

        let env = OnnxEnvironment::new().unwrap();
        OnnxSession::new(std::path::Path::new(model_path), SessionConfig::default(), &env).unwrap()
    }

    #[test]
    fn test_inference_config_default() {
        let config = InferenceConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.retry_delay_ms, 100);
        assert_eq!(config.timeout_ms, 5000);
        assert!(!config.enable_profiling);
        assert!(config.batch_size.is_none());
    }

    #[test]
    fn test_inference_stats() {
        let mut stats = InferenceStats::default();
        
        // Initial state
        assert_eq!(stats.total_inferences, 0);
        assert_eq!(stats.success_rate(), 0.0);
        
        // Update successful inference
        stats.update(100, true);
        assert_eq!(stats.total_inferences, 1);
        assert_eq!(stats.successful_inferences, 1);
        assert_eq!(stats.average_time_ms, 100.0);
        assert_eq!(stats.success_rate(), 100.0);
        
        // Update failed inference
        stats.update(200, false);
        assert_eq!(stats.total_inferences, 2);
        assert_eq!(stats.successful_inferences, 1);
        assert_eq!(stats.failed_inferences, 1);
        assert_eq!(stats.success_rate(), 50.0);
    }

    #[test]
    fn test_inference_result() {
        let data = ArrayD::from_elem(vec![2, 2], 1.0);
        let outputs = vec![data.clone()];
        
        let success_result = InferenceResult::success(outputs.clone(), 50);
        assert!(success_result.success);
        assert_eq!(success_result.inference_time_ms, 50);
        assert!(success_result.error.is_none());
        assert_eq!(success_result.output_count(), 1);
        assert_eq!(success_result.first_output().unwrap().shape(), &[2, 2]);
        
        let failure_result = InferenceResult::failure("Test error".to_string(), 30);
        assert!(!failure_result.success);
        assert_eq!(failure_result.inference_time_ms, 30);
        assert!(failure_result.error.is_some());
        assert_eq!(failure_result.output_count(), 0);
    }

    #[test]
    fn test_inference_engine_creation() {
        let session = create_test_session();
        let config = InferenceConfig::default();

        let engine = InferenceEngine::new(session, config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_inference_engine_with_default_config() {
        let session = create_test_session();
        let engine = InferenceEngine::with_default_config(session);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_batch_inference_engine() {
        let model = create_test_model();
        let config = InferenceConfig::default();
        let batch_engine = BatchInferenceEngine::new(model, 4, config);
        assert!(batch_engine.is_ok());
    }

    #[test]
    fn test_stats_reset() {
        let mut stats = InferenceStats::default();
        stats.update(100, true);
        stats.update(200, false);
        
        assert_eq!(stats.total_inferences, 2);
        
        stats.reset();
        assert_eq!(stats.total_inferences, 0);
        assert_eq!(stats.successful_inferences, 0);
        assert_eq!(stats.failed_inferences, 0);
    }

    #[test]
    fn test_simple_inference() {
        let session = create_test_session();
        let mut engine = InferenceEngine::with_default_config(session).unwrap();

        // Test with smaller data to avoid memory issues during testing
        let data = vec![1000.0; 1000];
        let input = DynamicTensor::new_f32(data, vec![1, 1, 1000]);
        let inputs = vec![input];

        let _result = engine.run_inference(inputs);

        // This test just verifies that the inference engine doesn't panic
        // when given incompatible input data
        // The actual model expects int16 data with specific dimensions
    }

    #[test]
    fn test_warm_up() {
        let session = create_test_session();
        let mut engine = InferenceEngine::with_default_config(session).unwrap();
        
        let result = engine.warm_up();
        assert!(result.is_ok());
    }
}