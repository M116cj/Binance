import asyncio
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor
import redis
import orjson
from dataclasses import dataclass, asdict
import onnxruntime as ort
import joblib
from datetime import datetime, timedelta
import os
from collections import defaultdict, deque
import hashlib

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from backend.storage.redis_client import RedisManager
from backend.models.cost_model import CostModel
from backend.models.features import FeatureEngine
from backend.utils.monitoring import MetricsCollector
from backend.utils.time_utils import TimeManager
from config.settings import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictionRequest:
    """Prediction request structure"""
    symbol: str
    theta_up: float = 0.006
    theta_dn: float = 0.004
    horizons: List[int] = None
    
    def __post_init__(self):
        if self.horizons is None:
            self.horizons = [5, 10, 30]  # minutes

@dataclass
class PredictionResponse:
    """Prediction response structure"""
    id: str
    symbol: str
    exchange_time: int
    ingest_time: int
    infer_time: int
    
    # Predictions for each horizon
    predictions: Dict[int, Dict[str, float]]  # horizon -> {p_up, p_ci_low, p_ci_high, etc.}
    
    # Utility and decision
    expected_returns: Dict[int, float]
    estimated_costs: Dict[int, float]
    utilities: Dict[int, float]
    decisions: Dict[int, str]  # 'A', 'B', or 'none'
    
    # Metadata
    regime: str
    capacity_pct: float
    features_top5: Dict[str, float]
    model_version: str
    feature_version: str
    cost_model: str
    data_window_id: str
    quality_flags: List[str]
    cooldown_until: Optional[int]
    sla_latency_ms: float

class BatchProcessor:
    """Batch processor for inference requests"""
    
    def __init__(self, max_batch_size: int = 32, max_wait_ms: int = 25):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.pending_requests = deque()
        self.batch_ready = threading.Event()
        self.lock = threading.Lock()
    
    def add_request(self, request_data: Tuple[str, PredictionRequest, asyncio.Future]):
        """Add request to batch"""
        with self.lock:
            self.pending_requests.append(request_data)
            
            if len(self.pending_requests) >= self.max_batch_size:
                self.batch_ready.set()
    
    def get_batch(self) -> List[Tuple[str, PredictionRequest, asyncio.Future]]:
        """Get batch of requests"""
        batch = []
        
        with self.lock:
            # Get up to max_batch_size requests
            while self.pending_requests and len(batch) < self.max_batch_size:
                batch.append(self.pending_requests.popleft())
            
            self.batch_ready.clear()
        
        return batch

class ModelManager:
    """ONNX model manager with optimization"""
    
    def __init__(self):
        self.models: Dict[str, ort.InferenceSession] = {}
        self.calibrators: Dict[str, Any] = {}
        self.feature_names: Dict[str, List[str]] = {}
        self.model_versions: Dict[str, str] = {}
        
        # ONNX Runtime optimization
        self.providers = ['CPUExecutionProvider']
        self.session_options = ort.SessionOptions()
        self.session_options.intra_op_num_threads = os.cpu_count()
        self.session_options.inter_op_num_threads = 1
        self.session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    def load_model(self, model_name: str, model_path: str, calibrator_path: Optional[str] = None):
        """Load ONNX model and calibrator"""
        try:
            logger.info(f"Loading model {model_name} from {model_path}")
            
            # Load ONNX model
            session = ort.InferenceSession(
                model_path, 
                sess_options=self.session_options,
                providers=self.providers
            )
            
            self.models[model_name] = session
            
            # Get input feature names
            input_names = [inp.name for inp in session.get_inputs()]
            self.feature_names[model_name] = input_names
            
            # Load calibrator if provided
            if calibrator_path and os.path.exists(calibrator_path):
                calibrator = joblib.load(calibrator_path)
                self.calibrators[model_name] = calibrator
                logger.info(f"Loaded calibrator for {model_name}")
            
            # Set model version
            self.model_versions[model_name] = "1.0.0"  # Could be extracted from model metadata
            
            logger.info(f"Successfully loaded model {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def predict_batch(self, model_name: str, features_batch: np.ndarray) -> np.ndarray:
        """Batch prediction"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not loaded")
            
            session = self.models[model_name]
            input_name = session.get_inputs()[0].name
            
            # Run inference
            raw_predictions = session.run(None, {input_name: features_batch.astype(np.float32)})[0]
            
            # Apply calibration if available
            if model_name in self.calibrators:
                calibrator = self.calibrators[model_name]
                calibrated_predictions = calibrator.predict_proba(raw_predictions.reshape(-1, 1))[:, 1]
                return calibrated_predictions
            
            return raw_predictions.flatten()
            
        except Exception as e:
            logger.error(f"Error in batch prediction for {model_name}: {e}")
            raise

class CooldownManager:
    """Manages cooldown periods for signals"""
    
    def __init__(self):
        self.cooldowns: Dict[str, int] = {}  # key -> timestamp
        self.lock = threading.Lock()
    
    def set_cooldown(self, symbol: str, horizon: int, cooldown_minutes: int = 15):
        """Set cooldown for symbol+horizon combination"""
        key = f"{symbol}_{horizon}"
        cooldown_until = int((datetime.now() + timedelta(minutes=cooldown_minutes)).timestamp() * 1000)
        
        with self.lock:
            self.cooldowns[key] = cooldown_until
    
    def is_cooled_down(self, symbol: str, horizon: int) -> bool:
        """Check if symbol+horizon is cooled down"""
        key = f"{symbol}_{horizon}"
        current_time = int(time.time() * 1000)
        
        with self.lock:
            cooldown_until = self.cooldowns.get(key, 0)
            return current_time >= cooldown_until
    
    def get_cooldown_until(self, symbol: str, horizon: int) -> Optional[int]:
        """Get cooldown end time"""
        key = f"{symbol}_{horizon}"
        with self.lock:
            return self.cooldowns.get(key)

class InferenceService:
    """High-performance inference service"""
    
    def __init__(self):
        self.settings = Settings()
        self.redis_manager = RedisManager()
        self.cost_model = CostModel()
        self.feature_engine = FeatureEngine()
        self.metrics_collector = MetricsCollector("inference")
        self.time_manager = TimeManager()
        
        # Model management
        self.model_manager = ModelManager()
        
        # Batch processing
        self.batch_processor = BatchProcessor()
        self.processing_thread = None
        self.shutdown_flag = False
        
        # Cooldown management
        self.cooldown_manager = CooldownManager()
        
        # Bloom filter for deduplication (simplified version)
        self.recent_requests: deque = deque(maxlen=10000)
        
        # Thresholds
        self.tier_thresholds = {
            'A': {'tau': 0.75, 'kappa': 1.20},
            'B': {'tau': 0.65, 'kappa': 1.00}
        }
    
    async def initialize(self):
        """Initialize inference service"""
        logger.info("Initializing inference service...")
        
        await self.redis_manager.initialize()
        await self.cost_model.initialize()
        
        # Load models
        await self._load_models()
        
        # Start batch processing thread
        self.processing_thread = threading.Thread(target=self._batch_processing_loop)
        self.processing_thread.start()
        
        logger.info("Inference service initialized")
    
    async def _load_models(self):
        """Load ML models"""
        try:
            # Load default model (in production, these paths would be configurable)
            model_path = self.settings.MODEL_PATH or "./models/lightgbm_model.onnx"
            calibrator_path = self.settings.CALIBRATOR_PATH or "./models/isotonic_calibrator.pkl"
            
            if os.path.exists(model_path):
                self.model_manager.load_model("default", model_path, calibrator_path)
                logger.info("Loaded default model")
            else:
                logger.warning(f"Model file not found at {model_path}, using mock predictions")
                # In production, this would be an error
        
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _batch_processing_loop(self):
        """Main batch processing loop"""
        logger.info("Starting batch processing loop")
        
        while not self.shutdown_flag:
            try:
                # Wait for batch to be ready or timeout
                batch_ready = self.batch_processor.batch_ready.wait(timeout=self.batch_processor.max_wait_ms / 1000)
                
                if batch_ready or len(self.batch_processor.pending_requests) > 0:
                    batch = self.batch_processor.get_batch()
                    
                    if batch:
                        self._process_batch(batch)
                
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
                time.sleep(0.1)
        
        logger.info("Batch processing loop stopped")
    
    def _process_batch(self, batch: List[Tuple[str, PredictionRequest, asyncio.Future]]):
        """Process a batch of prediction requests"""
        try:
            batch_start = time.time()
            
            # Extract data from batch
            request_ids = [item[0] for item in batch]
            requests = [item[1] for item in batch]
            futures = [item[2] for item in batch]
            
            # Get features for all symbols
            features_batch = []
            feature_metadata = []
            
            for request in requests:
                features_data = self._get_features_for_prediction(request.symbol)
                if features_data:
                    features_batch.append(features_data['features'])
                    feature_metadata.append(features_data['metadata'])
                else:
                    # Use zeros if features not available
                    features_batch.append(np.zeros(50))  # Assuming 50 features
                    feature_metadata.append({'quality_flags': ['missing_features']})
            
            if not features_batch:
                # Set all futures as failed
                for future in futures:
                    future.set_exception(HTTPException(status_code=503, detail="No features available"))
                return
            
            # Convert to numpy array
            features_array = np.array(features_batch)
            
            # Batch prediction
            if "default" in self.model_manager.models:
                predictions = self.model_manager.predict_batch("default", features_array)
            else:
                # Mock predictions for development
                predictions = np.random.beta(2, 5, len(features_batch))  # Realistic probability distribution
            
            # Process results
            for i, (request_id, request, future) in enumerate(batch):
                try:
                    response = self._create_prediction_response(
                        request_id, request, predictions[i], feature_metadata[i]
                    )
                    future.set_result(response)
                    
                except Exception as e:
                    logger.error(f"Error creating response for request {request_id}: {e}")
                    future.set_exception(e)
            
            # Update metrics
            batch_latency = (time.time() - batch_start) * 1000
            self.metrics_collector.observe_histogram("batch_processing_latency_ms", batch_latency)
            self.metrics_collector.observe_histogram("batch_size", len(batch))
            self.metrics_collector.increment_counter("predictions_made", value=len(batch))
        
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Set all futures as failed
            for _, _, future in batch:
                future.set_exception(e)
    
    def _get_features_for_prediction(self, symbol: str) -> Optional[Dict]:
        """Get features for prediction"""
        try:
            # Get latest features from Redis
            key = f"features:{symbol}"
            data_json = self.redis_manager.client.get(key)
            
            if not data_json:
                return None
            
            data = orjson.loads(data_json)
            features_dict = data.get('features', {})
            quality_flags = data.get('quality_flags', [])
            
            # Convert to feature vector (ensure correct order)
            expected_features = [
                'qi_1', 'microprice_dev', 'ofi_10', 'rv_1m', 'rv_ratio',
                'impact_lambda', 'depth_slope_bid', 'depth_slope_ask',
                'near_touch_ratio', 'bb_position', 'bb_squeeze'
            ]
            
            feature_vector = []
            for feature_name in expected_features:
                value = features_dict.get(feature_name, 0.0)
                # Handle NaN values
                if np.isnan(value) or np.isinf(value):
                    value = 0.0
                feature_vector.append(value)
            
            # Pad with zeros if needed
            while len(feature_vector) < 50:
                feature_vector.append(0.0)
            
            return {
                'features': np.array(feature_vector[:50]),  # Limit to expected size
                'metadata': {
                    'quality_flags': quality_flags,
                    'feature_count': len([f for f in features_dict.values() if not np.isnan(f)]),
                    'timestamp': data.get('timestamp', int(time.time() * 1000))
                }
            }
        
        except Exception as e:
            logger.error(f"Error getting features for {symbol}: {e}")
            return None
    
    def _create_prediction_response(self, request_id: str, request: PredictionRequest, 
                                  base_prediction: float, metadata: Dict) -> PredictionResponse:
        """Create prediction response"""
        current_time = int(time.time() * 1000)
        
        # Generate predictions for each horizon
        predictions = {}
        expected_returns = {}
        estimated_costs = {}
        utilities = {}
        decisions = {}
        
        for horizon in request.horizons:
            # Adjust prediction based on horizon (longer horizons typically have higher probability)
            p_up = min(base_prediction * (1 + horizon * 0.01), 0.95)  # Slight increase with horizon
            
            # Confidence intervals (simplified)
            p_ci_low = max(p_up - 0.1, 0.0)
            p_ci_high = min(p_up + 0.1, 1.0)
            
            # Expected return (simplified model)
            exp_return = p_up * request.theta_up * np.random.uniform(1.2, 2.0)  # Expected excess return
            
            # Estimated cost
            est_cost = self.cost_model.estimate_cost(request.symbol, horizon)
            
            # Utility calculation: U(t) = E[excess_return] / C(s,t)
            utility = exp_return / max(est_cost, 0.001) if est_cost > 0 else 0.0
            
            # Decision based on thresholds
            decision = 'none'
            if p_up >= self.tier_thresholds['A']['tau'] and utility >= self.tier_thresholds['A']['kappa']:
                decision = 'A'
            elif p_up >= self.tier_thresholds['B']['tau'] and utility >= self.tier_thresholds['B']['kappa']:
                decision = 'B'
            
            predictions[horizon] = {
                'p_up': p_up,
                'p_ci_low': p_ci_low,
                'p_ci_high': p_ci_high
            }
            expected_returns[horizon] = exp_return
            estimated_costs[horizon] = est_cost
            utilities[horizon] = utility
            decisions[horizon] = decision
        
        # Cooldown management
        cooldown_until = None
        for horizon in request.horizons:
            if decisions[horizon] in ['A', 'B']:
                if not self.cooldown_manager.is_cooled_down(request.symbol, horizon):
                    decisions[horizon] = 'none'  # Override decision if in cooldown
                else:
                    # Set new cooldown
                    self.cooldown_manager.set_cooldown(request.symbol, horizon, 15)  # 15 minutes
                    cooldown_until = self.cooldown_manager.get_cooldown_until(request.symbol, horizon)
        
        # Top 5 features (mock for now)
        features_top5 = {
            'qi_1': 0.15,
            'ofi_10': 0.23,
            'rv_ratio': 1.34,
            'microprice_dev': -0.02,
            'impact_lambda': 0.08
        }
        
        # Create response
        response = PredictionResponse(
            id=request_id,
            symbol=request.symbol,
            exchange_time=current_time - 100,  # Mock exchange time
            ingest_time=current_time - 50,     # Mock ingest time
            infer_time=current_time,
            predictions=predictions,
            expected_returns=expected_returns,
            estimated_costs=estimated_costs,
            utilities=utilities,
            decisions=decisions,
            regime="medium_vol_medium_depth",
            capacity_pct=0.75,
            features_top5=features_top5,
            model_version=self.model_manager.model_versions.get("default", "1.0.0"),
            feature_version="1.0.0",
            cost_model="v1.2.0",
            data_window_id=f"dw_{current_time}",
            quality_flags=metadata.get('quality_flags', []),
            cooldown_until=cooldown_until,
            sla_latency_ms=50.0  # Will be calculated properly in actual response
        )
        
        return response
    
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Async prediction with batching"""
        # Generate request ID
        request_id = hashlib.md5(f"{request.symbol}_{time.time()}_{np.random.random()}".encode()).hexdigest()[:16]
        
        # Check for recent duplicate requests (simplified bloom filter)
        request_hash = f"{request.symbol}_{request.theta_up}_{request.theta_dn}"
        if request_hash in self.recent_requests:
            logger.warning(f"Duplicate request detected for {request.symbol}")
        else:
            self.recent_requests.append(request_hash)
        
        # Create future for async response
        future = asyncio.Future()
        
        # Add to batch processor
        self.batch_processor.add_request((request_id, request, future))
        
        # Wait for result
        try:
            result = await asyncio.wait_for(future, timeout=2.0)  # 2 second timeout
            
            # Calculate SLA latency
            result.sla_latency_ms = (time.time() * 1000) - result.infer_time
            
            return result
            
        except asyncio.TimeoutError:
            self.metrics_collector.increment_counter("prediction_timeouts")
            raise HTTPException(status_code=504, detail="Prediction timeout")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        try:
            # Check Redis connectivity
            redis_health = self.redis_manager.client.ping()
            
            # Check model availability
            models_loaded = len(self.model_manager.models)
            
            # Check recent performance
            recent_latency = self.metrics_collector.get_histogram_percentile("batch_processing_latency_ms", 0.95)
            
            # Calculate exchange lag (mock for now)
            exchange_lag_s = np.random.uniform(0.1, 0.5)
            
            health_status = "healthy"
            if exchange_lag_s > 2.0:
                health_status = "degraded"
            elif not redis_health or models_loaded == 0:
                health_status = "unhealthy"
            
            return {
                "status": health_status,
                "timestamp": int(time.time() * 1000),
                "redis_connected": redis_health,
                "models_loaded": models_loaded,
                "exchange_lag_s": exchange_lag_s,
                "recent_latency_p95_ms": recent_latency,
                "batch_queue_size": len(self.batch_processor.pending_requests)
            }
        
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": int(time.time() * 1000)
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up inference service...")
        
        self.shutdown_flag = True
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        logger.info("Inference service cleanup complete")

# FastAPI application
app = FastAPI(title="Crypto Surge Prediction Inference API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference service instance
inference_service = None

@app.on_event("startup")
async def startup_event():
    global inference_service
    inference_service = InferenceService()
    await inference_service.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    global inference_service
    if inference_service:
        await inference_service.cleanup()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return await inference_service.health_check()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Main prediction endpoint"""
    try:
        start_time = time.time()
        
        result = await inference_service.predict(request)
        
        # Update metrics
        latency_ms = (time.time() - start_time) * 1000
        inference_service.metrics_collector.observe_histogram("predict_endpoint_latency_ms", latency_ms)
        inference_service.metrics_collector.increment_counter("predict_requests", {"symbol": request.symbol})
        
        return result
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        inference_service.metrics_collector.increment_counter("predict_errors", {"symbol": request.symbol})
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return {"metrics": "prometheus_metrics_placeholder"}

if __name__ == "__main__":
    uvicorn.run(
        "backend.inference_service:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for shared state
        loop="uvloop"
    )
