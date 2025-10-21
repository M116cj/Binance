import asyncio
import websockets
import json
import time
import logging
from typing import Dict, List, Any, Optional
import redis
import orjson
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass, asdict
import os
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque
import zlib
from binance import AsyncClient, BinanceSocketManager
import uvloop
import signal
import sys

from backend.storage.redis_client import RedisManager
from backend.storage.clickhouse_client import ClickHouseManager
from backend.utils.monitoring import MetricsCollector
from backend.utils.time_utils import TimeManager
from backend.utils.websocket_utils import WebSocketManager
from config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data structure with three timestamps"""
    symbol: str
    stream_type: str
    data: Dict[str, Any]
    exchange_time: int
    ingest_time: int
    sequence_id: Optional[int] = None
    quality_flags: List[str] = None
    
    def __post_init__(self):
        if self.quality_flags is None:
            self.quality_flags = []

class BinanceIngestionService:
    """High-performance Binance WebSocket ingestion service"""
    
    def __init__(self):
        self.settings = Settings()
        self.redis_manager = RedisManager()
        self.clickhouse_manager = ClickHouseManager()
        self.metrics_collector = MetricsCollector("ingestion")
        self.time_manager = TimeManager()
        self.ws_manager = WebSocketManager()
        
        # Ingestion configuration
        self.symbols_per_connection = 25
        self.micro_batch_ms = 20  # Auto-tuned 10-25ms
        self.snapshot_interval_s = 15
        self.heartbeat_interval_s = 5
        self.max_reconnect_delay_s = 8
        
        # Connection management
        self.connections: Dict[str, Any] = {}
        self.message_queues: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.last_update_ids: Dict[str, int] = {}
        self.sequence_numbers: Dict[str, int] = defaultdict(int)
        
        # Micro-batching
        self.batch_buffers: Dict[str, List[MarketData]] = defaultdict(list)
        self.last_batch_time: Dict[str, float] = defaultdict(time.time)
        
        # Quality monitoring
        self.packet_counts: Dict[str, int] = defaultdict(int)
        self.packet_losses: Dict[str, int] = defaultdict(int)
        self.gap_counts: Dict[str, int] = defaultdict(int)
        
        # Shutdown flag
        self.shutdown_flag = False
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.processing_threads: List[threading.Thread] = []
        
    async def initialize(self):
        """Initialize the ingestion service"""
        logger.info("Initializing Binance ingestion service...")
        
        await self.redis_manager.initialize()
        await self.clickhouse_manager.initialize()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Ingestion service initialized successfully")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_flag = True
    
    async def start_ingestion(self, symbols: List[str]):
        """Start ingestion for specified symbols"""
        logger.info(f"Starting ingestion for {len(symbols)} symbols...")
        
        # Split symbols into connection groups
        symbol_groups = [symbols[i:i + self.symbols_per_connection] 
                        for i in range(0, len(symbols), self.symbols_per_connection)]
        
        # Start WebSocket connections
        tasks = []
        for i, symbol_group in enumerate(symbol_groups):
            task = asyncio.create_task(self._start_connection_group(f"conn_{i}", symbol_group))
            tasks.append(task)
        
        # Start processing threads
        for i in range(4):  # 4 processing threads
            thread = threading.Thread(target=self._process_messages_thread, args=(f"processor_{i}",))
            thread.start()
            self.processing_threads.append(thread)
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitoring_thread)
        monitor_thread.start()
        self.processing_threads.append(monitor_thread)
        
        # Wait for all connections
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in ingestion: {e}")
        finally:
            await self.cleanup()
    
    async def _start_connection_group(self, conn_id: str, symbols: List[str]):
        """Start WebSocket connection for a group of symbols"""
        retry_count = 0
        max_retries = 10
        
        while not self.shutdown_flag and retry_count < max_retries:
            try:
                logger.info(f"Starting connection {conn_id} for symbols: {symbols}")
                
                # Create Binance client
                client = AsyncClient()
                bsm = BinanceSocketManager(client)
                
                # Setup streams
                streams = []
                for symbol in symbols:
                    streams.extend([
                        f"{symbol.lower()}@aggTrade",
                        f"{symbol.lower()}@bookTicker",
                        f"{symbol.lower()}@depth@100ms",
                        f"{symbol.lower()}@kline_1m",
                        f"{symbol.lower()}@ticker"
                    ])
                
                # Start multiplex socket
                socket = bsm.multiplex_socket(streams)
                
                await self._handle_socket_connection(conn_id, socket, symbols)
                
            except Exception as e:
                retry_count += 1
                delay = min(0.5 * (2 ** retry_count), self.max_reconnect_delay_s)
                logger.error(f"Connection {conn_id} failed (attempt {retry_count}): {e}")
                logger.info(f"Reconnecting in {delay:.1f} seconds...")
                await asyncio.sleep(delay)
        
        logger.warning(f"Connection {conn_id} exceeded max retries")
    
    async def _handle_socket_connection(self, conn_id: str, socket, symbols: List[str]):
        """Handle WebSocket connection messages"""
        async with socket as ws:
            logger.info(f"Connection {conn_id} established")
            self.connections[conn_id] = ws
            
            # Send initial snapshots
            for symbol in symbols:
                await self._request_snapshot(symbol)
            
            last_heartbeat = time.time()
            
            async for msg in ws:
                try:
                    if self.shutdown_flag:
                        break
                    
                    current_time = time.time()
                    
                    # Heartbeat check
                    if current_time - last_heartbeat > self.heartbeat_interval_s:
                        # Send heartbeat (handled by Binance automatically)
                        last_heartbeat = current_time
                    
                    # Process message
                    await self._process_ws_message(conn_id, msg)
                    
                    # Update metrics
                    self.metrics_collector.increment_counter("ws_messages_received", {"connection": conn_id})
                    
                except Exception as e:
                    logger.error(f"Error processing message on {conn_id}: {e}")
                    self.metrics_collector.increment_counter("ws_message_errors", {"connection": conn_id})
    
    async def _process_ws_message(self, conn_id: str, msg: dict):
        """Process individual WebSocket message"""
        try:
            if 'stream' not in msg or 'data' not in msg:
                return
            
            stream = msg['stream']
            data = msg['data']
            
            # Parse symbol and stream type
            if '@' not in stream:
                return
            
            symbol_part, stream_type = stream.split('@', 1)
            symbol = symbol_part.upper()
            
            # Create market data object
            exchange_time = int(data.get('E', 0)) if 'E' in data else int(time.time() * 1000)
            ingest_time = int(time.time() * 1000)
            
            market_data = MarketData(
                symbol=symbol,
                stream_type=stream_type,
                data=data,
                exchange_time=exchange_time,
                ingest_time=ingest_time
            )
            
            # Quality checks
            await self._perform_quality_checks(market_data)
            
            # Add to micro-batch buffer
            await self._add_to_batch(market_data)
            
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
            self.metrics_collector.increment_counter("message_processing_errors")
    
    async def _perform_quality_checks(self, market_data: MarketData):
        """Perform quality checks on market data"""
        # Time sync check
        time_diff = abs(market_data.ingest_time - market_data.exchange_time)
        if time_diff > 100:  # 100ms threshold
            market_data.quality_flags.append("degraded_time_sync")
            self.metrics_collector.increment_counter("time_sync_degraded", {"symbol": market_data.symbol})
        
        # Sequence check for depth updates
        if market_data.stream_type.startswith('depth'):
            data = market_data.data
            if 'u' in data and 'U' in data:
                last_update_id = self.last_update_ids.get(market_data.symbol, 0)
                if last_update_id > 0 and data['U'] > last_update_id + 1:
                    market_data.quality_flags.append("depth_gap")
                    self.gap_counts[market_data.symbol] += 1
                    self.metrics_collector.increment_counter("depth_gaps", {"symbol": market_data.symbol})
                
                self.last_update_ids[market_data.symbol] = data['u']
        
        # Assign sequence number
        self.sequence_numbers[market_data.symbol] += 1
        market_data.sequence_id = self.sequence_numbers[market_data.symbol]
    
    async def _add_to_batch(self, market_data: MarketData):
        """Add market data to micro-batch buffer"""
        batch_key = f"{market_data.symbol}_{market_data.stream_type}"
        self.batch_buffers[batch_key].append(market_data)
        
        current_time = time.time()
        last_batch = self.last_batch_time[batch_key]
        
        # Check if batch should be processed
        batch_size = len(self.batch_buffers[batch_key])
        time_elapsed = (current_time - last_batch) * 1000  # ms
        
        if time_elapsed >= self.micro_batch_ms or batch_size >= 50:
            await self._process_batch(batch_key)
    
    async def _process_batch(self, batch_key: str):
        """Process a micro-batch of market data"""
        if batch_key not in self.batch_buffers or not self.batch_buffers[batch_key]:
            return
        
        batch = self.batch_buffers[batch_key].copy()
        self.batch_buffers[batch_key].clear()
        self.last_batch_time[batch_key] = time.time()
        
        # Add to processing queue
        for msg_queue in self.message_queues.values():
            if len(msg_queue) < msg_queue.maxlen - len(batch):
                msg_queue.extend(batch)
                break
        
        # Update metrics
        self.metrics_collector.observe_histogram("batch_size", len(batch))
        self.metrics_collector.observe_histogram("batch_latency_ms", 
                                                (time.time() - batch[0].ingest_time / 1000) * 1000)
    
    def _process_messages_thread(self, thread_id: str):
        """Thread to process message queues"""
        logger.info(f"Starting processing thread {thread_id}")
        
        queue = self.message_queues[thread_id]
        
        while not self.shutdown_flag:
            try:
                if queue:
                    batch = []
                    # Collect up to 100 messages
                    for _ in range(min(100, len(queue))):
                        if queue:
                            batch.append(queue.popleft())
                    
                    if batch:
                        self._process_batch_sync(batch)
                else:
                    time.sleep(0.001)  # 1ms sleep when no messages
                    
            except Exception as e:
                logger.error(f"Error in processing thread {thread_id}: {e}")
                time.sleep(0.01)
        
        logger.info(f"Processing thread {thread_id} stopped")
    
    def _process_batch_sync(self, batch: List[MarketData]):
        """Synchronously process a batch of market data"""
        try:
            # Store in Redis (hot cache)
            self._store_redis_batch(batch)
            
            # Store in ClickHouse (persistent storage)
            self._store_clickhouse_batch(batch)
            
            # Update metrics
            self.metrics_collector.increment_counter("messages_processed", value=len(batch))
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            self.metrics_collector.increment_counter("batch_processing_errors")
    
    def _store_redis_batch(self, batch: List[MarketData]):
        """Store batch in Redis with pipelining"""
        try:
            pipe = self.redis_manager.client.pipeline()
            
            for data in batch:
                # Key format: {spot|perp}:{symbol}:{stream_type}
                key = f"spot:{data.symbol}:{data.stream_type}"
                
                # Serialize data
                serialized = orjson.dumps({
                    'data': data.data,
                    'exchange_time': data.exchange_time,
                    'ingest_time': data.ingest_time,
                    'sequence_id': data.sequence_id,
                    'quality_flags': data.quality_flags
                })
                
                # Store with TTL
                ttl = 10 if data.symbol in ['BTCUSDT', 'ETHUSDT'] else 5
                pipe.setex(key, ttl, serialized)
                
                # Also store in time series for features
                ts_key = f"ts:{data.symbol}:{data.stream_type}"
                pipe.zadd(ts_key, {serialized: data.exchange_time})
                pipe.expire(ts_key, 300)  # 5 minute expiry for time series
            
            pipe.execute()
            
        except Exception as e:
            logger.error(f"Error storing Redis batch: {e}")
    
    def _store_clickhouse_batch(self, batch: List[MarketData]):
        """Store batch in ClickHouse"""
        try:
            # Group by stream type for efficient insertion
            stream_batches = defaultdict(list)
            for data in batch:
                stream_batches[data.stream_type].append(data)
            
            for stream_type, stream_data in stream_batches.items():
                self.clickhouse_manager.insert_market_data_batch(stream_type, stream_data)
                
        except Exception as e:
            logger.error(f"Error storing ClickHouse batch: {e}")
    
    async def _request_snapshot(self, symbol: str):
        """Request order book snapshot for depth stream initialization"""
        try:
            # This would typically be done via REST API
            logger.info(f"Requesting snapshot for {symbol}")
            
            # Mark snapshot time for sequence validation
            self.last_update_ids[symbol] = 0
            
        except Exception as e:
            logger.error(f"Error requesting snapshot for {symbol}: {e}")
    
    def _monitoring_thread(self):
        """Thread for monitoring and metrics collection"""
        logger.info("Starting monitoring thread")
        
        while not self.shutdown_flag:
            try:
                # Collect metrics
                self._collect_performance_metrics()
                
                # Check SLAs and trigger alerts
                self._check_sla_violations()
                
                # Auto-tune micro-batch size based on latency
                self._auto_tune_batch_size()
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring thread: {e}")
        
        logger.info("Monitoring thread stopped")
    
    def _collect_performance_metrics(self):
        """Collect performance metrics"""
        # Calculate packet loss rates
        for symbol in self.packet_counts:
            total_packets = self.packet_counts[symbol]
            lost_packets = self.packet_losses[symbol]
            loss_rate = lost_packets / max(total_packets, 1)
            
            self.metrics_collector.observe_gauge("packet_loss_rate", loss_rate, {"symbol": symbol})
            
            # Reset counters periodically
            if total_packets > 10000:
                self.packet_counts[symbol] = 0
                self.packet_losses[symbol] = 0
        
        # Calculate gap rates
        for symbol in self.gap_counts:
            gap_rate = self.gap_counts[symbol] / 60  # per minute
            self.metrics_collector.observe_gauge("depth_gap_rate", gap_rate, {"symbol": symbol})
            
            # Reset gap counter
            if time.time() % 60 < 5:  # Reset every minute
                self.gap_counts[symbol] = 0
    
    def _check_sla_violations(self):
        """Check for SLA violations and trigger alerts"""
        # Check connection health
        for conn_id, conn in self.connections.items():
            if conn is None:
                logger.warning(f"Connection {conn_id} is down")
                self.metrics_collector.increment_counter("connection_down", {"connection": conn_id})
    
    def _auto_tune_batch_size(self):
        """Auto-tune micro-batch size based on p95 latency"""
        try:
            # Get current p95 latency
            current_latency = self.metrics_collector.get_histogram_percentile("batch_latency_ms", 0.95)
            
            if current_latency is not None:
                if current_latency > 25:  # Target max 25ms
                    self.micro_batch_ms = max(10, self.micro_batch_ms - 1)
                elif current_latency < 15:  # Could increase for better throughput
                    self.micro_batch_ms = min(25, self.micro_batch_ms + 1)
                
                logger.debug(f"Auto-tuned batch size to {self.micro_batch_ms}ms (p95 latency: {current_latency:.1f}ms)")
        
        except Exception as e:
            logger.error(f"Error auto-tuning batch size: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up ingestion service...")
        
        # Close WebSocket connections
        for conn_id, conn in self.connections.items():
            if conn:
                try:
                    await conn.close()
                except:
                    pass
        
        # Wait for processing threads to finish
        for thread in self.processing_threads:
            thread.join(timeout=5)
        
        # Close executor
        self.executor.shutdown(wait=True)
        
        logger.info("Ingestion service cleanup complete")

async def main():
    """Main entry point for ingestion service"""
    # Set event loop policy for better performance
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    
    ingestion_service = BinanceIngestionService()
    
    try:
        await ingestion_service.initialize()
        
        # Define symbols to track
        symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT',
            'LINKUSDT', 'LTCUSDT', 'XRPUSDT', 'BCHUSDT', 'EOSUSDT'
        ]
        
        await ingestion_service.start_ingestion(symbols)
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await ingestion_service.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
