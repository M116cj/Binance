"""
WebSocket utilities for high-performance market data ingestion.
Handles connection management, reconnection logic, and message processing.
"""

import asyncio
import websockets
import json
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
import ssl
import gzip
import zlib
from enum import Enum
import uuid
import os
from contextlib import asynccontextmanager

# Import orjson for faster JSON processing
try:
    import orjson
    JSON_LOADS = orjson.loads
    JSON_DUMPS = orjson.dumps
except ImportError:
    import json
    JSON_LOADS = json.loads
    JSON_DUMPS = json.dumps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    """WebSocket connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting" 
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"

@dataclass
class ConnectionConfig:
    """WebSocket connection configuration"""
    url: str
    symbols: List[str]
    streams: List[str]
    max_reconnect_attempts: int = 10
    initial_reconnect_delay: float = 0.5
    max_reconnect_delay: float = 60.0
    heartbeat_interval: float = 5.0
    ping_timeout: float = 10.0
    close_timeout: float = 10.0
    enable_compression: bool = True
    max_message_size: int = 1024 * 1024  # 1MB
    rate_limit_per_second: int = 100

@dataclass  
class ConnectionStats:
    """Connection statistics"""
    connection_id: str
    state: ConnectionState
    connected_at: Optional[float] = None
    disconnected_at: Optional[float] = None
    total_messages: int = 0
    total_bytes: int = 0
    errors: int = 0
    reconnect_attempts: int = 0
    last_ping_time: float = 0
    last_pong_time: float = 0
    avg_latency_ms: float = 0

class MessageBuffer:
    """Circular buffer for WebSocket messages"""
    
    def __init__(self, maxsize: int = 10000):
        self.maxsize = maxsize
        self.buffer = deque(maxlen=maxsize)
        self.lock = threading.Lock()
        self.overflow_count = 0
    
    def append(self, message: Any):
        """Add message to buffer"""
        with self.lock:
            if len(self.buffer) >= self.maxsize:
                self.overflow_count += 1
            self.buffer.append(message)
    
    def get_messages(self, count: Optional[int] = None) -> List[Any]:
        """Get messages from buffer"""
        with self.lock:
            if count is None:
                messages = list(self.buffer)
                self.buffer.clear()
            else:
                messages = []
                for _ in range(min(count, len(self.buffer))):
                    if self.buffer:
                        messages.append(self.buffer.popleft())
            return messages
    
    def size(self) -> int:
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)

class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, rate_per_second: int):
        self.rate = rate_per_second
        self.tokens = rate_per_second
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens"""
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_update
            
            # Add tokens based on elapsed time
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.last_update = current_time
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def wait_time(self, tokens: int = 1) -> float:
        """Get wait time for tokens"""
        with self.lock:
            deficit = tokens - self.tokens
            if deficit <= 0:
                return 0.0
            return deficit / self.rate

class WebSocketConnection:
    """Individual WebSocket connection with reconnection logic"""
    
    def __init__(self, config: ConnectionConfig, message_handler: Callable, 
                 connection_id: str, metrics_callback: Optional[Callable] = None):
        self.config = config
        self.message_handler = message_handler
        self.connection_id = connection_id
        self.metrics_callback = metrics_callback
        
        # Connection state
        self.state = ConnectionState.DISCONNECTED
        self.websocket = None
        self.stats = ConnectionStats(connection_id=connection_id, state=self.state)
        
        # Reconnection logic
        self.reconnect_attempts = 0
        self.should_reconnect = True
        self.reconnect_delay = config.initial_reconnect_delay
        
        # Message buffering
        self.message_buffer = MessageBuffer()
        self.rate_limiter = RateLimiter(config.rate_limit_per_second)
        
        # Tasks
        self.connection_task = None
        self.heartbeat_task = None
        
        # Events
        self.connected_event = asyncio.Event()
        self.shutdown_event = asyncio.Event()
        
        # Locks
        self.state_lock = asyncio.Lock()
    
    async def start(self):
        """Start the WebSocket connection"""
        logger.info(f"Starting WebSocket connection {self.connection_id}")
        
        self.connection_task = asyncio.create_task(self._connection_loop())
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
    
    async def stop(self):
        """Stop the WebSocket connection"""
        logger.info(f"Stopping WebSocket connection {self.connection_id}")
        
        self.should_reconnect = False
        self.shutdown_event.set()
        
        # Cancel tasks
        if self.connection_task:
            self.connection_task.cancel()
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        
        # Close WebSocket
        if self.websocket:
            await self.websocket.close()
        
        await self._set_state(ConnectionState.DISCONNECTED)
    
    async def _connection_loop(self):
        """Main connection loop with reconnection logic"""
        while self.should_reconnect and not self.shutdown_event.is_set():
            try:
                await self._connect()
                await self._message_loop()
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"Connection {self.connection_id} closed")
                await self._handle_disconnection()
                
            except Exception as e:
                logger.error(f"Connection {self.connection_id} error: {e}")
                await self._handle_error(e)
                
            # Wait before reconnecting
            if self.should_reconnect and not self.shutdown_event.is_set():
                await self._wait_for_reconnect()
    
    async def _connect(self):
        """Establish WebSocket connection"""
        await self._set_state(ConnectionState.CONNECTING)
        
        # SSL context for secure connections
        ssl_context = ssl.create_default_context() if self.config.url.startswith('wss://') else None
        
        # Connection headers
        headers = {}
        if self.config.enable_compression:
            headers['Sec-WebSocket-Extensions'] = 'permessage-deflate'
        
        logger.info(f"Connecting to {self.config.url} for connection {self.connection_id}")
        
        self.websocket = await websockets.connect(
            self.config.url,
            ssl=ssl_context,
            extra_headers=headers,
            ping_interval=self.config.heartbeat_interval,
            ping_timeout=self.config.ping_timeout,
            close_timeout=self.config.close_timeout,
            max_size=self.config.max_message_size,
            compression='deflate' if self.config.enable_compression else None
        )
        
        await self._set_state(ConnectionState.CONNECTED)
        self.connected_event.set()
        
        # Subscribe to streams
        await self._subscribe_streams()
        
        # Reset reconnection state
        self.reconnect_attempts = 0
        self.reconnect_delay = self.config.initial_reconnect_delay
        
        logger.info(f"Connection {self.connection_id} established successfully")
    
    async def _subscribe_streams(self):
        """Subscribe to WebSocket streams"""
        if not self.config.symbols or not self.config.streams:
            return
        
        # Build subscription message
        streams = []
        for symbol in self.config.symbols:
            for stream in self.config.streams:
                stream_name = f"{symbol.lower()}@{stream}"
                streams.append(stream_name)
        
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": int(time.time())
        }
        
        await self.websocket.send(JSON_DUMPS(subscribe_msg))
        logger.info(f"Subscribed to {len(streams)} streams on connection {self.connection_id}")
    
    async def _message_loop(self):
        """Main message processing loop"""
        async for message in self.websocket:
            try:
                # Rate limiting
                if not self.rate_limiter.acquire():
                    wait_time = self.rate_limiter.wait_time()
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                        continue
                
                # Process message
                await self._process_message(message)
                
            except Exception as e:
                logger.error(f"Error processing message on {self.connection_id}: {e}")
                self.stats.errors += 1
                
                if self.metrics_callback:
                    self.metrics_callback('message_error', 1, {'connection': self.connection_id})
    
    async def _process_message(self, message: Union[str, bytes]):
        """Process individual message"""
        start_time = time.time()
        
        try:
            # Handle binary messages (compressed)
            if isinstance(message, bytes):
                if self.config.enable_compression:
                    try:
                        # Try gzip decompression first
                        message = gzip.decompress(message).decode('utf-8')
                    except:
                        try:
                            # Try zlib decompression
                            message = zlib.decompress(message).decode('utf-8')
                        except:
                            # Raw bytes to string
                            message = message.decode('utf-8')
                else:
                    message = message.decode('utf-8')
            
            # Parse JSON
            data = JSON_LOADS(message)
            
            # Update statistics
            self.stats.total_messages += 1
            self.stats.total_bytes += len(message)
            
            # Handle different message types
            if isinstance(data, dict):
                if 'stream' in data and 'data' in data:
                    # Market data message
                    await self._handle_market_data(data)
                elif 'result' in data:
                    # Subscription response
                    await self._handle_subscription_response(data)
                elif 'error' in data:
                    # Error message
                    logger.error(f"WebSocket error on {self.connection_id}: {data['error']}")
                    self.stats.errors += 1
            
            # Calculate latency
            processing_time = (time.time() - start_time) * 1000
            self._update_latency(processing_time)
            
            # Call metrics callback
            if self.metrics_callback:
                self.metrics_callback('message_processed', 1, {
                    'connection': self.connection_id,
                    'processing_time_ms': processing_time
                })
                
        except Exception as e:
            logger.error(f"Error processing message on {self.connection_id}: {e}")
            self.stats.errors += 1
    
    async def _handle_market_data(self, data: dict):
        """Handle market data message"""
        try:
            # Add connection metadata
            data['connection_id'] = self.connection_id
            data['received_at'] = int(time.time() * 1000)
            
            # Buffer message for batch processing
            self.message_buffer.append(data)
            
            # Call message handler
            if self.message_handler:
                await self.message_handler(data)
                
        except Exception as e:
            logger.error(f"Error handling market data on {self.connection_id}: {e}")
    
    async def _handle_subscription_response(self, data: dict):
        """Handle subscription response"""
        if data.get('result') is None:
            logger.info(f"Successfully subscribed on connection {self.connection_id}")
        else:
            logger.warning(f"Subscription response on {self.connection_id}: {data}")
    
    def _update_latency(self, processing_time_ms: float):
        """Update average latency calculation"""
        if self.stats.avg_latency_ms == 0:
            self.stats.avg_latency_ms = processing_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats.avg_latency_ms = (alpha * processing_time_ms + 
                                        (1 - alpha) * self.stats.avg_latency_ms)
    
    async def _heartbeat_loop(self):
        """Heartbeat loop for connection health monitoring"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                if self.state == ConnectionState.CONNECTED and self.websocket:
                    # Send ping
                    ping_time = time.time()
                    pong_waiter = await self.websocket.ping()
                    
                    self.stats.last_ping_time = ping_time
                    
                    try:
                        await asyncio.wait_for(pong_waiter, timeout=self.config.ping_timeout)
                        self.stats.last_pong_time = time.time()
                        
                        # Update latency
                        latency_ms = (self.stats.last_pong_time - self.stats.last_ping_time) * 1000
                        self._update_latency(latency_ms)
                        
                    except asyncio.TimeoutError:
                        logger.warning(f"Ping timeout on connection {self.connection_id}")
                        # Connection might be dead, let it reconnect
                        break
                        
            except Exception as e:
                logger.error(f"Heartbeat error on connection {self.connection_id}: {e}")
                break
    
    async def _handle_disconnection(self):
        """Handle connection disconnection"""
        await self._set_state(ConnectionState.DISCONNECTED)
        self.connected_event.clear()
        
        self.stats.disconnected_at = time.time()
        
        if self.metrics_callback:
            self.metrics_callback('connection_disconnected', 1, {'connection': self.connection_id})
    
    async def _handle_error(self, error: Exception):
        """Handle connection error"""
        await self._set_state(ConnectionState.FAILED)
        self.stats.errors += 1
        
        if self.metrics_callback:
            self.metrics_callback('connection_error', 1, {
                'connection': self.connection_id,
                'error_type': type(error).__name__
            })
    
    async def _wait_for_reconnect(self):
        """Wait with exponential backoff before reconnecting"""
        if not self.should_reconnect:
            return
        
        await self._set_state(ConnectionState.RECONNECTING)
        
        self.reconnect_attempts += 1
        self.stats.reconnect_attempts = self.reconnect_attempts
        
        # Check max attempts
        if self.reconnect_attempts >= self.config.max_reconnect_attempts:
            logger.error(f"Connection {self.connection_id} exceeded max reconnection attempts")
            self.should_reconnect = False
            return
        
        # Exponential backoff
        delay = min(self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)), 
                   self.config.max_reconnect_delay)
        
        logger.info(f"Reconnecting {self.connection_id} in {delay:.1f} seconds (attempt {self.reconnect_attempts})")
        
        await asyncio.sleep(delay)
    
    async def _set_state(self, new_state: ConnectionState):
        """Set connection state thread-safely"""
        async with self.state_lock:
            old_state = self.state
            self.state = new_state
            self.stats.state = new_state
            
            if new_state == ConnectionState.CONNECTED:
                self.stats.connected_at = time.time()
            
            logger.debug(f"Connection {self.connection_id} state: {old_state} -> {new_state}")
    
    def get_stats(self) -> ConnectionStats:
        """Get connection statistics"""
        return self.stats
    
    def get_buffered_messages(self, count: Optional[int] = None) -> List[Any]:
        """Get buffered messages"""
        return self.message_buffer.get_messages(count)

class WebSocketManager:
    """Manages multiple WebSocket connections with load balancing"""
    
    def __init__(self, base_url: str = "wss://stream.binance.com:9443/ws/",
                 symbols_per_connection: int = 25):
        self.base_url = base_url
        self.symbols_per_connection = symbols_per_connection
        
        # Connection management
        self.connections: Dict[str, WebSocketConnection] = {}
        self.connection_configs: Dict[str, ConnectionConfig] = {}
        
        # Message handling
        self.message_handlers: List[Callable] = []
        self.metrics_callbacks: List[Callable] = []
        
        # Statistics
        self.total_connections = 0
        self.total_messages = 0
        self.total_errors = 0
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.shutdown_flag = False
        
        # Default streams
        self.default_streams = [
            'aggTrade', 'bookTicker', 'depth@100ms', 'kline_1m', 
            'ticker', 'markPrice', 'openInterest'
        ]
    
    def add_message_handler(self, handler: Callable):
        """Add message handler"""
        self.message_handlers.append(handler)
    
    def add_metrics_callback(self, callback: Callable):
        """Add metrics callback"""
        self.metrics_callbacks.append(callback)
    
    async def start_connections(self, symbols: List[str], streams: Optional[List[str]] = None):
        """Start WebSocket connections for symbols"""
        if streams is None:
            streams = self.default_streams
        
        logger.info(f"Starting WebSocket connections for {len(symbols)} symbols")
        
        # Group symbols by connection
        symbol_groups = [symbols[i:i + self.symbols_per_connection] 
                        for i in range(0, len(symbols), self.symbols_per_connection)]
        
        # Create connections
        for i, symbol_group in enumerate(symbol_groups):
            connection_id = f"conn_{i}"
            
            config = ConnectionConfig(
                url=self.base_url,
                symbols=symbol_group,
                streams=streams,
                max_reconnect_attempts=10,
                initial_reconnect_delay=0.5,
                max_reconnect_delay=self.symbols_per_connection,  # Scale with symbols
                heartbeat_interval=5.0,
                enable_compression=True,
                rate_limit_per_second=100
            )
            
            connection = WebSocketConnection(
                config=config,
                message_handler=self._handle_message,
                connection_id=connection_id,
                metrics_callback=self._handle_metrics
            )
            
            self.connections[connection_id] = connection
            self.connection_configs[connection_id] = config
            
            # Start connection
            await connection.start()
            self.total_connections += 1
        
        logger.info(f"Started {len(self.connections)} WebSocket connections")
    
    async def stop_all_connections(self):
        """Stop all WebSocket connections"""
        logger.info("Stopping all WebSocket connections...")
        
        self.shutdown_flag = True
        
        # Stop all connections
        stop_tasks = []
        for connection in self.connections.values():
            stop_tasks.append(connection.stop())
        
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("All WebSocket connections stopped")
    
    async def _handle_message(self, message: dict):
        """Handle message from any connection"""
        self.total_messages += 1
        
        # Call all message handlers
        for handler in self.message_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    # Run sync handler in thread pool
                    await asyncio.get_event_loop().run_in_executor(
                        self.executor, handler, message
                    )
            except Exception as e:
                logger.error(f"Error in message handler: {e}")
                self.total_errors += 1
    
    def _handle_metrics(self, metric_name: str, value: float, labels: dict):
        """Handle metrics from connections"""
        for callback in self.metrics_callbacks:
            try:
                callback(metric_name, value, labels)
            except Exception as e:
                logger.error(f"Error in metrics callback: {e}")
    
    def get_connection_stats(self) -> Dict[str, ConnectionStats]:
        """Get statistics for all connections"""
        return {conn_id: conn.get_stats() 
                for conn_id, conn in self.connections.items()}
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall statistics"""
        connection_stats = self.get_connection_stats()
        
        # Aggregate stats
        total_messages = sum(stats.total_messages for stats in connection_stats.values())
        total_bytes = sum(stats.total_bytes for stats in connection_stats.values())
        total_errors = sum(stats.errors for stats in connection_stats.values())
        
        connected_count = sum(1 for stats in connection_stats.values() 
                            if stats.state == ConnectionState.CONNECTED)
        
        avg_latency = 0
        if connection_stats:
            avg_latency = sum(stats.avg_latency_ms for stats in connection_stats.values()) / len(connection_stats)
        
        return {
            'total_connections': len(self.connections),
            'connected_connections': connected_count,
            'total_messages': total_messages,
            'total_bytes': total_bytes,
            'total_errors': total_errors,
            'average_latency_ms': avg_latency,
            'symbols_per_connection': self.symbols_per_connection,
            'error_rate': total_errors / max(total_messages, 1)
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of WebSocket connections"""
        connection_stats = self.get_connection_stats()
        overall_stats = self.get_overall_stats()
        
        # Determine overall health
        connected_pct = overall_stats['connected_connections'] / max(overall_stats['total_connections'], 1)
        error_rate = overall_stats['error_rate']
        
        if connected_pct >= 0.8 and error_rate < 0.01:
            health_status = 'healthy'
        elif connected_pct >= 0.5 and error_rate < 0.05:
            health_status = 'degraded'  
        else:
            health_status = 'unhealthy'
        
        return {
            'status': health_status,
            'connected_percentage': connected_pct * 100,
            'error_rate_percentage': error_rate * 100,
            'average_latency_ms': overall_stats['average_latency_ms'],
            'total_messages': overall_stats['total_messages'],
            'connections': {
                conn_id: {
                    'state': stats.state.value,
                    'messages': stats.total_messages,
                    'errors': stats.errors,
                    'latency_ms': stats.avg_latency_ms,
                    'reconnect_attempts': stats.reconnect_attempts
                }
                for conn_id, stats in connection_stats.items()
            }
        }
    
    async def reconnect_failed_connections(self):
        """Reconnect any failed connections"""
        failed_connections = [
            conn_id for conn_id, conn in self.connections.items()
            if conn.state in [ConnectionState.FAILED, ConnectionState.DISCONNECTED]
        ]
        
        if failed_connections:
            logger.info(f"Reconnecting {len(failed_connections)} failed connections")
            
            for conn_id in failed_connections:
                try:
                    connection = self.connections[conn_id]
                    connection.should_reconnect = True
                    await connection.start()
                except Exception as e:
                    logger.error(f"Error reconnecting {conn_id}: {e}")
    
    async def rebalance_connections(self, new_symbols: List[str]):
        """Rebalance connections with new symbol list"""
        logger.info(f"Rebalancing connections with {len(new_symbols)} symbols")
        
        # Stop existing connections
        await self.stop_all_connections()
        
        # Clear old connections
        self.connections.clear()
        self.connection_configs.clear()
        self.total_connections = 0
        
        # Start new connections
        await self.start_connections(new_symbols)
        
        logger.info("Connection rebalancing completed")
