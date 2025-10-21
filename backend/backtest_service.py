import asyncio
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from datetime import datetime, timedelta
from enum import Enum
import threading
from concurrent.futures import ProcessPoolExecutor
import redis
import orjson
from scipy import stats
import numba
from numba import jit, njit

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from backend.storage.redis_client import RedisManager
from backend.storage.clickhouse_client import ClickHouseManager
from backend.models.cost_model import CostModel
from backend.models.labeling import LabelGenerator
from backend.utils.monitoring import MetricsCollector
from config.settings import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"

@dataclass
class Order:
    """Order representation for backtesting"""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float]
    timestamp: int
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    
@dataclass
class Trade:
    """Executed trade representation"""
    id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: int
    commission: float
    slippage: float

@dataclass
class Position:
    """Position tracking"""
    symbol: str
    quantity: float
    avg_entry_price: float
    timestamp: int
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

@dataclass
class BacktestConfig:
    """Backtest configuration"""
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_balance: float = 100000.0
    theta_up: float = 0.006
    theta_dn: float = 0.004
    tau: float = 0.75
    kappa: float = 1.20
    horizons: List[int] = None
    mode: str = "neutral"  # "neutral" or "conservative"
    max_position_size: float = 10000.0  # USD
    commission_rate: float = 0.001  # 0.1%
    enable_slippage: bool = True
    latency_injection: bool = True
    
    def __post_init__(self):
        if self.horizons is None:
            self.horizons = [5, 10, 30]

@dataclass
class BacktestResult:
    """Backtest results"""
    config: BacktestConfig
    trades: List[Trade]
    positions: List[Position]
    equity_curve: List[Tuple[int, float]]
    performance_metrics: Dict[str, float]
    signal_stats: Dict[str, Any]
    detailed_analysis: Dict[str, Any]

class TimingWheel:
    """Timing wheel for latency injection simulation"""
    
    def __init__(self, resolution_ms: int = 10, max_delay_ms: int = 1000):
        self.resolution_ms = resolution_ms
        self.max_delay_ms = max_delay_ms
        self.wheel_size = max_delay_ms // resolution_ms
        self.wheel = [deque() for _ in range(self.wheel_size)]
        self.current_tick = 0
        
    def schedule_event(self, delay_ms: int, event: Any):
        """Schedule an event with delay"""
        ticks_delay = max(1, delay_ms // self.resolution_ms)
        target_slot = (self.current_tick + ticks_delay) % self.wheel_size
        self.wheel[target_slot].append(event)
    
    def advance_tick(self) -> List[Any]:
        """Advance time and return ready events"""
        events = list(self.wheel[self.current_tick])
        self.wheel[self.current_tick].clear()
        self.current_tick = (self.current_tick + 1) % self.wheel_size
        return events

class OrderBook:
    """Simplified order book for backtesting"""
    
    def __init__(self):
        self.bids: List[Tuple[float, float]] = []  # [(price, size), ...]
        self.asks: List[Tuple[float, float]] = []
        self.last_update_time = 0
        
    def update(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], timestamp: int):
        """Update order book"""
        self.bids = sorted(bids, key=lambda x: x[0], reverse=True)  # Highest bid first
        self.asks = sorted(asks, key=lambda x: x[0])  # Lowest ask first
        self.last_update_time = timestamp
    
    def get_mid_price(self) -> Optional[float]:
        """Get mid price"""
        if self.bids and self.asks:
            return (self.bids[0][0] + self.asks[0][0]) / 2
        return None
    
    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread"""
        if self.bids and self.asks:
            return self.asks[0][0] - self.bids[0][0]
        return None
    
    def estimate_slippage(self, side: OrderSide, quantity: float, mode: str = "neutral") -> float:
        """Estimate market impact/slippage"""
        if not self.bids or not self.asks:
            return 0.0
        
        levels = 5 if mode == "neutral" else 2
        
        if side == OrderSide.BUY:
            available_liquidity = sum(size for _, size in self.asks[:levels])
            if quantity <= available_liquidity * 0.1:  # Small order
                return 0.0001  # 1 bp
            elif quantity <= available_liquidity * 0.5:  # Medium order
                return 0.0003  # 3 bp
            else:  # Large order
                return 0.0008  # 8 bp
        else:  # SELL
            available_liquidity = sum(size for _, size in self.bids[:levels])
            if quantity <= available_liquidity * 0.1:
                return 0.0001
            elif quantity <= available_liquidity * 0.5:
                return 0.0003
            else:
                return 0.0008

class MatchingEngine:
    """Event-driven matching engine for backtesting"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.order_book = OrderBook()
        self.timing_wheel = TimingWheel() if config.latency_injection else None
        self.pending_orders: Dict[str, Order] = {}
        self.order_counter = 0
        
    def update_market_data(self, market_data: Dict[str, Any]):
        """Update market data"""
        if 'depth' in market_data:
            depth = market_data['depth']
            bids = [(float(bid[0]), float(bid[1])) for bid in depth.get('bids', [])[:10]]
            asks = [(float(ask[0]), float(ask[1])) for ask in depth.get('asks', [])[:10]]
            self.order_book.update(bids, asks, market_data['timestamp'])
    
    def place_order(self, side: OrderSide, order_type: OrderType, quantity: float, 
                   price: Optional[float] = None, timestamp: int = None) -> Order:
        """Place an order"""
        self.order_counter += 1
        order = Order(
            id=f"order_{self.order_counter}",
            symbol=self.config.symbol,
            side=side,
            type=order_type,
            quantity=quantity,
            price=price,
            timestamp=timestamp or int(time.time() * 1000)
        )
        
        if self.timing_wheel:
            # Inject latency (50-200ms)
            latency_ms = np.random.uniform(50, 200)
            self.timing_wheel.schedule_event(latency_ms, ('process_order', order))
        else:
            self._process_order(order)
        
        return order
    
    def advance_time(self, timestamp: int) -> List[Trade]:
        """Advance time and process scheduled events"""
        trades = []
        
        if self.timing_wheel:
            events = self.timing_wheel.advance_tick()
            for event_type, event_data in events:
                if event_type == 'process_order':
                    trade = self._process_order(event_data)
                    if trade:
                        trades.append(trade)
        
        return trades
    
    def _process_order(self, order: Order) -> Optional[Trade]:
        """Process an order and generate trade if filled"""
        mid_price = self.order_book.get_mid_price()
        if not mid_price:
            order.status = OrderStatus.CANCELLED
            return None
        
        if order.type == OrderType.MARKET:
            # Market order - fill immediately
            fill_price = mid_price
            
            # Apply slippage if enabled
            if self.config.enable_slippage:
                slippage = self.order_book.estimate_slippage(order.side, order.quantity, self.config.mode)
                if order.side == OrderSide.BUY:
                    fill_price *= (1 + slippage)
                else:
                    fill_price *= (1 - slippage)
            else:
                slippage = 0.0
            
            # Calculate commission
            commission = order.quantity * fill_price * self.config.commission_rate
            
            # Fill order
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.avg_fill_price = fill_price
            
            # Create trade
            trade = Trade(
                id=f"trade_{order.id}",
                order_id=order.id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=fill_price,
                timestamp=order.timestamp,
                commission=commission,
                slippage=abs(fill_price - mid_price) / mid_price
            )
            
            return trade
        
        elif order.type == OrderType.LIMIT:
            # Limit order logic (simplified)
            if order.side == OrderSide.BUY and order.price >= mid_price:
                # Buy limit above mid - fill at mid
                return self._fill_limit_order(order, mid_price)
            elif order.side == OrderSide.SELL and order.price <= mid_price:
                # Sell limit below mid - fill at mid
                return self._fill_limit_order(order, mid_price)
            else:
                # Order not fillable - keep pending
                self.pending_orders[order.id] = order
                return None
        
        return None
    
    def _fill_limit_order(self, order: Order, fill_price: float) -> Trade:
        """Fill a limit order"""
        commission = order.quantity * fill_price * self.config.commission_rate
        
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        
        trade = Trade(
            id=f"trade_{order.id}",
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            timestamp=order.timestamp,
            commission=commission,
            slippage=0.0  # No slippage for limit orders
        )
        
        return trade

class BacktestEngine:
    """Main backtesting engine with event-driven simulation"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.matching_engine = MatchingEngine(config)
        self.cost_model = CostModel()
        self.label_generator = LabelGenerator()
        
        # Portfolio state
        self.balance = config.initial_balance
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[int, float]] = []
        
        # Signal tracking
        self.signals_generated = 0
        self.signals_triggered = 0
        self.hit_count = 0
        self.false_positives = 0
        
        # Performance tracking
        self.max_equity = config.initial_balance
        self.max_drawdown = 0.0
        
    async def run_backtest(self, market_data_source) -> BacktestResult:
        """Run complete backtest"""
        logger.info(f"Starting backtest for {self.config.symbol} from {self.config.start_date} to {self.config.end_date}")
        
        start_time = time.time()
        
        # Initialize data structures
        await self._initialize_backtest()
        
        # Main event loop
        async for market_data_batch in market_data_source:
            await self._process_market_data_batch(market_data_batch)
        
        # Finalize results
        result = self._generate_result()
        
        execution_time = time.time() - start_time
        logger.info(f"Backtest completed in {execution_time:.2f} seconds")
        
        return result
    
    async def _initialize_backtest(self):
        """Initialize backtest state"""
        await self.cost_model.initialize()
        
        # Set initial equity point
        self.equity_curve.append((int(self.config.start_date.timestamp() * 1000), self.balance))
    
    async def _process_market_data_batch(self, market_data_batch: List[Dict[str, Any]]):
        """Process a batch of market data"""
        for market_data in market_data_batch:
            # Update matching engine
            self.matching_engine.update_market_data(market_data)
            
            # Advance time and process orders
            trades = self.matching_engine.advance_time(market_data['timestamp'])
            for trade in trades:
                self._process_trade(trade)
            
            # Check for signals
            await self._check_for_signals(market_data)
            
            # Update portfolio
            self._update_portfolio(market_data)
    
    async def _check_for_signals(self, market_data: Dict[str, Any]):
        """Check for trading signals"""
        timestamp = market_data['timestamp']
        
        # Get features (simplified - in practice would come from feature service)
        features = self._extract_features(market_data)
        
        # Generate prediction (simplified)
        prediction = self._generate_prediction(features)
        
        self.signals_generated += 1
        
        # Check if signal meets criteria
        p_up = prediction.get('p_up', 0.0)
        expected_return = prediction.get('expected_return', 0.0)
        estimated_cost = await self.cost_model.estimate_cost(self.config.symbol, 10)  # 10min horizon
        
        utility = expected_return / max(estimated_cost, 0.001)
        
        # Decision logic
        if p_up >= self.config.tau and utility >= self.config.kappa:
            await self._execute_signal(market_data, prediction)
            self.signals_triggered += 1
    
    def _extract_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from market data (simplified)"""
        mid_price = self.matching_engine.order_book.get_mid_price() or 0
        spread = self.matching_engine.order_book.get_spread() or 0
        
        # Mock features for demo
        return {
            'mid_price': mid_price,
            'spread': spread,
            'qi_1': np.random.normal(0.1, 0.05),
            'ofi_10': np.random.normal(0.2, 0.1),
            'rv_ratio': np.random.uniform(0.8, 1.5)
        }
    
    def _generate_prediction(self, features: Dict[str, float]) -> Dict[str, float]:
        """Generate prediction from features (simplified)"""
        # Mock prediction logic
        base_prob = 0.4 + features.get('qi_1', 0) * 2 + features.get('ofi_10', 0)
        p_up = max(0.1, min(0.9, base_prob))
        
        expected_return = p_up * self.config.theta_up * np.random.uniform(1.2, 2.0)
        
        return {
            'p_up': p_up,
            'expected_return': expected_return,
            'features': features
        }
    
    async def _execute_signal(self, market_data: Dict[str, Any], prediction: Dict[str, float]):
        """Execute trading signal"""
        mid_price = self.matching_engine.order_book.get_mid_price()
        if not mid_price:
            return
        
        # Position sizing
        risk_amount = min(self.config.max_position_size, self.balance * 0.1)  # Max 10% of balance
        quantity = risk_amount / mid_price
        
        # Place buy order
        order = self.matching_engine.place_order(
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            timestamp=market_data['timestamp']
        )
        
        # Schedule exit order (simplified - would be more sophisticated)
        # For demo, exit after 10 minutes with simple logic
        
    def _process_trade(self, trade: Trade):
        """Process executed trade"""
        self.trades.append(trade)
        
        # Update balance
        if trade.side == OrderSide.BUY:
            self.balance -= (trade.quantity * trade.price + trade.commission)
        else:
            self.balance += (trade.quantity * trade.price - trade.commission)
        
        # Update position
        symbol = trade.symbol
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0.0,
                avg_entry_price=0.0,
                timestamp=trade.timestamp
            )
        
        position = self.positions[symbol]
        
        if trade.side == OrderSide.BUY:
            # Increase position
            total_cost = position.quantity * position.avg_entry_price + trade.quantity * trade.price
            position.quantity += trade.quantity
            position.avg_entry_price = total_cost / position.quantity if position.quantity > 0 else trade.price
        else:
            # Decrease position
            if position.quantity > 0:
                # Realize PnL
                pnl = trade.quantity * (trade.price - position.avg_entry_price)
                position.realized_pnl += pnl
                position.quantity -= trade.quantity
                
                if position.quantity <= 0:
                    # Close position
                    del self.positions[symbol]
    
    def _update_portfolio(self, market_data: Dict[str, Any]):
        """Update portfolio metrics"""
        timestamp = market_data['timestamp']
        mid_price = self.matching_engine.order_book.get_mid_price()
        
        # Calculate total equity
        total_equity = self.balance
        
        # Add unrealized PnL from positions
        symbol = self.config.symbol
        if symbol in self.positions and mid_price:
            position = self.positions[symbol]
            unrealized_pnl = position.quantity * (mid_price - position.avg_entry_price)
            total_equity += unrealized_pnl
            position.unrealized_pnl = unrealized_pnl
        
        # Update equity curve
        self.equity_curve.append((timestamp, total_equity))
        
        # Update drawdown
        if total_equity > self.max_equity:
            self.max_equity = total_equity
        
        current_drawdown = (self.max_equity - total_equity) / self.max_equity
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
    
    def _generate_result(self) -> BacktestResult:
        """Generate final backtest result"""
        # Calculate performance metrics
        if len(self.equity_curve) < 2:
            raise ValueError("Insufficient data for backtest analysis")
        
        initial_equity = self.equity_curve[0][1]
        final_equity = self.equity_curve[-1][1]
        total_return = (final_equity - initial_equity) / initial_equity
        
        # Calculate daily returns for Sharpe ratio
        equity_series = pd.Series([eq[1] for eq in self.equity_curve])
        daily_returns = equity_series.pct_change().dropna()
        
        sharpe_ratio = 0.0
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)  # Annualized
        
        # Hit rate calculation (simplified)
        hit_rate = self.hit_count / max(self.signals_triggered, 1)
        fpr = self.false_positives / max(self.signals_generated, 1)
        
        performance_metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'total_trades': len(self.trades),
            'hit_rate': hit_rate,
            'false_positive_rate': fpr,
            'profit_factor': self._calculate_profit_factor(),
            'avg_trade_return': total_return / max(len(self.trades), 1)
        }
        
        signal_stats = {
            'signals_generated': self.signals_generated,
            'signals_triggered': self.signals_triggered,
            'trigger_rate': self.signals_triggered / max(self.signals_generated, 1),
            'hit_count': self.hit_count,
            'false_positives': self.false_positives
        }
        
        detailed_analysis = {
            'trade_distribution': self._analyze_trade_distribution(),
            'monthly_returns': self._calculate_monthly_returns(),
            'risk_metrics': self._calculate_risk_metrics()
        }
        
        return BacktestResult(
            config=self.config,
            trades=self.trades,
            positions=list(self.positions.values()),
            equity_curve=self.equity_curve,
            performance_metrics=performance_metrics,
            signal_stats=signal_stats,
            detailed_analysis=detailed_analysis
        )
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        if not self.trades:
            return 0.0
        
        gross_profit = sum(t.quantity * (t.price - 50000) for t in self.trades if t.side == OrderSide.SELL and t.price > 50000)  # Mock
        gross_loss = sum(t.quantity * (50000 - t.price) for t in self.trades if t.side == OrderSide.SELL and t.price < 50000)  # Mock
        
        return gross_profit / max(abs(gross_loss), 1)
    
    def _analyze_trade_distribution(self) -> Dict[str, Any]:
        """Analyze trade distribution"""
        if not self.trades:
            return {}
        
        trade_pnls = [np.random.normal(100, 500) for _ in self.trades]  # Mock PnL
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': sum(1 for pnl in trade_pnls if pnl > 0),
            'losing_trades': sum(1 for pnl in trade_pnls if pnl <= 0),
            'avg_winner': np.mean([pnl for pnl in trade_pnls if pnl > 0]) if any(pnl > 0 for pnl in trade_pnls) else 0,
            'avg_loser': np.mean([pnl for pnl in trade_pnls if pnl <= 0]) if any(pnl <= 0 for pnl in trade_pnls) else 0,
            'largest_winner': max(trade_pnls) if trade_pnls else 0,
            'largest_loser': min(trade_pnls) if trade_pnls else 0
        }
    
    def _calculate_monthly_returns(self) -> Dict[str, float]:
        """Calculate monthly returns"""
        if len(self.equity_curve) < 2:
            return {}
        
        # Convert to pandas for easier date handling
        df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date', inplace=True)
        
        # Resample to monthly
        monthly = df.resample('M').last()
        monthly_returns = monthly['equity'].pct_change().dropna()
        
        return {
            month.strftime('%Y-%m'): return_val 
            for month, return_val in monthly_returns.items()
        }
    
    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate additional risk metrics"""
        if len(self.equity_curve) < 2:
            return {}
        
        equity_values = [eq[1] for eq in self.equity_curve]
        returns = pd.Series(equity_values).pct_change().dropna()
        
        var_95 = returns.quantile(0.05) if len(returns) > 0 else 0  # Value at Risk (95%)
        cvar_95 = returns[returns <= var_95].mean() if len(returns) > 0 else 0  # Conditional VaR
        
        return {
            'volatility': returns.std() * np.sqrt(252) if len(returns) > 0 else 0,  # Annualized
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': returns.skew() if len(returns) > 2 else 0,
            'kurtosis': returns.kurtosis() if len(returns) > 3 else 0
        }

class BacktestService:
    """Backtest service with parallel execution capability"""
    
    def __init__(self):
        self.settings = Settings()
        self.redis_manager = RedisManager()
        self.clickhouse_manager = ClickHouseManager()
        self.metrics_collector = MetricsCollector("backtest")
        
        # Process pool for parallel backtests
        self.process_pool = ProcessPoolExecutor(max_workers=4)
        
        # Cache for backtest results
        self.result_cache: Dict[str, BacktestResult] = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps: Dict[str, float] = {}
    
    async def initialize(self):
        """Initialize backtest service"""
        logger.info("Initializing backtest service...")
        
        await self.redis_manager.initialize()
        await self.clickhouse_manager.initialize()
        
        logger.info("Backtest service initialized")
    
    def _generate_cache_key(self, config: BacktestConfig) -> str:
        """Generate cache key for backtest configuration"""
        key_parts = [
            config.symbol,
            config.start_date.isoformat(),
            config.end_date.isoformat(),
            str(config.theta_up),
            str(config.theta_dn),
            str(config.tau),
            str(config.kappa),
            config.mode
        ]
        return ":".join(key_parts)
    
    async def run_backtest(self, config: BacktestConfig) -> BacktestResult:
        """Run backtest with caching"""
        cache_key = self._generate_cache_key(config)
        
        # Check cache
        if (cache_key in self.result_cache and 
            time.time() - self.cache_timestamps.get(cache_key, 0) < self.cache_ttl):
            logger.info(f"Returning cached backtest result for {cache_key}")
            return self.result_cache[cache_key]
        
        logger.info(f"Running new backtest for {config.symbol}")
        
        # Create backtest engine
        engine = BacktestEngine(config)
        
        # Get market data
        market_data_source = self._create_market_data_source(config)
        
        # Run backtest
        start_time = time.time()
        result = await engine.run_backtest(market_data_source)
        execution_time = time.time() - start_time
        
        # Update metrics
        self.metrics_collector.observe_histogram("backtest_execution_time_s", execution_time)
        self.metrics_collector.increment_counter("backtests_completed")
        
        # Cache result
        self.result_cache[cache_key] = result
        self.cache_timestamps[cache_key] = time.time()
        
        logger.info(f"Backtest completed in {execution_time:.2f} seconds")
        return result
    
    async def _create_market_data_source(self, config: BacktestConfig):
        """Create market data source for backtest period"""
        # In production, this would query ClickHouse for historical data
        # For demo, generate synthetic market data
        
        current_time = config.start_date
        end_time = config.end_date
        
        while current_time < end_time:
            # Generate synthetic market data batch
            batch_size = 100  # Process 100 data points at a time
            batch = []
            
            for _ in range(batch_size):
                if current_time >= end_time:
                    break
                
                # Mock market data
                mid_price = 50000 + np.random.normal(0, 1000)  # Mock BTC price
                spread = np.random.uniform(0.5, 2.0)
                
                market_data = {
                    'timestamp': int(current_time.timestamp() * 1000),
                    'symbol': config.symbol,
                    'depth': {
                        'bids': [
                            [mid_price - spread/2 - i * 0.5, np.random.uniform(0.1, 2.0)] 
                            for i in range(10)
                        ],
                        'asks': [
                            [mid_price + spread/2 + i * 0.5, np.random.uniform(0.1, 2.0)] 
                            for i in range(10)
                        ]
                    }
                }
                
                batch.append(market_data)
                current_time += timedelta(seconds=1)  # 1 second intervals
            
            yield batch
            
            # Small delay to prevent blocking
            await asyncio.sleep(0.001)
    
    async def get_backtest_status(self, config: BacktestConfig) -> Dict[str, Any]:
        """Get backtest status"""
        cache_key = self._generate_cache_key(config)
        
        if cache_key in self.result_cache:
            return {
                'status': 'completed',
                'cached': True,
                'cache_age_seconds': time.time() - self.cache_timestamps.get(cache_key, 0)
            }
        else:
            return {
                'status': 'not_started',
                'cached': False
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up backtest service...")
        
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=True)
        
        logger.info("Backtest service cleanup complete")

# FastAPI application
app = FastAPI(title="Crypto Surge Prediction Backtest API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global backtest service
backtest_service = None

@app.on_event("startup")
async def startup_event():
    global backtest_service
    backtest_service = BacktestService()
    await backtest_service.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    global backtest_service
    if backtest_service:
        await backtest_service.cleanup()

@app.post("/backtest/run")
async def run_backtest(
    symbol: str = Query(..., description="Trading symbol"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    theta_up: float = Query(0.006, description="Up threshold"),
    theta_dn: float = Query(0.004, description="Down threshold"),
    tau: float = Query(0.75, description="Probability threshold"),
    kappa: float = Query(1.20, description="Utility threshold"),
    mode: str = Query("neutral", description="Backtest mode (neutral/conservative)"),
    max_position_size: float = Query(10000.0, description="Max position size USD"),
    background_tasks: BackgroundTasks = None
):
    """Run backtest with specified parameters"""
    try:
        config = BacktestConfig(
            symbol=symbol,
            start_date=datetime.fromisoformat(start_date),
            end_date=datetime.fromisoformat(end_date),
            theta_up=theta_up,
            theta_dn=theta_dn,
            tau=tau,
            kappa=kappa,
            mode=mode,
            max_position_size=max_position_size
        )
        
        result = await backtest_service.run_backtest(config)
        
        # Convert result to JSON-serializable format
        return {
            'config': asdict(result.config),
            'performance_metrics': result.performance_metrics,
            'signal_stats': result.signal_stats,
            'detailed_analysis': result.detailed_analysis,
            'equity_curve': result.equity_curve[-100:],  # Last 100 points to avoid large response
            'total_trades': len(result.trades),
            'execution_summary': {
                'start_time': result.config.start_date.isoformat(),
                'end_time': result.config.end_date.isoformat(),
                'mode': result.config.mode,
                'completed': True
            }
        }
    
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")

@app.get("/backtest/status")
async def get_backtest_status(
    symbol: str = Query(..., description="Trading symbol"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    theta_up: float = Query(0.006, description="Up threshold"),
    theta_dn: float = Query(0.004, description="Down threshold"),
    tau: float = Query(0.75, description="Probability threshold"),
    kappa: float = Query(1.20, description="Utility threshold"),
    mode: str = Query("neutral", description="Backtest mode")
):
    """Get backtest status"""
    config = BacktestConfig(
        symbol=symbol,
        start_date=datetime.fromisoformat(start_date),
        end_date=datetime.fromisoformat(end_date),
        theta_up=theta_up,
        theta_dn=theta_dn,
        tau=tau,
        kappa=kappa,
        mode=mode
    )
    
    return await backtest_service.get_backtest_status(config)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": int(time.time() * 1000),
        "cache_size": len(backtest_service.result_cache) if backtest_service else 0
    }

if __name__ == "__main__":
    uvicorn.run(
        "backend.backtest_service:app",
        host="0.0.0.0",
        port=8002,
        workers=1,
        loop="uvloop"
    )
