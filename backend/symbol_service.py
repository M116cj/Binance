"""
币安交易对服务
从币安API获取所有可用的USDT交易对
"""

import logging
import httpx
import asyncio
from typing import List, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class BinanceSymbolService:
    """币安交易对服务"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com"
        self.cache: Optional[Dict] = None
        self.cache_time: Optional[datetime] = None
        self.cache_ttl = timedelta(hours=1)
    
    async def get_usdt_symbols(self) -> List[Dict[str, str]]:
        """
        获取所有USDT交易对
        返回: [{"symbol": "BTCUSDT", "name": "比特币", "baseAsset": "BTC"}, ...]
        """
        if self._is_cache_valid():
            logger.info("返回缓存的交易对数据")
            return self.cache
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/v3/exchangeInfo")
                response.raise_for_status()
                data = response.json()
                
                symbols = []
                for symbol_info in data.get('symbols', []):
                    quote_asset = symbol_info.get('quoteAsset', '')
                    base_asset = symbol_info.get('baseAsset', '')
                    symbol = symbol_info.get('symbol', '')
                    status = symbol_info.get('status', '')
                    
                    if quote_asset == 'USDT' and status == 'TRADING':
                        name = self._get_chinese_name(base_asset)
                        symbols.append({
                            'symbol': symbol,
                            'baseAsset': base_asset,
                            'name': name,
                            'displayName': f"{name} ({base_asset})"
                        })
                
                symbols.sort(key=lambda x: x['symbol'])
                
                self.cache = symbols
                self.cache_time = datetime.now()
                
                logger.info(f"成功获取 {len(symbols)} 个USDT交易对")
                return symbols
                
        except Exception as e:
            logger.error(f"获取币安交易对失败: {e}")
            if self.cache:
                logger.info("返回旧缓存数据")
                return self.cache
            return self._get_fallback_symbols()
    
    def _is_cache_valid(self) -> bool:
        """检查缓存是否有效"""
        if self.cache is None or self.cache_time is None:
            return False
        return datetime.now() - self.cache_time < self.cache_ttl
    
    def _get_chinese_name(self, base_asset: str) -> str:
        """获取币种的中文名称"""
        name_map = {
            'BTC': '比特币',
            'ETH': '以太坊',
            'BNB': '币安币',
            'SOL': '索拉纳',
            'XRP': '瑞波币',
            'ADA': '艾达币',
            'DOGE': '狗狗币',
            'MATIC': '马蹄币',
            'DOT': '波卡币',
            'AVAX': '雪崩币',
            'SHIB': '柴犬币',
            'LTC': '莱特币',
            'LINK': '链克币',
            'UNI': '优尼币',
            'ATOM': '宇宙币',
            'ETC': '以太经典',
            'XLM': '恒星币',
            'BCH': '比特现金',
            'NEAR': '近距币',
            'APT': 'Aptos',
            'ARB': 'Arbitrum',
            'OP': 'Optimism',
            'TRX': '波场币',
            'FIL': '文件币',
            'ALGO': '阿尔戈',
            'VET': '唯链',
            'ICP': '互联网计算机',
            'SAND': '沙盒',
            'MANA': '去中心地',
            'AXS': 'Axie',
            'THETA': '西塔',
            'FTM': '幻影',
            'EGLD': 'Elrond',
            'XTZ': 'Tezos',
            'AAVE': 'Aave',
            'GRT': '图谱',
            'EOS': '柚子币',
            'CAKE': '煎饼',
            'MKR': 'Maker',
            'STX': 'Stacks',
            'INJ': 'Injective',
            'PEPE': '佩佩蛙',
        }
        return name_map.get(base_asset, base_asset)
    
    def _get_fallback_symbols(self) -> List[Dict[str, str]]:
        """返回备用的交易对列表（当API调用失败时）"""
        logger.warning("使用备用交易对列表")
        fallback = [
            {'symbol': 'BTCUSDT', 'baseAsset': 'BTC', 'name': '比特币', 'displayName': '比特币 (BTC)'},
            {'symbol': 'ETHUSDT', 'baseAsset': 'ETH', 'name': '以太坊', 'displayName': '以太坊 (ETH)'},
            {'symbol': 'BNBUSDT', 'baseAsset': 'BNB', 'name': '币安币', 'displayName': '币安币 (BNB)'},
            {'symbol': 'SOLUSDT', 'baseAsset': 'SOL', 'name': '索拉纳', 'displayName': '索拉纳 (SOL)'},
            {'symbol': 'XRPUSDT', 'baseAsset': 'XRP', 'name': '瑞波币', 'displayName': '瑞波币 (XRP)'},
        ]
        return fallback

symbol_service = BinanceSymbolService()
