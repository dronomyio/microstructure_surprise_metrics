import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import asyncio
import aiohttp
from dataclasses import dataclass
from datetime import datetime, timedelta
import pyarrow.parquet as pq
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pysurprise_metrics  # Our C++ extension

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SurpriseConfig:
    """Configuration for surprise metrics calculation"""
    garch_omega: float = 0.00001
    garch_alpha: float = 0.05
    garch_beta: float = 0.94
    jump_threshold: float = 4.0
    window_size: int = 100
    hawkes_mu: float = 0.1
    hawkes_phi: float = 0.3
    hawkes_kappa: float = 0.8
    num_gpus: int = 1
    buffer_size: int = 1_000_000

class PolygonDataFetcher:
    """Async fetcher for Polygon.io flat files"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v3/files/flatfiles"
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
        
    async genuineSymbolAlertData(self, date: str, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch both trades and quotes for a symbol on a specific date"""
        trades_task = self._fetch_trades(date, symbol)
        quotes_task = self._fetch_quotes(date, symbol)
        
        trades_df, quotes_df = await asyncio.gather(trades_task, quotes_task)
        return trades_df, quotes_df
        
    async def _fetch_trades(self, date: str, symbol: str) -> pd.DataFrame:
        """Fetch trades data"""
        url = f"{self.base_url}/us/stocks/trades/{date}/{symbol}.csv.gz"
        params = {"apiKey": self.api_key}
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.read()
                # Decompress and parse
                import gzip
                import io
                decompressed = gzip.decompress(data)
                df = pd.read_csv(io.BytesIO(decompressed))
                
                # Convert nanosecond timestamps
                df['timestamp'] = pd.to_datetime(df['participant_timestamp'], unit='ns')
                return df
            else:
                logger.error(f"Failed to fetch trades: {response.status}")
                return pd.DataFrame()
                
    async def _fetch_quotes(self, date: str, symbol: str) -> pd.DataFrame:
        """Fetch quotes data"""
        url = f"{self.base_url}/us/stocks/quotes/{date}/{symbol}.csv.gz"
        params = {"apiKey": self.api_key}
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.read()
                import gzip
                import io
                decompressed = gzip.decompress(data)
                df = pd.read_csv(io.BytesIO(decompressed))
                
                df['timestamp'] = pd.to_datetime(df['participant_timestamp'], unit='ns')
                return df
            else:
                logger.error(f"Failed to fetch quotes: {response.status}")
                return pd.DataFrame()

class SurpriseMetricsEngine:
    """Main engine for computing surprise metrics"""
    
    def __init__(self, config: SurpriseConfig = SurpriseConfig()):
        self.config = config
        self.calculator = pysurprise_metrics.MetricsCalculator(
            num_gpus=config.num_gpus,
            buffer_size=config.buffer_size
        )
        self._configure_calculator()
        
    def _configure_calculator(self):
        """Configure the C++ calculator with parameters"""
        self.calculator.set_garch_params(
            self.config.garch_omega,
            self.config.garch_alpha,
            self.config.garch_beta
        )
        self.calculator.set_jump_threshold(self.config.jump_threshold)
        self.calculator.set_window_size(self.config.window_size)
        
    def process_trades(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Process trades and compute surprise metrics"""
        if trades_df.empty:
            return pd.DataFrame()
            
        # Convert to numpy arrays for C++ processing
        timestamps = trades_df['participant_timestamp'].values.astype(np.int64)
        prices = trades_df['price'].values.astype(np.float32)
        sizes = trades_df['size'].values.astype(np.int64)
        
        # Call C++ implementation
        metrics = self.calculator.process_trades_batch(timestamps, prices, sizes)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(metrics, columns=[
            'timestamp', 'standardized_return', 'lee_mykland_stat',
            'bns_stat', 'trade_intensity_zscore', 'jump_detected'
        ])
        results_df['timestamp'] = pd.to_datetime(results_df['timestamp'], unit='ns')
        
        return results_df
        
    def process_quotes(self, quotes_df: pd.DataFrame) -> pd.DataFrame:
        """Process quotes and compute microstructure metrics"""
        if quotes_df.empty:
            return pd.DataFrame()
            
        # Compute bid-ask spread
        quotes_df['spread'] = quotes_df['ask_price'] - quotes_df['bid_price']
        quotes_df['mid_price'] = (quotes_df['ask_price'] + quotes_df['bid_price']) / 2
        
        # Compute order imbalance
        quotes_df['order_imbalance'] = (
            (quotes_df['bid_size'] - quotes_df['ask_size']) /
            (quotes_df['bid_size'] + quotes_df['ask_size'])
        )
        
        return quotes_df
        
    def compute_composite_alerts(self, 
                                trades_metrics: pd.DataFrame,
                                quotes_metrics: pd.DataFrame) -> pd.DataFrame:
        """Combine trades and quotes metrics for composite alerting"""
        
        # Merge on timestamp (with tolerance for nanosecond precision)
        merged = pd.merge_asof(
            trades_metrics.sort_values('timestamp'),
            quotes_metrics.sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
            tolerance=pd.Timedelta(microseconds=1)
        )
        
        # Compute composite score
        merged['alert_score'] = (
            merged['standardized_return'].abs() * 0.3 +
            merged['lee_mykland_stat'] * 0.2 +
            merged['bns_stat'] * 0.2 +
            merged['trade_intensity_zscore'] * 0.2 +
            merged['order_imbalance'].abs() * 0.1
        )
        
        # Flag alerts
        merged['alert'] = merged['alert_score'] > 3.0
        
        return merged

class RealTimeMonitor:
    """Real-time monitoring system"""
    
    def __init__(self, api_key: str, symbols: List[str], config: SurpriseConfig = SurpriseConfig()):
        self.api_key = api_key
        self.symbols = symbols
        self.engine = SurpriseMetricsEngine(config)
        self.is_running = False
        
    async def start(self):
        """Start real-time monitoring"""
        self.is_running = True
        
        async with PolygonDataFetcher(self.api_key) as fetcher:
            while self.is_running:
                try:
                    # Process each symbol
                    tasks = []
                    for symbol in self.symbols:
                        task = self._process_symbol(fetcher, symbol)
                        tasks.append(task)
                        
                    results = await asyncio.gather(*tasks)
                    
                    # Aggregate alerts
                    all_alerts = pd.concat([r for r in results if r is not None])
                    
                    if not all_alerts.empty:
                        high_priority = all_alerts[all_alerts['alert_score'] > 5.0]
                        if not high_priority.empty:
                            await self._send_alerts(high_priority)
                            
                    # Sleep briefly before next iteration
                    await asyncio.sleep(0.1)  # 100ms for high-frequency monitoring
                    
                except Exception as e:
                    logger.error(f"Error in monitoring: {e}")
                    await asyncio.sleep(1)
                    
    async def _process_symbol(self, fetcher: PolygonDataFetcher, symbol: str) -> Optional[pd.DataFrame]:
        """Process a single symbol"""
        try:
            # Get current date
            date = datetime.now().strftime("%Y-%m-%d")
            
            # Fetch data
            trades_df, quotes_df = await fetcher.fetch_symbol_data(date, symbol)
            
            # Compute metrics
            trades_metrics = self.engine.process_trades(trades_df)
            quotes_metrics = self.engine.process_quotes(quotes_df)
            
            # Generate alerts
            alerts = self.engine.compute_composite_alerts(trades_metrics, quotes_metrics)
            alerts['symbol'] = symbol
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return None
            
    async def _send_alerts(self, alerts: pd.DataFrame):
        """Send high-priority alerts"""
        for _, alert in alerts.iterrows():
            logger.warning(
                f"ALERT: {alert['symbol']} - Score: {alert['alert_score']:.2f}, "
                f"Jump: {alert['jump_detected']}, "
                f"Timestamp: {alert['timestamp']}"
            )
            
    def stop(self):
        """Stop monitoring"""
        self.is_running = False

# Example usage
async def main():
    config = SurpriseConfig(
        num_gpus=4,
        buffer_size=10_000_000,
        jump_threshold=4.5
    )
    
    monitor = RealTimeMonitor(
        api_key="YOUR_POLYGON_API_KEY",
        symbols=["AAPL", "GOOGL", "MSFT", "AMZN"],
        config=config
    )
    
    await monitor.start()

if __name__ == "__main__":
    asyncio.run(main())
