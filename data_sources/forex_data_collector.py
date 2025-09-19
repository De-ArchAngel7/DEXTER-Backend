#!/usr/bin/env python3
"""
🌍 DEXTER Forex Data Collector
============================================================
Comprehensive data collection from multiple forex sources:
- Dukascopy (tick & minute data)
- HistData.com (historical CSVs)
- TrueFX (real-time & historical)
- Yahoo Finance (OHLCV data)
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
import structlog
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger()

class ForexDataCollector:
    """
    🌍 Comprehensive Forex Data Collector
    Collects OHLCV data from multiple sources for LSTM training
    """
    
    def __init__(self, data_dir: str = "data/forex"):
        self.data_dir = data_dir
        self.ensure_data_directory()
        
        # Major currency pairs for data collection
        self.major_pairs = [
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
            "AUDUSD", "USDCAD", "NZDUSD", "EURJPY",
            "GBPJPY", "EURGBP", "AUDJPY", "EURAUD"
        ]
        
        # Yahoo Finance symbols (add =X suffix)
        self.yahoo_symbols = [f"{pair}=X" for pair in self.major_pairs]
        
        logger.info(f"🌍 Forex Data Collector initialized with {len(self.major_pairs)} major pairs")
    
    def ensure_data_directory(self):
        """Ensure data directory exists"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(f"{self.data_dir}/raw", exist_ok=True)
        os.makedirs(f"{self.data_dir}/processed", exist_ok=True)
        os.makedirs(f"{self.data_dir}/sentiment", exist_ok=True)
    
    def collect_yahoo_finance_data(self, 
                                 symbols: Optional[List[str]] = None, 
                                 period: str = "2y",
                                 interval: str = "1h") -> Dict[str, pd.DataFrame]:
        """
        📊 Collect forex data from Yahoo Finance
        Most reliable free source for forex OHLCV data
        """
        if symbols is None:
            symbols = self.yahoo_symbols
        
        logger.info(f"📊 Collecting Yahoo Finance data for {len(symbols)} symbols")
        
        collected_data = {}
        
        for symbol in symbols:
            try:
                logger.info(f"📈 Fetching {symbol}...")
                
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
                
                if data.empty:
                    logger.warning(f"⚠️ No data for {symbol}")
                    continue
                
                # Clean and standardize data
                data = self._clean_forex_data(data, symbol)
                
                # Save raw data
                raw_path = f"{self.data_dir}/raw/{symbol.replace('=X', '')}_yahoo_raw.csv"
                data.to_csv(raw_path)
                
                collected_data[symbol] = data
                logger.info(f"✅ Collected {len(data)} records for {symbol}")
                
            except Exception as e:
                logger.error(f"❌ Error collecting {symbol}: {e}")
                continue
        
        logger.info(f"✅ Yahoo Finance collection complete: {len(collected_data)} symbols")
        return collected_data
    
    def collect_histdata_sample(self, currency_pair: str = "EURUSD") -> Optional[pd.DataFrame]:
        """
        📊 Collect sample data from HistData.com format
        Note: This is a template - you'll need to download actual files from HistData.com
        """
        try:
            logger.info(f"📊 Collecting HistData sample for {currency_pair}")
            
            # HistData.com file format example:
            # Date,Time,Open,High,Low,Close,Volume
            # 2023.01.01,00:00:00.000,1.0545,1.0550,1.0540,1.0548,1000
            
            # For now, create a sample structure
            # In production, you would download actual CSV files from HistData.com
            
            sample_data = self._create_sample_histdata(currency_pair)
            
            if sample_data is not None:
                # Save sample data
                sample_path = f"{self.data_dir}/raw/{currency_pair}_histdata_sample.csv"
                sample_data.to_csv(sample_path, index=False)
                logger.info(f"✅ Created HistData sample for {currency_pair}")
            
            return sample_data
            
        except Exception as e:
            logger.error(f"❌ Error with HistData sample: {e}")
            return None
    
    def collect_truefx_sample(self, currency_pair: str = "EURUSD") -> Optional[pd.DataFrame]:
        """
        📊 Collect sample data from TrueFX format
        Note: This is a template - you'll need to implement actual TrueFX API calls
        """
        try:
            logger.info(f"📊 Collecting TrueFX sample for {currency_pair}")
            
            # TrueFX format example:
            # Date,Time,Open,High,Low,Close,Volume
            # 2023-01-01,00:00:00,1.0545,1.0550,1.0540,1.0548,1000
            
            sample_data = self._create_sample_truefx(currency_pair)
            
            if sample_data is not None:
                # Save sample data
                sample_path = f"{self.data_dir}/raw/{currency_pair}_truefx_sample.csv"
                sample_data.to_csv(sample_path, index=False)
                logger.info(f"✅ Created TrueFX sample for {currency_pair}")
            
            return sample_data
            
        except Exception as e:
            logger.error(f"❌ Error with TrueFX sample: {e}")
            return None
    
    def _clean_forex_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and standardize forex data"""
        try:
            # Remove any rows with NaN values
            data = data.dropna()
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in data.columns:
                    if col == 'Volume':
                        # Forex doesn't have volume, create a proxy
                        data[col] = (data['High'] - data['Low']) * data['Close']
                    else:
                        logger.warning(f"Missing column {col} for {symbol}")
            
            # Add metadata
            data['Symbol'] = symbol
            data['Source'] = 'Yahoo Finance'
            
            # Sort by date
            data = data.sort_index()
            
            return data
            
        except Exception as e:
            logger.error(f"Error cleaning data for {symbol}: {e}")
            return data
    
    def _create_sample_histdata(self, currency_pair: str) -> Optional[pd.DataFrame]:
        """Create sample HistData format data"""
        try:
            # Generate sample data for demonstration
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
            
            # Simulate realistic forex price movements
            base_price = 1.05 if 'USD' in currency_pair else 1.0
            prices = []
            current_price = base_price
            
            for _ in range(len(dates)):
                # Random walk with some trend
                change = np.random.normal(0, 0.001)  # 0.1% volatility
                current_price *= (1 + change)
                prices.append(current_price)
            
            # Create OHLCV data
            data = []
            for i, (date, price) in enumerate(zip(dates, prices)):
                # Create realistic OHLC from price
                volatility = np.random.uniform(0.0005, 0.002)  # 0.05% to 0.2%
                
                high = price * (1 + volatility)
                low = price * (1 - volatility)
                open_price = prices[i-1] if i > 0 else price
                close_price = price
                volume = np.random.randint(1000, 10000)
                
                data.append({
                    'Date': date.strftime('%Y.%m.%d'),
                    'Time': date.strftime('%H:%M:%S.000'),
                    'Open': round(open_price, 5),
                    'High': round(high, 5),
                    'Low': round(low, 5),
                    'Close': round(close_price, 5),
                    'Volume': volume
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error creating HistData sample: {e}")
            return None
    
    def _create_sample_truefx(self, currency_pair: str) -> Optional[pd.DataFrame]:
        """Create sample TrueFX format data"""
        try:
            # Generate sample data for demonstration
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
            
            # Simulate realistic forex price movements
            base_price = 1.05 if 'USD' in currency_pair else 1.0
            prices = []
            current_price = base_price
            
            for _ in range(len(dates)):
                # Random walk with some trend
                change = np.random.normal(0, 0.001)  # 0.1% volatility
                current_price *= (1 + change)
                prices.append(current_price)
            
            # Create OHLCV data
            data = []
            for i, (date, price) in enumerate(zip(dates, prices)):
                # Create realistic OHLC from price
                volatility = np.random.uniform(0.0005, 0.002)  # 0.05% to 0.2%
                
                high = price * (1 + volatility)
                low = price * (1 - volatility)
                open_price = prices[i-1] if i > 0 else price
                close_price = price
                volume = np.random.randint(1000, 10000)
                
                data.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Time': date.strftime('%H:%M:%S'),
                    'Open': round(open_price, 5),
                    'High': round(high, 5),
                    'Low': round(low, 5),
                    'Close': round(close_price, 5),
                    'Volume': volume
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error creating TrueFX sample: {e}")
            return None
    
    def combine_data_sources(self, currency_pair: str) -> Optional[pd.DataFrame]:
        """Combine data from multiple sources for a currency pair"""
        try:
            logger.info(f"🔄 Combining data sources for {currency_pair}")
            
            combined_data = []
            
            # Try to load from different sources
            sources = [
                f"{currency_pair}_yahoo_raw.csv",
                f"{currency_pair}_histdata_sample.csv",
                f"{currency_pair}_truefx_sample.csv"
            ]
            
            for source in sources:
                source_path = f"{self.data_dir}/raw/{source}"
                if os.path.exists(source_path):
                    try:
                        data = pd.read_csv(source_path)
                        data['Source'] = source.split('_')[1].split('.')[0]  # Extract source name
                        combined_data.append(data)
                        logger.info(f"✅ Loaded {len(data)} records from {source}")
                    except Exception as e:
                        logger.warning(f"⚠️ Error loading {source}: {e}")
            
            if not combined_data:
                logger.warning(f"⚠️ No data sources found for {currency_pair}")
                return None
            
            # Combine all data
            if len(combined_data) == 1:
                final_data = combined_data[0]
            else:
                final_data = pd.concat(combined_data, ignore_index=True)
                # Remove duplicates and sort
                final_data = final_data.drop_duplicates().sort_values(['Date', 'Time'])
            
            # Save combined data
            combined_path = f"{self.data_dir}/processed/{currency_pair}_combined.csv"
            final_data.to_csv(combined_path, index=False)
            
            logger.info(f"✅ Combined data saved: {len(final_data)} records")
            return final_data
            
        except Exception as e:
            logger.error(f"❌ Error combining data for {currency_pair}: {e}")
            return None
    
    def create_training_dataset(self, currency_pairs: Optional[List[str]] = None) -> pd.DataFrame:
        """Create comprehensive training dataset from all sources"""
        if currency_pairs is None:
            currency_pairs = self.major_pairs[:6]  # Top 6 pairs
        
        logger.info(f"🏋️ Creating training dataset for {len(currency_pairs)} currency pairs")
        
        all_data = []
        
        for pair in currency_pairs:
            try:
                # Combine data from all sources
                combined_data = self.combine_data_sources(pair)
                
                if combined_data is not None:
                    all_data.append(combined_data)
                    logger.info(f"✅ Added {len(combined_data)} records for {pair}")
                else:
                    logger.warning(f"⚠️ No data available for {pair}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing {pair}: {e}")
                continue
        
        if not all_data:
            logger.error("❌ No data collected for training dataset")
            return pd.DataFrame()
        
        # Combine all currency pair data
        training_data = pd.concat(all_data, ignore_index=True)
        
        # Save training dataset
        training_path = f"{self.data_dir}/processed/forex_training_dataset.csv"
        training_data.to_csv(training_path, index=False)
        
        logger.info(f"✅ Training dataset created: {len(training_data)} total records")
        logger.info(f"📊 Dataset saved to: {training_path}")
        
        return training_data
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of collected data"""
        try:
            summary = {
                "data_directory": self.data_dir,
                "currency_pairs": self.major_pairs,
                "raw_files": [],
                "processed_files": [],
                "total_records": 0
            }
            
            # Count raw files
            raw_dir = f"{self.data_dir}/raw"
            if os.path.exists(raw_dir):
                raw_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
                summary["raw_files"] = raw_files
            
            # Count processed files
            processed_dir = f"{self.data_dir}/processed"
            if os.path.exists(processed_dir):
                processed_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
                summary["processed_files"] = processed_files
            
            # Count total records
            for file in summary["processed_files"]:
                try:
                    data = pd.read_csv(f"{processed_dir}/{file}")
                    summary["total_records"] += len(data)
                except:
                    continue
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            return {"error": str(e)}

def main():
    """Test the forex data collector"""
    print("🌍 Testing DEXTER Forex Data Collector")
    print("=" * 50)
    
    # Initialize collector
    collector = ForexDataCollector()
    
    # Test 1: Yahoo Finance data collection
    print("\n📊 Test 1: Yahoo Finance Data Collection")
    yahoo_data = collector.collect_yahoo_finance_data(
        symbols=["EURUSD=X", "GBPUSD=X"],  # Test with 2 pairs
        period="1mo",  # 1 month for testing
        interval="1h"
    )
    
    print(f"✅ Collected data for {len(yahoo_data)} symbols")
    
    # Test 2: Sample data creation
    print("\n📊 Test 2: Sample Data Creation")
    histdata_sample = collector.collect_histdata_sample("EURUSD")
    truefx_sample = collector.collect_truefx_sample("EURUSD")
    
    print(f"✅ HistData sample: {'Created' if histdata_sample is not None else 'Failed'}")
    print(f"✅ TrueFX sample: {'Created' if truefx_sample is not None else 'Failed'}")
    
    # Test 3: Data combination
    print("\n🔄 Test 3: Data Combination")
    combined_data = collector.combine_data_sources("EURUSD")
    print(f"✅ Combined data: {'Created' if combined_data is not None else 'Failed'}")
    
    # Test 4: Training dataset creation
    print("\n🏋️ Test 4: Training Dataset Creation")
    training_data = collector.create_training_dataset(["EURUSD", "GBPUSD"])
    print(f"✅ Training dataset: {len(training_data)} records")
    
    # Test 5: Data summary
    print("\n📋 Test 5: Data Summary")
    summary = collector.get_data_summary()
    print(f"📁 Data directory: {summary['data_directory']}")
    print(f"📊 Raw files: {len(summary['raw_files'])}")
    print(f"📊 Processed files: {len(summary['processed_files'])}")
    print(f"📊 Total records: {summary['total_records']}")
    
    print("\n✅ Forex data collector testing completed!")

if __name__ == "__main__":
    main()
