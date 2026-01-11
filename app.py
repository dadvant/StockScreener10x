# Flask API Server for 10x Hunter - AI-Powered Stock Analysis
# Real-time stock screening with AI analysis and detailed fundamentals
# 
# SETUP:
# pip install flask flask-cors yfinance pandas numpy requests beautifulsoup4
# 
# RUN:
# python app.py
# 
# Then open: http://localhost:5000

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import requests
from bs4 import BeautifulSoup
import json
import os
import re
import sqlite3
import math

app = Flask(__name__)
CORS(app)

SETTINGS_FILE = 'settings.json'

# Load settings from file if exists
def load_settings():
    """Load settings from JSON file"""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading settings: {e}")
    return None

def save_settings(settings):
    """Save settings to JSON file"""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False

# Default settings
default_settings = {
    'conviction_threshold': 4.0,
    'rsi_min': 20,
    'rsi_max': 80,
    'volume_ratio_min': 0.8,
    'ma_min_period': 150,
    'ma_max_period': 200,
    'font_size': 1.0,
    'show_ma_20': True,
    'show_ma_50': True,
    'show_ma_optimal_trend': True,
    'ma_trend_min_period': 150,
    'ma_trend_max_period': 200
}

# Load saved settings or use defaults
saved_settings = load_settings()
if saved_settings:
    # Merge saved settings with defaults (in case new settings were added)
    default_settings.update(saved_settings)

screening_state = {
    'status': 'idle',
    'progress': 0,
    'results': [],
    'last_scan': None,
    'detailed_stocks': {},
    'settings': default_settings
}

# --- Utility: Normalize price DataFrame columns to expected lowercase names ---
def normalize_price_df(df):
    try:
        if df is None or len(df) == 0:
            return df
        # Flatten MultiIndex columns from yfinance if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(col[0]) for col in df.columns]
        # Lowercase and map common names
        col_map = {}
        for c in df.columns:
            lc = str(c).strip().lower().replace(' ', '_')
            # Map common variants
            if lc in ('open', 'high', 'low', 'close', 'volume'):
                col_map[c] = lc
            elif lc in ('adj_close', 'adjclose'):
                col_map[c] = 'adj_close'
            elif lc == 'vol':
                col_map[c] = 'volume'
            elif lc == 'closing_price':
                col_map[c] = 'close'
            else:
                # Keep lowercase by default
                col_map[c] = lc
        df = df.rename(columns=col_map)
        return df
    except Exception:
        return df

class StockDatabase:
    """Local SQLite database for caching stock data"""
    
    def __init__(self, db_path='stocks.db'):
        self.db_path = db_path
        self.init_db()
    
    def get_conn(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_db(self):
        """Initialize database tables"""
        conn = self.get_conn()
        c = conn.cursor()
        
        # Price data table
        c.execute('''CREATE TABLE IF NOT EXISTS price_data (
            id INTEGER PRIMARY KEY,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            UNIQUE(ticker, date)
        )''')
        
        # Fundamentals table
        c.execute('''CREATE TABLE IF NOT EXISTS fundamentals (
            ticker TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            last_updated TEXT NOT NULL
        )''')
        
        # News table
        c.execute('''CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY,
            ticker TEXT NOT NULL,
            title TEXT NOT NULL,
            source TEXT,
            link TEXT,
            published TEXT,
            date_stored TEXT,
            UNIQUE(ticker, title)
        )''')
        
        # Metadata table
        c.execute('''CREATE TABLE IF NOT EXISTS metadata (
            ticker TEXT PRIMARY KEY,
            last_price_update TEXT,
            last_fundamental_update TEXT,
            last_news_update TEXT
        )''')
        
        conn.commit()
        conn.close()
    
    def get_latest_price_date(self, ticker):
        """Get the latest date we have price data for"""
        try:
            conn = self.get_conn()
            c = conn.cursor()
            c.execute('SELECT MAX(date) as latest FROM price_data WHERE ticker = ?', (ticker,))
            result = c.fetchone()
            conn.close()
            return result['latest'] if result['latest'] else None
        except:
            return None
    
    def save_price_data(self, ticker, df):
        """Save price data to database"""
        try:
            conn = self.get_conn()
            c = conn.cursor()
            
            for idx, row in df.iterrows():
                # Convert timestamp to string if necessary
                if hasattr(idx, 'strftime'):
                    date_str = idx.strftime('%Y-%m-%d')
                else:
                    date_str = str(idx).split(' ')[0]
                
                # Convert all values to Python floats
                open_val = float(row['Open']) if 'Open' in row else None
                high_val = float(row['High']) if 'High' in row else None
                low_val = float(row['Low']) if 'Low' in row else None
                close_val = float(row['Close']) if 'Close' in row else None
                volume_val = float(row['Volume']) if 'Volume' in row else None
                
                c.execute('''INSERT OR REPLACE INTO price_data 
                    (ticker, date, open, high, low, close, volume) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (ticker, date_str, open_val, high_val, low_val, close_val, volume_val))
            
            # Update metadata
            c.execute('''INSERT OR REPLACE INTO metadata (ticker, last_price_update) 
                VALUES (?, ?)''', (ticker, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving price data: {e}")
    
    def get_price_data(self, ticker, days=1095):
        """Get price data from database"""
        try:
            conn = self.get_conn()
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            c = conn.cursor()
            c.execute('SELECT date, open, high, low, close, volume FROM price_data WHERE ticker = ? AND date >= ? ORDER BY date',
                     (ticker, cutoff_date))
            rows = c.fetchall()
            conn.close()
            
            if rows:
                # Convert sqlite3.Row objects to dictionaries
                data = [dict(row) for row in rows]
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                return df
            return None
        except Exception as e:
            print(f"❌ get_price_data error for {ticker}: {e}")
            return None
    
    def save_fundamentals(self, ticker, fundamentals):
        """Save fundamental data to database"""
        try:
            conn = self.get_conn()
            c = conn.cursor()
            c.execute('''INSERT OR REPLACE INTO fundamentals (ticker, data, last_updated) 
                VALUES (?, ?, ?)''',
                (ticker, json.dumps(fundamentals), datetime.now().isoformat()))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving fundamentals: {e}")
    
    def get_fundamentals(self, ticker):
        """Get fundamentals from database"""
        try:
            conn = self.get_conn()
            c = conn.cursor()
            c.execute('SELECT data, last_updated FROM fundamentals WHERE ticker = ?', (ticker,))
            result = c.fetchone()
            conn.close()
            
            if result and result['data']:
                data = json.loads(result['data'])
                # Check if older than 30 days
                last_updated = datetime.fromisoformat(result['last_updated'])
                if (datetime.now() - last_updated).days < 30:
                    return data, True  # Fresh data
                return data, False  # Stale data
            return None, False
        except:
            return None, False
    
    def save_news(self, ticker, articles):
        """Save news articles to database"""
        try:
            conn = self.get_conn()
            c = conn.cursor()
            
            for article in articles:
                c.execute('''INSERT OR IGNORE INTO news 
                    (ticker, title, source, link, published, date_stored) 
                    VALUES (?, ?, ?, ?, ?, ?)''',
                    (ticker, article.get('title'), article.get('source'), 
                     article.get('link'), article.get('published'), datetime.now().isoformat()))
            
            # Update metadata
            c.execute('''INSERT OR REPLACE INTO metadata (ticker, last_news_update) 
                VALUES (?, ?)''', (ticker, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving news: {e}")
    
    def get_news(self, ticker, limit=5):
        """Get news from database"""
        try:
            conn = self.get_conn()
            c = conn.cursor()
            c.execute('''SELECT title, source, link, published FROM news 
                WHERE ticker = ? ORDER BY date_stored DESC LIMIT ?''',
                (ticker, limit))
            rows = c.fetchall()
            conn.close()
            
            return [dict(row) for row in rows] if rows else []
        except:
            return []

db = StockDatabase()


# (moved) warm_cache_for_tickers is defined earlier near DB init


def clean_nan_values(obj):
    """Recursively replace NaN/Infinity (including numpy types) with None so JSON is valid."""
    # Handle numpy types
    try:
        import numpy as _np
    except Exception:
        _np = None

    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nan_values(v) for v in obj]

    # numpy scalar types
    if _np is not None and isinstance(obj, (_np.floating, _np.integer)):
        try:
            if _np.isfinite(obj):
                return float(obj) if isinstance(obj, _np.floating) else int(obj)
            return None
        except Exception:
            return None

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    return obj

class TenXHunter:
    def __init__(self, capital=50000, risk_tolerance=8):
        self.capital = capital
        self.risk_tolerance = risk_tolerance

    def get_universe(self, size=2000):
        universes = {
            500: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'BRK.B', 'JNJ', 'V', 'WMT',
                  'JPM', 'MA', 'PG', 'COST', 'MCD', 'ABT', 'HD', 'NFLX', 'CRM', 'ACN',
                  'PEP', 'LLY', 'NKE', 'ADBE', 'IBM', 'INTC', 'AMD', 'QCOM', 'TXN', 'ASML'] * 17,
            2000: (['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'BRK.B', 'JNJ', 'V', 'WMT',
                    'JPM', 'MA', 'PG', 'COST', 'MCD', 'ABT', 'HD', 'NFLX', 'CRM', 'ACN',
                    'PEP', 'LLY', 'NKE', 'ADBE', 'IBM', 'INTC', 'AMD', 'QCOM', 'TXN', 'ASML'] * 17 +
                   ['ASTS', 'OKLO', 'SMR', 'RKLB', 'GSIT', 'PLTR', 'CRWD', 'SNOW', 'ZS', 'UPST',
                    'AI', 'COIN', 'MSTR', 'ABNB', 'DASH', 'UBER', 'BIRD', 'LCID', 'RIVN', 'SOFI'] * 5 +
                   ['BLZE', 'VERV', 'RVMD', 'MARA', 'RIOT', 'CLSK', 'IREN', 'WEJO', 'BGRY', 'VXUS'] * 20),
            5000: (['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'BRK.B', 'JNJ', 'V', 'WMT',
                    'JPM', 'MA', 'PG', 'COST', 'MCD', 'ABT', 'HD', 'NFLX', 'CRM', 'ACN'] * 25 +
                   ['ASTS', 'OKLO', 'SMR', 'RKLB', 'GSIT', 'PLTR', 'CRWD', 'SNOW', 'ZS', 'UPST'] * 25 +
                   ['QQQ', 'XLK', 'VGT', 'CIBR', 'ARKK', 'ARKW', 'ARKQ', 'SARK', 'WEBL', 'SMH'] * 50)
        }
        tickers = list(set(universes.get(size, universes[2000])))[:size]
        return tickers

    def optimize_ma(self, ticker):
        try:
            # Try to get from database first
            data = db.get_price_data(ticker, days=1095)
            data = normalize_price_df(data)
            
            # If not in DB or stale, fetch from yfinance
            if data is None or len(data) < 200:
                latest_date = db.get_latest_price_date(ticker)
                if latest_date:
                    # Fetch only new data since last update
                    new_data = yf.download(ticker, start=latest_date, progress=False, auto_adjust=False)
                    if not new_data.empty:
                        if isinstance(new_data.columns, pd.MultiIndex):
                            new_data.columns = [col[0] for col in new_data.columns]
                        db.save_price_data(ticker, new_data)
                else:
                    # First time - fetch 5 years
                    new_data = yf.download(ticker, period='5y', progress=False, auto_adjust=False)
                    if not new_data.empty:
                        if isinstance(new_data.columns, pd.MultiIndex):
                            new_data.columns = [col[0] for col in new_data.columns]
                        db.save_price_data(ticker, new_data)
                
                # Get the data again from DB
                data = db.get_price_data(ticker, days=1095)
            
            if data is None or len(data) < 200:
                return 172, 5.0
            
            prices = data['close'].values
            best_score = -np.inf
            best_ma = 172
            
            for ma_days in range(150, 201, 5):
                sma = pd.Series(prices).rolling(window=ma_days).mean().values
                
                last_price = prices[-1]
                last_ma = sma[-1]
                if pd.isna(last_ma):
                    continue
                
                above_ma = 1 if last_price > last_ma else -1
                distance = abs(last_price - last_ma) / last_ma
                
                valid_ma = [m for m in sma[-252:] if pd.notna(m)]
                if valid_ma:
                    win_count = sum(1 for p, m in zip(prices[-252:], sma[-252:]) 
                                   if pd.notna(m) and p > m)
                    win_rate = win_count / len(valid_ma)
                else:
                    win_rate = 0.5
                
                score = (above_ma * distance * 2) + (win_rate * 3)
                
                if score > best_score:
                    best_score = score
                    best_ma = ma_days
            
            return best_ma, min(max(best_score, 0), 10)
        except:
            return 172, 5.0

    def get_fundamentals(self, ticker):
        """Fetch fundamental data for a stock"""
        try:
            # Check database first
            cached_fund, is_fresh = db.get_fundamentals(ticker)
            if cached_fund and is_fresh:
                return cached_fund
            
            # If not fresh or not in DB, fetch new data
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
            except Exception as e:
                print(f"Warning: Failed to fetch yfinance data for {ticker}: {e}")
                # Return cached data even if stale, or default structure
                if cached_fund:
                    return cached_fund
                # Return default structure if no cached data
                return {
                    'pe_ratio': 'N/A',
                    'pb_ratio': 'N/A',
                    'dividend_yield': 'N/A',
                    'market_cap': 'N/A',
                    'revenue': 'N/A',
                    'profit_margin': 'N/A',
                    'roa': 'N/A',
                    'roe': 'N/A',
                    'debt_to_equity': 'N/A',
                    'current_ratio': 'N/A',
                    'sector': 'N/A',
                    'industry': 'N/A',
                    'company_name': ticker,
                    '52_week_high': 'N/A',
                    '52_week_low': 'N/A',
                }
            
            fundamentals = {
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'pb_ratio': info.get('priceToBook', 'N/A'),
                'dividend_yield': info.get('dividendYield', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'revenue': info.get('totalRevenue', 'N/A'),
                'profit_margin': info.get('profitMargins', 'N/A'),
                'roa': info.get('returnOnAssets', 'N/A'),
                'roe': info.get('returnOnEquity', 'N/A'),
                'debt_to_equity': info.get('debtToEquity', 'N/A'),
                'current_ratio': info.get('currentRatio', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'company_name': info.get('longName', ticker),
                '52_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
                '52_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
            }
            
            # Save to database
            db.save_fundamentals(ticker, fundamentals)
            return fundamentals
        except:
            # Return cached data even if stale
            cached_fund, _ = db.get_fundamentals(ticker)
            return cached_fund if cached_fund else {}

    def get_news(self, ticker, limit=5):
        """Fetch recent news for a stock"""
        try:
            # Check database first
            cached_news = db.get_news(ticker, limit)
            if cached_news and len(cached_news) > 0:
                return cached_news
            
            # If not in DB, fetch new news from yfinance
            articles = []
            try:
                stock = yf.Ticker(ticker)
                news = stock.news
                if news and isinstance(news, list):
                    for item in news[:limit]:
                        title = item.get('title', '').strip()
                        source = item.get('source', 'Unknown')
                        link = item.get('link', '') or ''
                        published = item.get('providerPublishTime', '')
                        # Only add if we have a title
                        if title:
                            articles.append({
                                'title': title,
                                'source': source,
                                'link': link,
                                'published': published
                            })
            except Exception as e:
                pass  # Silently fail and try cached
            
            # Save to database if we got fresh articles
            if articles:
                db.save_news(ticker, articles)
                return articles
            
            # Fallback to cached
            return db.get_news(ticker, limit)
        except Exception as e:
            # Last resort: return empty list
            return []

    def ai_analysis(self, ticker, data_dict):
        """Generate AI-based analysis for 10x potential"""
        # Using a simple rule-based system for now (can be replaced with OpenAI API)
        analysis = {
            'ticker': ticker,
            'ai_reasoning': self._generate_reasoning(data_dict),
            'ten_x_score': self._calculate_ten_x_score(data_dict),
            'bull_case': self._generate_bull_case(data_dict),
            'bear_case': self._generate_bear_case(data_dict),
            'catalysts': self._identify_catalysts(data_dict)
        }
        return analysis

    def _generate_reasoning(self, data):
        """Generate AI reasoning for investment potential"""
        reasons = []
        
        if data.get('conviction', 0) >= 8.5:
            reasons.append(f"Strong technical setup with conviction score of {data['conviction']:.1f}")
        
        if data.get('volume_ratio', 0) > 2.0:
            reasons.append(f"Exceptional volume surge ({data['volume_ratio']:.1f}x average)")
        
        if data.get('rsi', 0) > 50 and data.get('rsi', 0) < 70:
            reasons.append(f"Momentum building with RSI at {data['rsi']:.0f} (not overbought)")
        
        if data.get('price', 0) > data.get('ma_value', 1):
            percent_above = ((data['price'] / data['ma_value']) - 1) * 100
            reasons.append(f"Price {percent_above:.1f}% above optimal MA - Strong uptrend")
        
        return reasons if reasons else ["Stock shows technical promise for potential gains"]

    def _calculate_ten_x_score(self, data):
        """Score potential for 10x returns (0-10 scale)"""
        score = 0
        
        # Technical score (0-4 points)
        if data.get('conviction', 0) >= 7:
            score += min(4, (data['conviction'] / 10) * 4)
        
        # Volume score (0-2 points)
        if data.get('volume_ratio', 0) > 1.5:
            score += min(2, (min(data['volume_ratio'], 4) / 4) * 2)
        
        # Price action score (0-2 points)
        if data.get('price', 0) > data.get('ma_value', 1):
            percent_above = ((data['price'] / data['ma_value']) - 1) * 100
            score += min(2, (min(percent_above, 20) / 20) * 2)
        
        # RSI momentum score (0-2 points)
        rsi = data.get('rsi', 50)
        if 40 < rsi < 70:
            score += 2 * ((70 - rsi) / 30)
        
        return min(10, score)

    def _generate_bull_case(self, data):
        """Generate bull case for the stock"""
        return [
            f"Technical breakout confirmed with price above {data.get('optimal_ma', 0)}-day MA",
            "Volume surge indicates institutional accumulation",
            "RSI suggests room for further upside before overbought conditions",
            "Early stage of potential trend with high conviction score"
        ]

    def _generate_bear_case(self, data):
        """Generate bear case for the stock"""
        return [
            "Early-stage accumulation means volatility and risk of pullbacks",
            "May face resistance at recent highs",
            "Broader market corrections could pressure stock",
            "Position sizing crucial due to micro-cap/small-cap nature"
        ]

    def _identify_catalysts(self, data):
        """Identify potential catalysts for 10x move"""
        return [
            "Earnings beat/raise could accelerate upside",
            "Product/service breakthrough or announcement",
            "Industry tailwinds or sector rotation",
            "M&A activity or strategic partnerships",
            "Insider accumulation signals confidence"
        ]

    def screen_stock(self, ticker):
        """
        =========================================
        SCREENING ALGORITHM - MOMENTUM + OPTIMAL MA
        =========================================
        1. Get 5 years of price data
        2. Find OPTIMAL MA (150-200) for THIS stock
        3. FILTER: Price MUST be above optimal MA
        4. Calculate recent gains (1m, 3m, 6m, 1y)
        5. Calculate conviction based on momentum
        6. Return if conviction >= threshold + return optimal MA used
        =========================================
        """
        try:
            # STEP 1: Get price data (5 years for better MA calculation)
            data = db.get_price_data(ticker, days=1825)
            
            if data is None or len(data) < 200:
                latest_date = db.get_latest_price_date(ticker)
                if latest_date:
                    new_data = yf.download(ticker, start=latest_date, progress=False, auto_adjust=False)
                else:
                    new_data = yf.download(ticker, period='3y', progress=False, auto_adjust=False)
                
                if new_data.empty or len(new_data) < 200:
                    return None
                
                # Handle MultiIndex columns from yfinance
                if isinstance(new_data.columns, pd.MultiIndex):
                    # Single ticker MultiIndex: flatten to simple columns
                    new_data.columns = [col[0] for col in new_data.columns]
                # Normalize columns to lowercase expected names
                new_data = normalize_price_df(new_data)
                db.save_price_data(ticker, new_data)
                data = db.get_price_data(ticker, days=1095)
                data = normalize_price_df(data)
            
            if data is None or len(data) < 200:
                return None
            
            data = normalize_price_df(data)
            prices = data['close'].values
            current_price = prices[-1]
            
            # STEP 2: Find OPTIMAL MA (150-200) using residual error minimization
            def _optimal_ma_by_residual(series: np.ndarray, min_p: int = 150, max_p: int = 200, lookback: int = 120):
                best_period = None
                best_rmse = None
                best_value = None
                s = pd.Series(series)
                for p in range(min_p, max_p + 1):
                    if len(series) < p:
                        continue
                    sma = s.rolling(window=p).mean().values
                    # Use last `lookback` days for residuals (skip NaNs)
                    tail_prices = series[-lookback:]
                    tail_sma = sma[-lookback:]
                    mask = ~pd.isna(tail_sma)
                    if not mask.any():
                        continue
                    residuals = tail_prices[mask] - tail_sma[mask]
                    rmse = float(np.sqrt(np.mean(residuals**2))) if residuals.size else None
                    if rmse is None:
                        continue
                    if best_rmse is None or rmse < best_rmse:
                        best_rmse = rmse
                        best_period = p
                        best_value = float(s.rolling(window=p).mean().values[-1])
                return best_period, best_value, best_rmse

            optimal_ma, ma_value, _rmse = _optimal_ma_by_residual(prices)
            if optimal_ma is None or ma_value is None:
                return None
            
            # STEP 3: HARD FILTER - Price MUST be above optimal MA
            if current_price <= ma_value:
                return None
            
            # How far above the optimal MA
            pct_above_ma = ((current_price - ma_value) / ma_value) * 100
            
            # STEP 4: Calculate recent gains
            price_1m_ago = prices[-21] if len(prices) > 21 else prices[0]
            price_3m_ago = prices[-63] if len(prices) > 63 else prices[0]
            price_6m_ago = prices[-126] if len(prices) > 126 else prices[0]
            price_1y_ago = prices[0] if len(prices) > 0 else current_price
            
            gain_1m = ((current_price - price_1m_ago) / price_1m_ago) * 100 if price_1m_ago > 0 else 0
            gain_3m = ((current_price - price_3m_ago) / price_3m_ago) * 100 if price_3m_ago > 0 else 0
            gain_6m = ((current_price - price_6m_ago) / price_6m_ago) * 100 if price_6m_ago > 0 else 0
            gain_1y = ((current_price - price_1y_ago) / price_1y_ago) * 100 if price_1y_ago > 0 else 0
            
            settings = screening_state['settings']
            
            # STEP 5: Calculate conviction based on momentum + MA distance sweet spots
            conviction = 0.0
            breakdown = {}

            # Helper: sweet-spot score for MA distance (0..1)
            def _sweet_spot_score(pct: float, spot_min: float, spot_max: float, lo: float, hi: float) -> float:
                # Outside tolerance bounds -> 0
                if pct <= lo or pct >= hi:
                    return 0.0
                # Rise from lo to spot_min
                if pct < spot_min:
                    return max(0.0, (pct - lo) / (spot_min - lo))
                # Peak plateau inside [spot_min, spot_max]
                if spot_min <= pct <= spot_max:
                    return 1.0
                # Fall from spot_max to hi
                return max(0.0, (hi - pct) / (hi - spot_max))

            # Compute MA20/50 distances (last values)
            s_prices = pd.Series(prices)
            ma20_last = float(s_prices.rolling(window=20).mean().values[-1]) if len(prices) >= 20 else None
            ma50_last = float(s_prices.rolling(window=50).mean().values[-1]) if len(prices) >= 50 else None
            pct_above_ma20 = ((current_price - ma20_last) / ma20_last) * 100 if ma20_last and ma20_last > 0 else None
            pct_above_ma50 = ((current_price - ma50_last) / ma50_last) * 100 if ma50_last and ma50_last > 0 else None

            # Base points for being above optimal MA
            breakdown['base_above_ma'] = 2.0
            conviction += breakdown['base_above_ma']

            # Composite MA-distance score (up to 1 + 1 + 2 = 4 points)
            # Sweet spots (heuristic, can later be replaced by learned histograms):
            # MA20: 3%-7%, MA50: 2%-6%, MA150/200: 1%-5% (double weight)
            score20 = _sweet_spot_score(pct_above_ma20 or 0.0, 3.0, 7.0, -5.0, 12.0)
            score50 = _sweet_spot_score(pct_above_ma50 or 0.0, 2.0, 6.0, -5.0, 10.0)
            scoreTrend = _sweet_spot_score(pct_above_ma, 1.0, 5.0, -4.0, 9.0)
            breakdown['ma_distance_score_20'] = float(score20)
            breakdown['ma_distance_score_50'] = float(score50)
            breakdown['ma_distance_score_trend'] = float(scoreTrend)
            breakdown['ma_distance_composite'] = float(score20 + score50 + 2.0 * scoreTrend)
            conviction += breakdown['ma_distance_composite']
            
            # Recent momentum components
            breakdown['momentum_1m'] = float(min(1.5, gain_1m / 10)) if gain_1m > 0 else 0.0
            conviction += breakdown['momentum_1m']
            breakdown['momentum_3m'] = 1.0 if gain_3m > 5 else 0.0
            conviction += breakdown['momentum_3m']
            breakdown['momentum_6m'] = 1.5 if gain_6m > 20 else 0.0
            conviction += breakdown['momentum_6m']
            breakdown['momentum_1y'] = 2.0 if gain_1y > 50 else 0.0
            conviction += breakdown['momentum_1y']
            
            # Bonus for strong recent momentum (last month)
            breakdown['bonus_1m'] = 1.0 if gain_1m > 15 else 0.0
            conviction += breakdown['bonus_1m']

            # News/Event scoring (sentiment + upcoming catalysts)
            def _news_event_score(tkr: str):
                try:
                    articles = self.get_news(tkr, limit=10) or []
                    pos_kw = ['beat', 'record', 'contract', 'deal', 'partnership', 'award', 'approval', 'fda approval', 'upgrade', 'raised', 'surge', 'strong', 'profit', 'revenue growth', 'expansion', 'backlog', 'milestone', 'buyback']
                    neg_kw = ['downgrade', 'miss', 'delay', 'investigation', 'sec', 'ftc', 'antitrust', 'recall', 'layoff', 'cut', 'guidance cut', 'short report', 'fraud', 'lawsuit', 'terminated', 'reject', 'fda rejection', 'halt']

                    pos_hits = 0
                    neg_hits = 0
                    found_catalysts = []
                    for a in articles:
                        title = (a.get('title') or '').lower()
                        if not title:
                            continue
                        if any(k in title for k in pos_kw):
                            pos_hits += 1
                            if 'contract' in title and 'New contract win' not in found_catalysts:
                                found_catalysts.append('New contract win')
                            elif ('partnership' in title or 'deal' in title) and 'Strategic partnership announced' not in found_catalysts:
                                found_catalysts.append('Strategic partnership announced')
                            elif 'upgrade' in title and 'Analyst upgrade' not in found_catalysts:
                                found_catalysts.append('Analyst upgrade')
                            elif 'approval' in title and 'Regulatory approval milestone' not in found_catalysts:
                                found_catalysts.append('Regulatory approval milestone')
                            elif 'buyback' in title and 'Share buyback' not in found_catalysts:
                                found_catalysts.append('Share buyback')
                        if any(k in title for k in neg_kw):
                            neg_hits += 1

                    # base sentiment score clipped
                    base = pos_hits * 0.25 - neg_hits * 0.30
                    base = max(-1.5, min(1.5, base))

                    # upcoming earnings bonus if within ~10 days (lightweight check)
                    earn_bonus = 0.0
                    try:
                        stock = yf.Ticker(tkr)
                        info = stock.info or {}
                        ed = info.get('earningsDate')
                        if isinstance(ed, (list, tuple)) and len(ed) > 0:
                            earn_date = ed[0]
                            try:
                                if hasattr(earn_date, 'to_pydatetime'):
                                    earn_dt = earn_date.to_pydatetime()
                                else:
                                    earn_dt = pd.to_datetime(earn_date).to_pydatetime()
                                days_to = (earn_dt - datetime.now()).days
                                if 0 <= days_to <= 3:
                                    earn_bonus = 0.6
                                    if 'Earnings in 1-3 days' not in found_catalysts:
                                        found_catalysts.append('Earnings in 1-3 days')
                                elif 0 < days_to <= 10:
                                    earn_bonus = 0.4
                                    if 'Earnings upcoming' not in found_catalysts:
                                        found_catalysts.append('Earnings upcoming')
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # modest bonus for government/contract related wins
                    contract_bonus = 0.3 if any('contract' in (a.get('title','').lower()) for a in articles) else 0.0

                    total = base + earn_bonus + contract_bonus
                    total = max(-2.0, min(2.0, total))

                    return float(round(total, 2)), found_catalysts[:3]
                except Exception:
                    return 0.0, []

            score, catalysts = _news_event_score(ticker)
            breakdown['news_event'] = score
            conviction += score
            
            conviction = float(max(0, min(10, conviction)))
            breakdown['total'] = conviction
            
            # STEP 6: Check threshold
            if conviction < settings['conviction_threshold']:
                return None
            
            # STEP 7: Get company info (sector, description)
            sector = 'Technology'
            description = ''
            company_name = ticker
            
            try:
                # Quick fetch of company info
                stock = yf.Ticker(ticker)
                info = stock.info
                sector = info.get('sector', 'Technology')
                description = info.get('longBusinessSummary', '')
                company_name = info.get('longName', ticker)
                
                # Shorten description to one sentence
                if description:
                    # Take first sentence or first 150 chars
                    first_period = description.find('.')
                    if first_period > 0 and first_period < 200:
                        description = description[:first_period + 1]
                    else:
                        description = description[:150] + '...' if len(description) > 150 else description
            except:
                pass

            # Build result dict
            result = {
                'ticker': ticker,
                'conviction': conviction,
                'price': float(current_price),
                'optimal_ma': int(optimal_ma),
                'ma_value': float(ma_value),
                'pct_above_ma': float(pct_above_ma),
                'pct_above_ma20': float(pct_above_ma20) if pct_above_ma20 is not None else None,
                'pct_above_ma50': float(pct_above_ma50) if pct_above_ma50 is not None else None,
                'gain_1m': float(gain_1m),
                'gain_3m': float(gain_3m),
                'gain_6m': float(gain_6m),
                'gain_1y': float(gain_1y),
                'signal': 'BUY' if conviction >= 7.0 else 'STRONG BUY' if conviction >= 5.5 else 'ACCUMULATE',
                
                # Trading recommendations
                'stop_loss': float(current_price * 0.92),  # 8% stop loss
                'take_profit': float(current_price * (1 + (conviction / 10) * 1.0)),  # More realistic target
                'potential_x': round((conviction / 10) * 10, 1),  # 10x for perfect 10/10, scaled down
                'position_size': int(5000 if conviction >= 8.0 else 3000 if conviction >= 6.0 else 2000),
                
                # Company info
                'sector': sector,
                'description': description,
                'company_name': company_name,
                'catalysts': catalysts,
                'score_breakdown': breakdown

            }

            # Classify candidate: momentum vs potential 10x (early-trend)
            # Simple heuristic: high recent gains -> momentum; strong MA-distance composite with moderate recent gains -> 10x
            momentum_score = float(min(10.0, (max(0.0, gain_1m) / 10.0) * 3 + (max(0.0, gain_3m) / 10.0) * 2 + (max(0.0, gain_6m) / 20.0) * 3))
            tenx_score = float(min(10.0, (breakdown['ma_distance_composite'] * 2.0) + (max(0.0, gain_1y) / 50.0) * 2 + breakdown.get('news_event', 0)))
            result['momentum_score'] = momentum_score
            result['ten_x_score'] = tenx_score
            result['is_momentum'] = momentum_score >= 6.0
            result['is_ten_x'] = tenx_score >= 6.0

            # Add AI analysis / reasoning for card summary (short form)
            try:
                ai_input = {
                    'conviction': conviction,
                    'volume_ratio': 1.0,
                    'rsi': None,
                    'price': float(current_price),
                    'ma_value': float(ma_value),
                    'optimal_ma': int(optimal_ma)
                }
                ai = self.ai_analysis(ticker, ai_input)
                if isinstance(ai, dict):
                    result['ai_reasoning'] = ai.get('ai_reasoning', [])
                    result['ten_x_score'] = ai.get('ten_x_score', None)
                else:
                    result['ai_reasoning'] = []
                    result['ten_x_score'] = None
            except Exception:
                result['ai_reasoning'] = []
                result['ten_x_score'] = None

            return result
        except Exception as e:
            return None

    def get_detailed_analysis(self, ticker):
        """Get comprehensive analysis for a specific stock with technical levels"""
        global screening_state
        
        try:
            # Basic screening data
            screen_data = self.screen_stock(ticker)
            if not screen_data:
                screen_data = {
                    'ticker': ticker,
                    'price': 0,
                    'conviction': 0,
                    'optimal_ma': 200,
                    'ma_value': 0,
                    'pct_above_ma': 0,
                    'gain_1m': 0,
                    'gain_3m': 0,
                    'gain_6m': 0,
                    'gain_1y': 0,
                    'signal': 'HOLD'
                }
            
            # Get 5 years of historical data for chart (for wider context and MA)
            data = db.get_price_data(ticker, days=1825)
            
            if data is None:
                latest_date = db.get_latest_price_date(ticker)
                if latest_date:
                    new_data = yf.download(ticker, start=latest_date, progress=False, auto_adjust=False)
                else:
                    new_data = yf.download(ticker, period='3y', progress=False, auto_adjust=False)
                
                if not new_data.empty:
                    if isinstance(new_data.columns, pd.MultiIndex):
                        new_data.columns = [col[0] for col in new_data.columns]
                    db.save_price_data(ticker, new_data)
                    data = db.get_price_data(ticker, days=1095)
            
            # Calculate technical indicators and levels
            technical_analysis = {}
            ma_50 = []  # Initialize to avoid undefined reference
            ma_200 = []  # Initialize to avoid undefined reference
            if data is not None and len(data) > 0:
                data = normalize_price_df(data)
                prices = data['close'].values
                highs = data['high'].values
                lows = data['low'].values
                current_price = prices[-1]
                
                # Moving Averages - Calculate all required MAs
                ma_20 = pd.Series(prices).rolling(window=20).mean().values
                ma_50 = pd.Series(prices).rolling(window=50).mean().values
                ma_150 = pd.Series(prices).rolling(window=150).mean().values if len(prices) >= 150 else None
                ma_200 = pd.Series(prices).rolling(window=200).mean().values if len(prices) >= 200 else None
                
                # Calculate optimal trend MA (150-200) based on smoothness/crossovers in recent data
                optimal_trend_ma = None
                optimal_trend_period = None
                if len(prices) >= 200:
                    # Find the MA between 150-200 that best tracks the current trend
                    # Metric: lowest standard deviation of price deviations from MA in last 60 days
                    best_score = float('inf')
                    best_period = 150
                    
                    for period in range(150, 201, 5):  # Check every 5-day interval
                        ma_values = pd.Series(prices).rolling(window=period).mean().values
                        # Use last 60 days to evaluate
                        recent_prices = prices[-60:]
                        recent_ma = ma_values[-60:]
                        if len(recent_ma) > 0 and recent_ma[-1] is not None:
                            # Standard deviation of deviations from MA
                            deviations = recent_prices - recent_ma
                            score = np.std(deviations[~np.isnan(deviations)])
                            if score < best_score:
                                best_score = score
                                best_period = period
                    
                    optimal_trend_period = best_period
                    optimal_trend_ma = pd.Series(prices).rolling(window=best_period).mean().values
                
                # RSI
                delta = pd.Series(prices).diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = -delta.where(delta < 0, 0).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
                
                # Support and Resistance Levels
                # Find recent swing highs/lows
                window = 20
                support_levels = []
                resistance_levels = []
                
                for i in range(window, len(prices) - window):
                    # Check if it's a local minimum (support)
                    if lows[i] == min(lows[i-window:i+window]):
                        support_levels.append(float(lows[i]))
                    # Check if it's a local maximum (resistance)
                    if highs[i] == max(highs[i-window:i+window]):
                        resistance_levels.append(float(highs[i]))
                
                # Cluster nearby levels
                def cluster_levels(levels, tolerance=0.02):
                    if not levels:
                        return []
                    levels = sorted(levels)
                    clustered = []
                    current_cluster = [levels[0]]
                    
                    for level in levels[1:]:
                        if (level - current_cluster[-1]) / current_cluster[-1] < tolerance:
                            current_cluster.append(level)
                        else:
                            clustered.append(np.mean(current_cluster))
                            current_cluster = [level]
                    clustered.append(np.mean(current_cluster))
                    return clustered
                
                support_levels = cluster_levels(support_levels)[-3:]  # Top 3 support
                resistance_levels = cluster_levels(resistance_levels)[-3:]  # Top 3 resistance
                
                # Filter to only relevant levels (near current price)
                support_levels = [s for s in support_levels if s < current_price and s > current_price * 0.7]
                resistance_levels = [r for r in resistance_levels if r > current_price and r < current_price * 1.3]
                
                technical_analysis = {
                    'current_price': float(current_price),
                    'ma_50': float(ma_50[-1]) if len(ma_50) > 0 else None,
                    'ma_150': float(ma_150[-1]) if len(ma_150) > 0 else None,
                    'ma_200': float(ma_200[-1]) if ma_200 is not None and len(ma_200) > 0 else None,
                    'rsi': float(current_rsi),
                    'support_levels': support_levels,
                    'resistance_levels': resistance_levels,
                    '52w_high': float(max(prices)),
                    '52w_low': float(min(prices)),
                    'volatility': float(np.std(np.diff(np.log(prices))) * np.sqrt(252) * 100) if len(prices) > 1 else 0
                }
            
            # Get fundamentals
            fundamentals = self.get_fundamentals(ticker)
            if not fundamentals:
                fundamentals = {}
            
            # Get news
            news = self.get_news(ticker, limit=10)
            if not news:
                news = []
            
            # Get AI analysis
            ai_analysis = self.ai_analysis(ticker, screen_data)
            if not ai_analysis:
                ai_analysis = {}
            
            # Prepare chart data with OHLC for candlestick
            chart_data = {}
            if data is not None and len(data) > 0:
                chart_data = {
                    'dates': data.index.strftime('%Y-%m-%d').tolist(),
                    'open': data['open'].values.tolist(),
                    'high': data['high'].values.tolist(),
                    'low': data['low'].values.tolist(),
                    'close': data['close'].values.tolist(),
                    'prices': data['close'].values.tolist(),  # For line chart
                    'volumes': (data['volume'] / 1e6).values.tolist(),  # Millions
                    'ma_20': ma_20.tolist() if len(ma_20) > 0 else [],
                    'ma_50': ma_50.tolist() if len(ma_50) > 0 else [],
                    'ma_150': ma_150.tolist() if ma_150 is not None and len(ma_150) > 0 else [],
                    'ma_200': ma_200.tolist() if ma_200 is not None and len(ma_200) > 0 else [],
                    'ma_optimal_trend': optimal_trend_ma.tolist() if optimal_trend_ma is not None and len(optimal_trend_ma) > 0 else [],
                    'ma_optimal_trend_period': optimal_trend_period
                }
            
            detailed_analysis = {
                'ticker': ticker,
                'screen_data': screen_data,
                'technical_analysis': technical_analysis,
                'fundamentals': fundamentals,
                'news': news,
                'ai_analysis': ai_analysis,
                'chart_data': chart_data
            }
            
            screening_state['detailed_stocks'][ticker] = detailed_analysis
            return detailed_analysis
        except Exception as e:
            print(f"Error in get_detailed_analysis for {ticker}: {e}")
            import traceback
            traceback.print_exc()
            
            # Return minimal valid structure on error
            return {
                'ticker': ticker,
                'screen_data': {
                    'ticker': ticker,
                    'price': 0,
                    'conviction': 0,
                    'optimal_ma': 200,
                    'ma_value': 0,
                    'pct_above_ma': 0,
                    'gain_1m': 0,
                    'gain_3m': 0,
                    'gain_6m': 0,
                    'gain_1y': 0,
                    'signal': 'HOLD',
                    'stop_loss': 0,
                    'take_profit': 0,
                    'potential_x': 0,
                    'position_size': 0,
                    'sector': 'Unknown',
                    'description': 'Error loading data',
                    'catalysts': []
                },
                'technical_analysis': {},
                'fundamentals': {},
                'news': [],
                'ai_analysis': {'error': str(e)},
                'chart_data': {}
            }

    def run_screen(self, universe_size=2000):
        global screening_state
        screening_state['status'] = 'scanning'
        screening_state['progress'] = 0
        screening_state['results'] = []  # Reset results
        tickers = self.get_universe(universe_size)
        candidates = []
        total_tickers = len(tickers)
        
        print(f"\n{'='*60}")
        print(f"Starting screen of {total_tickers} stocks...")
        print(f"{'='*60}")
        
        for i, ticker in enumerate(tickers):
            try:
                # Update progress BEFORE processing (so it shows immediately)
                screening_state['progress'] = int(((i + 1) / total_tickers) * 100)
                
                result = self.screen_stock(ticker)
                if result:
                    print(f"✅ {ticker}: conviction={result['conviction']:.2f}, price=${result['price']:.2f}")
                    candidates.append(result)
                
                if i % 20 == 0:
                    time.sleep(0.1)
            except Exception as e:
                continue
        
        print(f"\n{'='*60}")
        print(f"Scan complete: Found {len(candidates)} candidates")
        print(f"{'='*60}")
        
        # Sort by conviction (descending) - NO LIMIT, user can filter
        candidates = sorted(candidates, key=lambda x: x['conviction'], reverse=True)
        
        screening_state['results'] = candidates
        screening_state['last_scan'] = datetime.now().isoformat()
        screening_state['status'] = 'complete'
        screening_state['progress'] = 100

        # Background warm cache for top results to accelerate detail loads
        try:
            top_tickers = [c['ticker'] for c in candidates[:100]]
            if top_tickers:
                func = globals().get('warm_cache_for_tickers')
                if func:
                    threading.Thread(target=func, args=(top_tickers,), daemon=True).start()
                    print(f"[WarmCache] Background warm started for top {len(top_tickers)} tickers")
        except Exception as e:
            print(f"[WarmCache] Unable to start background warm: {e}")

@app.route('/api/scan', methods=['POST'])
def start_scan():
    data = request.json
    universe_size = data.get('universe_size', 2000)
    capital = data.get('capital', 50000)
    risk_tolerance = data.get('risk_tolerance', 8)
    
    hunter = TenXHunter(capital, risk_tolerance)
    thread = threading.Thread(target=hunter.run_screen, args=(universe_size,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'scanning'})

@app.route('/api/warm_cache', methods=['POST'])
def warm_cache_api():
    """Trigger cache warm for tickers or top recent results.
    Body: { tickers?: ["AAPL",...], top_n?: 50 }
    If tickers missing, uses last screening results (top_n).
    """
    try:
        body = request.json or {}
        tickers = body.get('tickers')
        top_n = int(body.get('top_n', 50))
        if not tickers:
            tickers = [c['ticker'] for c in screening_state.get('results', [])[:top_n]]
        if not tickers:
            return jsonify({'ok': False, 'message': 'No tickers provided and no recent results'}), 400
        func = globals().get('warm_cache_for_tickers')
        if func:
            th = threading.Thread(target=func, args=(tickers,), daemon=True)
            th.start()
        return jsonify({'ok': True, 'started': len(tickers)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify(clean_nan_values(screening_state))

@app.route('/api/results', methods=['GET'])
def get_results():
    min_conviction = request.args.get('min_conviction', default=screening_state['settings']['conviction_threshold'], type=float)
    filtered = [c for c in screening_state['results'] if c['conviction'] >= min_conviction]
    resp = {
        'results': filtered,
        'count': len(filtered),
        'total': len(screening_state['results']),
        'min_conviction': min_conviction
    }
    return jsonify(clean_nan_values(resp))

@app.route('/api/stock/<ticker>', methods=['GET'])
def get_stock_detail(ticker):
    """Get detailed analysis for a specific stock"""
    hunter = TenXHunter()
    analysis = hunter.get_detailed_analysis(ticker.upper())

    # Clean NaN values before returning using shared helper
    cleaned_analysis = clean_nan_values(analysis)
    return jsonify(cleaned_analysis)

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get historical statistics from database for visualization"""
    try:
        conn = db.get_conn()
        c = conn.cursor()
        
        # Get min/max settings for MA period range
        settings = screening_state['settings']
        ma_min = settings.get('ma_min_period', 150)
        ma_max = settings.get('ma_max_period', 200)
        
        # Define index symbols
        sp500_symbols = get_sp500_symbols()
        nasdaq100_symbols = get_nasdaq100_symbols()
        
        # Query price data to calculate statistics
        c.execute('SELECT DISTINCT ticker FROM price_data')
        all_tickers = [row[0] for row in c.fetchall()]
        
        ma20_gains = {'sp500': [], 'nasdaq100': [], 'full5000': [], 'labels': []}
        ma50_gains = {'sp500': [], 'nasdaq100': [], 'full5000': [], 'labels': []}
        ma_period_counts = {}
        
        # Process each ticker
        for ticker in all_tickers[:500]:  # Limit to first 500 for performance
            try:
                # Categorize ticker
                is_sp500 = ticker in sp500_symbols
                is_nasdaq100 = ticker in nasdaq100_symbols
                
                # Get price data
                df = db.get_price_data(ticker, days=1095)
                if df is None or len(df) < 200:
                    continue
                
                prices = df['close'].values
                
                # Calculate MAs
                ma20 = pd.Series(prices).rolling(window=20).mean().values
                ma50 = pd.Series(prices).rolling(window=50).mean().values
                
                # Calculate % above each MA (last 30 days average)
                if len(prices) >= 30:
                    recent_idx = list(range(len(prices)-30, len(prices)))
                    pct_above_ma20 = np.nanmean([
                        ((prices[i] - ma20[i]) / ma20[i] * 100) if ma20[i] else 0 
                        for i in recent_idx if i < len(ma20) and ma20[i]
                    ])
                    pct_above_ma50 = np.nanmean([
                        ((prices[i] - ma50[i]) / ma50[i] * 100) if ma50[i] else 0 
                        for i in recent_idx if i < len(ma50) and ma50[i]
                    ])
                    
                    # Calculate forward gain (30 days if available)
                    if len(prices) > 30:
                        forward_gain = ((prices[-1] - prices[-31]) / prices[-31] * 100) if prices[-31] else 0
                    else:
                        forward_gain = 0
                    
                    # Bucket and store
                    ma20_bucket = min(int(pct_above_ma20 / 2) * 2, 20)  # Bucket by 2%
                    ma50_bucket = min(int(pct_above_ma50 / 2) * 2, 20)
                    
                    if is_sp500:
                        if len(ma20_gains['sp500']) <= ma20_bucket:
                            ma20_gains['sp500'].extend([0] * (ma20_bucket + 1 - len(ma20_gains['sp500'])))
                        if len(ma50_gains['sp500']) <= ma50_bucket:
                            ma50_gains['sp500'].extend([0] * (ma50_bucket + 1 - len(ma50_gains['sp500'])))
                        ma20_gains['sp500'][ma20_bucket] = (ma20_gains['sp500'][ma20_bucket] + forward_gain) / 2
                        ma50_gains['sp500'][ma50_bucket] = (ma50_gains['sp500'][ma50_bucket] + forward_gain) / 2
                    
                    if is_nasdaq100:
                        if len(ma20_gains['nasdaq100']) <= ma20_bucket:
                            ma20_gains['nasdaq100'].extend([0] * (ma20_bucket + 1 - len(ma20_gains['nasdaq100'])))
                        if len(ma50_gains['nasdaq100']) <= ma50_bucket:
                            ma50_gains['nasdaq100'].extend([0] * (ma50_bucket + 1 - len(ma50_gains['nasdaq100'])))
                        ma20_gains['nasdaq100'][ma20_bucket] = (ma20_gains['nasdaq100'][ma20_bucket] + forward_gain) / 2
                        ma50_gains['nasdaq100'][ma50_bucket] = (ma50_gains['nasdaq100'][ma50_bucket] + forward_gain) / 2
                    
                    # All stocks
                    if len(ma20_gains['full5000']) <= ma20_bucket:
                        ma20_gains['full5000'].extend([0] * (ma20_bucket + 1 - len(ma20_gains['full5000'])))
                    if len(ma50_gains['full5000']) <= ma50_bucket:
                        ma50_gains['full5000'].extend([0] * (ma50_bucket + 1 - len(ma50_gains['full5000'])))
                    ma20_gains['full5000'][ma20_bucket] = (ma20_gains['full5000'][ma20_bucket] + forward_gain) / 2
                    ma50_gains['full5000'][ma50_bucket] = (ma50_gains['full5000'][ma50_bucket] + forward_gain) / 2
                
                # Track optimal MA period
                if len(prices) >= ma_max:
                    best_period = 175  # Default
                    best_score = float('inf')
                    for period in range(ma_min, ma_max + 1, 5):
                        ma_vals = pd.Series(prices).rolling(window=period).mean().values
                        recent_ma = ma_vals[-60:]
                        recent_prices = prices[-60:]
                        if len(recent_ma) > 0:
                            deviations = recent_prices - recent_ma
                            score = np.std(deviations[~np.isnan(deviations)])
                            if score < best_score:
                                best_score = score
                                best_period = period
                    
                    if best_period not in ma_period_counts:
                        ma_period_counts[best_period] = {'sp500': 0, 'nasdaq100': 0, 'full5000': 0}
                    
                    if is_sp500:
                        ma_period_counts[best_period]['sp500'] += 1
                    if is_nasdaq100:
                        ma_period_counts[best_period]['nasdaq100'] += 1
                    ma_period_counts[best_period]['full5000'] += 1
            except Exception as e:
                continue
        
        conn.close()
        
        # Format MA20/50 labels
        ma20_gains['labels'] = [f'{i*2}%' for i in range(len(ma20_gains['sp500']))]
        ma50_gains['labels'] = [f'{i*2}%' for i in range(len(ma50_gains['sp500']))]
        
        # Normalize MA period counts to frequencies
        ma_period_sorted = sorted(ma_period_counts.keys())
        max_count = max([max(ma_period_counts[p].values()) for p in ma_period_sorted]) if ma_period_sorted else 1
        
        ma_period_dist = {
            'labels': [str(p) for p in ma_period_sorted],
            'sp500': [ma_period_counts[p]['sp500'] / max_count for p in ma_period_sorted],
            'nasdaq100': [ma_period_counts[p]['nasdaq100'] / max_count for p in ma_period_sorted],
            'full5000': [ma_period_counts[p]['full5000'] / max_count for p in ma_period_sorted]
        }
        
        return jsonify({
            'ma20_gains': ma20_gains,
            'ma50_gains': ma50_gains,
            'ma_period_distribution': ma_period_dist
        })
    except Exception as e:
        print(f"Error calculating statistics: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def get_sp500_symbols():
    """Get S&P 500 symbols (cached)"""
    sp500 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'JNJ', 'WMT',
             'V', 'MA', 'PG', 'XOM', 'HD', 'MCD', 'NKE', 'ADBE', 'CRM', 'CVX']
    return set(sp500)

def get_nasdaq100_symbols():
    """Get Nasdaq 100 symbols (cached)"""
    nasdaq100 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX', 'AVGO', 'COST',
                 'ADBE', 'CMCSA', 'ASML', 'PEP', 'QCOM', 'INTC', 'AMD', 'CSCO', 'INTU', 'SBUX']
    return set(nasdaq100)

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get current screening settings"""
    return jsonify(screening_state['settings'])

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update screening settings and persist to disk"""
    data = request.json
    screening_state['settings'].update(data)
    
    # Save to disk
    if save_settings(screening_state['settings']):
        return jsonify({'success': True, 'settings': screening_state['settings']})
    else:
        return jsonify({'success': False, 'error': 'Failed to save settings', 'settings': screening_state['settings']}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Simple health check for UI tester"""
    return jsonify({
        'ok': True,
        'status': screening_state.get('status', 'idle'),
        'progress': screening_state.get('progress', 0),
        'results_count': len(screening_state.get('results', []))
    })

@app.route('/tester')
def tester():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>10x Hunter - UI Tester</title>
    <style>
        body { font-family: Segoe UI, Roboto, sans-serif; background: #0f1419; color: #e0e0e0; padding: 20px; }
        .card { background: #1a1f2e; border: 1px solid #2a3f5f; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
        button { background: #1e88e5; color: white; border: none; border-radius: 4px; padding: 10px 16px; cursor: pointer; }
        pre { background: #0f1419; padding: 12px; border-radius: 6px; border: 1px solid #2a3f5f; }
    </style>
    <script>
        async function api(url, options) {
            const res = await fetch(url, options);
            const text = await res.text();
            try { return JSON.parse(text); } catch { return { raw: text, status: res.status }; }
        }
        async function testHealth() {
            const out = document.getElementById('healthOut');
            out.textContent = JSON.stringify(await api('/api/health'), null, 2);
        }
        async function testScan() {
            const out = document.getElementById('scanOut');
            out.textContent = 'Starting scan...';
            await api('/api/scan', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ capital: 50000, risk_tolerance: 8, universe_size: 500 }) });
            // Poll status until complete
            for (let i = 0; i < 120; i++) { // up to ~60s
                const status = await api('/api/status');
                out.textContent = JSON.stringify(status, null, 2);
                if (status.status === 'complete') break;
                await new Promise(r => setTimeout(r, 500));
            }
        }
        async function testSettings() {
            const out = document.getElementById('settingsOut');
            const threshold = Math.round(Math.random() * 50) / 10; // 0.0 - 5.0
            const updated = await api('/api/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ conviction_threshold: threshold }) });
            const results = await api('/api/results?min_conviction=' + encodeURIComponent(updated.conviction_threshold));
            out.textContent = JSON.stringify(results, null, 2);
        }
    </script>
</head>
<body onload="try{initUI();}catch(e){console.error('onload init error', e); var s=document.getElementById('uiStatus'); if(s) s.textContent='UI status: onload init error: '+e.message;}">
    <h1>10x Hunter - UI Tester</h1>
    <div class="card">
        <h3>Health</h3>
        <button onclick="testHealth()">Ping Health</button>
        <pre id="healthOut">Click to test health</pre>
    </div>
    <div class="card">
        <h3>Scan</h3>
        <button onclick="testScan()">Start Scan (500)</button>
        <pre id="scanOut">Click to start a test scan</pre>
    </div>
    <div class="card">
        <h3>Settings + Results</h3>
        <button onclick="testSettings()">Apply Random Conviction Threshold</button>
        <pre id="settingsOut">Click to update settings and list results</pre>
    </div>
</body>
</html>'''

@app.route('/test_click')
def serve_test_click():
    """Serve the standalone click tester HTML file (if present)."""
    try:
        if os.path.exists('test_click.html'):
            return send_file('test_click.html')
        return "test_click.html not found", 404
    except Exception as e:
        return (f"Error serving test_click.html: {e}"), 500


@app.route('/api/mock_results', methods=['GET'])
def api_mock_results():
    """Return mock results for debug/testing when scans are not available.
    No limits: if real results exist, return all filtered by current threshold.
    """
    try:
        results = screening_state.get('results', [])
        if results:
            min_conviction = screening_state['settings'].get('conviction_threshold', 4.0)
            filtered = [c for c in results if c.get('conviction', 0) >= min_conviction]
            resp = {'results': filtered, 'count': len(filtered), 'total': len(results), 'min_conviction': min_conviction}
            return jsonify(clean_nan_values(resp))
    except Exception:
        pass

    # Fallback mock data
    mock = [
        {
            'ticker': 'AAPL',
            'conviction': 8.5,
            'price': 180.25,
            'optimal_ma': 172,
            'ma_value': 165.0,
            'stop_loss': 165.0,
            'take_profit': 250.0,
            'position_size': 5000,
            'company_name': 'Apple Inc.',
            'sector': 'Technology',
            'description': 'Mocked company for testing.',
            'ai_reasoning': ['Strong momentum', 'Above optimal MA'],
            'potential_x': 5.0
        },
        {
            'ticker': 'MSFT',
            'conviction': 7.2,
            'price': 315.12,
            'optimal_ma': 180,
            'ma_value': 300.0,
            'stop_loss': 290.0,
            'take_profit': 380.0,
            'position_size': 4000,
            'company_name': 'Microsoft Corp.',
            'sector': 'Technology',
            'description': 'Mocked company for testing.',
            'ai_reasoning': ['Healthy earnings cadence', 'Institutional accumulation'],
            'potential_x': 3.2
        }
    ]
    resp = {'results': mock, 'count': len(mock), 'total': len(mock)}
    return jsonify(clean_nan_values(resp))

    def warm_cache_for_tickers(tickers, days=1095):
        """Prefetch and persist price (3y), fundamentals, and news for given tickers."""
        try:
            print(f"\n[WarmCache] Starting warm for {len(tickers)} tickers…")
            hunter = TenXHunter()
            for i, ticker in enumerate(tickers):
                try:
                    t = ticker.upper()
                    print(f"[WarmCache] [{i+1}/{len(tickers)}] {t}")

                    # Prices: ensure 3y present
                    df = db.get_price_data(t, days=days)
                    need_prices = (df is None) or (len(df) < 200)
                    if need_prices:
                        try:
                            new_df = yf.download(t, period='5y', progress=False, auto_adjust=False)
                            if not new_df.empty:
                                if isinstance(new_df.columns, pd.MultiIndex):
                                    new_df.columns = [col[0] for col in new_df.columns]
                                new_df = normalize_price_df(new_df)
                                db.save_price_data(t, new_df)
                                print(f"[WarmCache] Saved prices (3y) for {t} rows={len(new_df)}")
                        except Exception as e:
                            print(f"[WarmCache] Price fetch failed for {t}: {e}")

                    # Fundamentals: fetch & save if missing/stale
                    try:
                        fund, is_fresh = db.get_fundamentals(t)
                    except Exception:
                        fund, is_fresh = (None, False)
                    if not fund or not is_fresh:
                        try:
                            fund_data = hunter.get_fundamentals(t)
                            if fund_data:
                                db.save_fundamentals(t, fund_data)
                                print(f"[WarmCache] Saved fundamentals for {t}")
                        except Exception as e:
                            print(f"[WarmCache] Fundamentals fetch failed for {t}: {e}")

                    # News: fetch recent and save
                    try:
                        articles = hunter.get_news(t, limit=10)
                        if articles:
                            db.save_news(t, articles)
                            print(f"[WarmCache] Saved news ({len(articles)}) for {t}")
                    except Exception as e:
                        print(f"[WarmCache] News fetch failed for {t}: {e}")
                except Exception as e:
                    print(f"[WarmCache] Error warming {ticker}: {e}")
            print("[WarmCache] Done.")
        except Exception as e:
            print(f"[WarmCache] Fatal error: {e}")


@app.route('/debug_cards')
def debug_cards():
    """Standalone debug page to reproduce card click -> detail flow."""
    return '''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Debug Cards - Robust Tester</title>
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <style>
        body{font-family:Segoe UI,Roboto,Arial;background:#0f1419;color:#e0e0e0;padding:20px}
        .card{background:#1a1f2e;padding:16px;border-radius:8px;margin:10px 0;cursor:pointer;border:1px solid #2a3f5f}
        #detail{background:#0f1419;border:1px solid #2a3f5f;padding:12px;border-radius:8px;margin-top:12px;white-space:pre-wrap}
        button{margin-right:8px}
        #log{margin-top:12px;background:#0f1419;border:1px solid #2a3f5f;padding:8px;height:120px;overflow:auto}
    </style>
</head>
<body>
    <h2>Debug Cards - Robust Tester</h2>
    <div>
        <button id="btnMock">Load Mock Results</button>
        <button id="btnLive">Load Live /api/results</button>
        <button id="btnClear">Clear</button>
    </div>
    <div id="cards" style="margin-top:12px"></div>
    <div id="detail"><strong>Detail</strong><pre id="detailJson">(click a card)</pre></div>
    <div id="log"></div>

    <script>
    function log(msg){ const el=document.getElementById('log'); const d=document.createElement('div'); d.textContent=(new Date()).toLocaleTimeString()+': '+msg; el.appendChild(d); el.scrollTop=el.scrollHeight; console.log(msg);} 

    function createCard(o){
        const d = document.createElement('div');
        d.className='card';
        d.setAttribute('data-ticker', o.ticker);
        d.innerHTML = `<strong>${o.ticker}</strong> - ${o.company_name||''} <span style="float:right">${(o.conviction||0).toFixed(1)}/10</span><div style="margin-top:8px;color:#b0b0b0">${(o.ai_reasoning||[]).slice(0,2).join(' • ')}</div>`;

        // Bind a robust click handler using addEventListener
        d.addEventListener('click', function(e){
            e.stopPropagation();
            const ticker = this.getAttribute('data-ticker');
            log('Card clicked -> ' + ticker);
            // report to server for capture
            navigator.sendBeacon && navigator.sendBeacon('/client_log', JSON.stringify({event:'card_click', ticker: ticker}));
            showOverlay(ticker);
            fetch('/api/stock/' + encodeURIComponent(ticker))
                .then(r=>{ log('fetch /api/stock status: ' + r.status); return r.json(); })
                .then(j=>{
                    hideOverlay();
                    document.getElementById('detailJson').textContent = JSON.stringify(j, null, 2);
                    log('Detail received for ' + ticker);
                })
                .catch(err=>{ hideOverlay(); log('Error fetching detail: '+err); console.error(err); });
        });

        return d;
    }

    function showOverlay(t){ let el=document.getElementById('ov'); if(!el){ el=document.createElement('div'); el.id='ov'; el.style='position:fixed;left:0;top:0;right:0;bottom:0;background:rgba(0,0,0,0.7);display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;z-index:9999'} el.textContent='Loading details... ('+t+')'; document.body.appendChild(el);} 
    function hideOverlay(){ const el=document.getElementById('ov'); if(el) el.remove(); }

    document.getElementById('btnMock').addEventListener('click', function(){ fetch('/api/mock_results').then(r=>r.json()).then(d=>{ const c=document.getElementById('cards'); c.innerHTML=''; d.results.forEach(rw=> c.appendChild(createCard(rw))); log('Loaded mock results'); }).catch(e=>log('Error loading mock: '+e)); });
    document.getElementById('btnLive').addEventListener('click', function(){ fetch('/api/results').then(r=>r.json()).then(d=>{ const arr=d.results||d; const c=document.getElementById('cards'); c.innerHTML=''; arr.forEach(rw=> c.appendChild(createCard(rw))); log('Loaded live results'); }).catch(e=>log('Error loading live: '+e)); });
    document.getElementById('btnClear').addEventListener('click', function(){ document.getElementById('cards').innerHTML=''; document.getElementById('detailJson').textContent='(click a card)'; document.getElementById('log').innerHTML=''; });

    // Auto-load mock on open for convenience
    fetch('/api/mock_results').then(r=>r.json()).then(d=>{ const c=document.getElementById('cards'); c.innerHTML=''; d.results.forEach(rw=> c.appendChild(createCard(rw))); log('Auto-loaded mock results on open'); }).catch(e=>log('Auto-load error: '+e));
    </script>
</body>
</html>'''

@app.route('/')
def index():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>10x Hunter Pro - AI Stock Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/luxon@2.5.2/build/global/luxon.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1.2.0"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-financial@0.2.1"></script>
    <script>
        window.chartPluginsLoaded = true;
        try {
            const fin = window['chartjs-chart-financial'];
            if (fin && Chart && Chart.register) {
                const { CandlestickController, OhlcController, CandlestickElement, OhlcElement } = fin;
                Chart.register(CandlestickController, OhlcController, CandlestickElement, OhlcElement);
                console.log('✓ Financial controllers registered');
            } else {
                console.warn('Financial controllers not available for registration');
            }
        } catch (e) {
            console.error('Financial registration error', e);
        }
        console.log('✓ Chart.js ' + Chart.version + ' loaded');
        console.log('✓ Financial plugin loaded');
    </script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f1419;
            color: #e0e0e0;
        }
        .container { max-width: 1800px; margin: 0 auto; padding: 20px; }
        header {
            background: linear-gradient(135deg, #1e88e5, #0d47a1);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 4px 20px rgba(30,136,229,0.3);
        }
        header h1 { font-size: 2.5em; margin-bottom: 10px; }
        header p { opacity: 0.9; }
        .view-container { display: none; }
        .view-container.active { display: block; }
        .controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .control-group {
            background: #1a1f2e;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #2a3f5f;
        }
        .control-group label { display: block; margin-bottom: 8px; font-weight: 600; }
        .control-group input, .control-group select {
            width: 100%;
            padding: 10px;
            border: 1px solid #2a3f5f;
            border-radius: 4px;
            font-size: 1em;
            background: #0f1419;
            color: #e0e0e0;
        }
        button {
            padding: 12px 20px;
            border: none;
            border-radius: 4px;
            font-weight: 600;
            cursor: pointer;
            background: #1e88e5;
            color: white;
            width: 100%;
            transition: all 0.3s;
        }
        button:hover { background: #1565c0; transform: translateY(-2px); }
        button:disabled { background: #555; cursor: not-allowed; }
        .progress {
            background: #1a1f2e;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: none;
            border: 1px solid #2a3f5f;
        }
        .progress.active { display: block; }
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #0f1419;
            border-radius: 4px;
            overflow: hidden;
            border: 1px solid #2a3f5f;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #1e88e5, #00c853);
            width: 0%;
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
        }
        .results { background: #1a1f2e; border-radius: 8px; border: 1px solid #2a3f5f; }
        .opportunity-card {
            background: linear-gradient(135deg, #2d2d2d 0%, #252525 100%);
            border-radius: 12px;
            margin-bottom: 20px;
            cursor: pointer !important;
            transition: all 0.3s;
            border-left: 5px solid #00c853;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            position: relative;
            pointer-events: auto !important;
        }
        .opportunity-card * {
            pointer-events: none !important;
        }
        .opportunity-card .buy-btn,
        .opportunity-card .skip-btn {
            pointer-events: auto !important;
        }
        .opportunity-card:hover {
            border-left-color: #ffb300;
            transform: translateY(-4px);
            box-shadow: 0 8px 20px rgba(0,200,83,0.3);
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 20px;
            background: rgba(0,200,83,0.1);
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .ticker-section {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .ticker-large {
            font-size: 1.5em;
            font-weight: bold;
            color: #00c853;
            letter-spacing: 0.5px;
        }
        .company-name {
            font-size: 0.85em;
            color: #999;
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .card-reasoning {
            margin-left: 20px;
            color: #bcd;
            font-size: 0.9em;
            max-width: 600px;
        }
        .card-reasoning .reason-item {
            color: #cfe8d8;
            opacity: 0.95;
            margin-top: 4px;
            font-size: 0.9em;
        }
        .sector-tag {
            background: rgba(255,179,0,0.2);
            color: #ffb300;
            padding: 4px 12px;
            border-radius: 6px;
            font-size: 0.85em;
            font-weight: 500;
        }
        .conviction-section {
            text-align: right;
        }
        .conviction-score {
            display: block;
            font-size: 1.8em;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .potential-label {
            display: block;
            font-size: 0.9em;
            color: #ffb300;
            font-weight: 600;
            margin-top: 2px;
        }
        .card-body {
            padding: 20px;
        }
        .metrics-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
            gap: 12px;
            margin-bottom: 16px;
        }
        .metric-box {
            background: rgba(255,255,255,0.03);
            padding: 10px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.05);
        }
        .metric-label {
            font-size: 0.75em;
            color: #999;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-value {
            font-size: 1.1em;
            font-weight: 600;
            color: #e0e0e0;
        }
        .metric-value.primary {
            color: #00c853;
            font-size: 1.3em;
        }
        .metric-value.success {
            color: #00c853;
        }
        .analysis-text {
            background: rgba(0,200,83,0.05);
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 16px;
            color: #b0b0b0;
            font-size: 0.95em;
            line-height: 1.5;
            border-left: 3px solid #00c853;
        }
        .catalysts-section {
            margin-bottom: 16px;
        }
        .catalysts-section strong {
            color: #e0e0e0;
            font-size: 0.9em;
        }
        .catalyst-list {
            margin: 8px 0 0 0;
            padding-left: 20px;
            list-style: none;
        }
        .catalyst-list li {
            color: #999;
            font-size: 0.9em;
            margin-bottom: 4px;
            position: relative;
        }
        .catalyst-list li:before {
            content: "•";
            color: #00c853;
            font-weight: bold;
            position: absolute;
            left: -15px;
        }
        .trade-setup {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: 16px;
        }
        .buy-btn {
            background: linear-gradient(135deg, #00c853 0%, #00a843 100%);
            color: white;
            border: none;
            padding: 12px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.2s;
        }
        .buy-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(0,200,83,0.4);
        }
        .skip-btn {
            background: rgba(255,179,0,0.1);
            color: #ffb300;
            border: 1px solid rgba(255,179,0,0.3);
            padding: 12px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.2s;
        }
        .skip-btn:hover {
            background: rgba(255,179,0,0.2);
        }
        .empty { padding: 40px; text-align: center; color: #666; }
        h2 { margin-bottom: 20px; margin-top: 30px; color: #e0e0e0; }
        
        /* Detail view styles */
        .detail-view { background: #1a1f2e; padding: 30px; border-radius: 8px; border: 1px solid #2a3f5f; }
        .detail-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #2a3f5f;
        }
        .detail-header-left {
            display: flex;
            align-items: center;
            gap: 20px;
        }
        .detail-ticker { font-size: 2.5em; font-weight: 700; color: #1e88e5; }
        .detail-company-name { font-size: 0.9em; color: #999; }
        .detail-price { font-size: 1.5em; color: #00c853; }
        .back-btn { background: #2a3f5f; padding: 8px 16px; margin-right: auto; cursor: pointer; }
        
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        @media (max-width: 1200px) {
            .grid-2 { grid-template-columns: 1fr; }
        }
        
        .card {
            background: #0f1419;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #2a3f5f;
        }
        
        .card h3 { margin-bottom: 15px; color: #1e88e5; }
        
        .metric-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #2a3f5f;
        }
        
        .metric-label { color: #999; }
        .metric-value { font-weight: 600; color: #e0e0e0; }
        
        .ai-insights {
            background: #1a2633;
            padding: 20px;
            border-left: 4px solid #1e88e5;
            border-radius: 4px;
            margin-bottom: 15px;
        }
        
        .ai-insights ul {
            margin-left: 20px;
            margin-top: 10px;
        }
        
        .ai-insights li {
            margin: 8px 0;
            color: #e0e0e0;
            line-height: 1.6;
        }
        
        .chart-container {
            position: relative;
            height: 420px;
            margin: 20px 0;
        }

        .chart-header {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            gap: 12px;
            margin-bottom: 10px;
            flex-wrap: nowrap;
            width: 100%;
        }
        .chart-title { margin: 0; flex-shrink: 0; }
        .chart-actions {
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: nowrap;
            flex-shrink: 0;
        }
        .chart-range-group {
            display: flex;
            flex-wrap: nowrap;
            gap: 8px;
            flex-shrink: 0;
        }

        .range-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 64px;
            background: #1a2633;
            color: #e0e0e0;
            border: 1px solid #2a3f5f;
            padding: 8px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
        }
        .range-btn[aria-pressed="true"] {
            background: #1e88e5;
            border-color: #1e88e5;
            color: #fff;
        }
        .chart-toggle {
            background: #2a3f5f;
            padding: 10px 16px;
            border-radius: 6px;
            cursor: pointer;
            border: none;
            color: #e0e0e0;
            font-size: 0.9em;
        }
        
        .news-item {
            padding: 15px 0;
            border-bottom: 1px solid #2a3f5f;
        }
        
        .news-title {
            color: #1e88e5;
            text-decoration: none;
            font-weight: 600;
            margin-bottom: 5px;
        }
        
        .news-source {
            color: #999;
            font-size: 0.85em;
        }
        /* Loading overlay for detail view (visible when loading data) */
        #loadingOverlay {
            position: fixed;
            left: 0;
            top: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.6);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 9999;
            color: #fff;
            font-size: 1.1em;
            font-weight: 700;
        }
        #loadingOverlay .box {
            background: #0f1419;
            border: 1px solid #2a3f5f;
            padding: 18px 24px;
            border-radius: 8px;
            display: flex;
            gap: 12px;
            align-items: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div id="loadingOverlay"><div class="box">Loading details... <span id="loadingTicker"></span></div></div>
        <header>
            <h1>🚀 10x Hunter Pro</h1>
            <p>AI-Powered Stock Analysis for 10x Return Opportunities</p>
        </header>

        <!-- Main List View -->
        <div class="view-container active" id="listView">
            <div style="margin-bottom: 20px; text-align: right; display: flex; gap: 10px; justify-content: flex-end;">
                <button id="statsBtn" onclick="openStatistics()" style="background: #2a3f5f; padding: 8px 16px; width: auto;">📊 Statistics</button>
                <button id="settingsBtn" onclick="toggleSettings()" style="background: #2a3f5f; padding: 8px 16px; width: auto;">⚙ Settings</button>
            </div>

            <div id="uiStatus" style="margin-bottom: 12px; color: #1e88e5; font-weight: 700; font-size: 0.95em; background: rgba(30,136,229,0.1); border: 1px solid #2a3f5f; padding: 8px 10px; border-radius: 6px;">
                UI status: binding…
            </div>
            <script>
                try {
                    var s = document.getElementById('uiStatus');
                    if (s) s.textContent = 'UI status: inline script alive';
                    // Inline fallback binder removed (redundant). Main script handles bindings.
                } catch (e) { console.error('inline status set error', e); }
            </script>

            <div id="settingsPanel" style="display: none; background: #1a1f2e; padding: 20px; border-radius: 8px; border: 1px solid #2a3f5f; margin-bottom: 20px;">
                <h3 style="margin-bottom: 15px; color: #1e88e5;">Screening Filters</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div class="control-group">
                        <label>Min Conviction Score</label>
                        <input type="number" id="convictionThreshold" step="0.1" min="0" max="10" placeholder="0-10">
                    </div>
                    <div class="control-group">
                        <label>Font Size</label>
                        <input type="range" id="fontSize" step="0.1" min="0.8" max="1.5" value="1.0">
                        <div id="fontSizeValue" style="margin-top: 8px; font-weight: 600; color: #1e88e5;">100%</div>
                    </div>
                    <div class="control-group">
                        <label>RSI Min</label>
                        <input type="number" id="rsiMin" step="1" min="0" max="100" placeholder="0-100">
                    </div>
                    <div class="control-group">
                        <label>RSI Max</label>
                        <input type="number" id="rsiMax" step="1" min="0" max="100" placeholder="0-100">
                    </div>
                    <div class="control-group">
                        <label>Min Volume Ratio</label>
                        <input type="number" id="volumeRatioMin" step="0.1" min="0" placeholder="0.8-2.0">
                    </div>
                    <div class="control-group">
                        <label>MA Min Period</label>
                        <input type="number" id="maMinPeriod" step="5" min="50" max="300" placeholder="50-300">
                    </div>
                    <div class="control-group">
                        <label>MA Max Period</label>
                        <input type="number" id="maMaxPeriod" step="5" min="50" max="300" placeholder="50-300">
                    </div>
                </div>
                <button onclick="loadAndApplySettings()" style="margin-top: 15px; width: 100%;">💾 Apply Settings</button>
            </div>

            <div class="controls">
                <div class="control-group">
                    <label>Capital ($)</label>
                    <input type="number" id="capital" value="50000" min="1000">
                </div>
                <div class="control-group">
                    <label>Risk Tolerance (1-10)</label>
                    <input type="range" id="riskTolerance" min="1" max="10" value="8">
                    <div id="riskValue" style="margin-top: 8px; font-weight: 600; color: #1e88e5;">8/10</div>
                </div>
                <div class="control-group">
                    <label>Universe Size</label>
                    <select id="universeSize">
                        <option value="500">Top 500 (3 min)</option>
                        <option value="2000" selected>Top 2000 (12 min)</option>
                        <option value="5000">Top 5000 (25 min)</option>
                    </select>
                </div>
                <div class="control-group">
                    <button id="scanBtn" onclick="startScan()">▶ Run AI Screen</button>
                    <button id="testAaplBtn" style="background:#ff5252;margin-left:10px;">🧪 TEST AAPL</button>
                </div>
            </div>

            <div class="progress" id="progressDiv">
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill">0%</div>
                </div>
                <p style="margin-top: 10px; text-align: center; color: #999;">Analyzing stocks with AI...</p>
            </div>

            <h2>🔥 AI-Identified 10x Candidates</h2>
            <div class="results" id="resultsDiv">
                <div class="empty">Run a scan to see results</div>
            </div>
        </div>

        <!-- Detail View -->
        <div class="view-container" id="detailView">
            <button class="back-btn" onclick="backToList()">← Back to Results</button>
            
            <div class="detail-view">
                <div class="detail-header">
                    <div class="detail-header-left">
                        <div>
                            <div class="detail-ticker" id="detailTicker">-</div>
                            <div class="detail-company-name" id="detailCompanyName">-</div>
                        </div>
                    </div>
                    <div class="detail-price" id="detailPrice">$-</div>
                </div>

                <div class="grid-2">
                    <div>
                        <h3>📊 Technical Analysis</h3>
                        <div class="card">
                            <div class="metric-row">
                                <span class="metric-label">Current Price</span>
                                <span class="metric-value" id="tech-price">-</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">Optimal MA</span>
                                <span class="metric-value" id="tech-ma">-</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">RSI (14)</span>
                                <span class="metric-value" id="tech-rsi">-</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">Volume Ratio</span>
                                <span class="metric-value" id="tech-volume">-</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">Conviction Score</span>
                                <span class="metric-value" id="tech-conviction">-</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">1M Returns</span>
                                <span class="metric-value" id="returns-1m">-</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">3M Returns</span>
                                <span class="metric-value" id="returns-3m">-</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">1Y Returns</span>
                                <span class="metric-value" id="returns-1y">-</span>
                            </div>
                        </div>
                    </div>

                    <div>
                        <h3>💰 Fundamentals</h3>
                        <div class="card">
                            <div class="metric-row">
                                <span class="metric-label">P/E Ratio</span>
                                <span class="metric-value" id="fund-pe">-</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">Market Cap</span>
                                <span class="metric-value" id="fund-mc">-</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">Sector</span>
                                <span class="metric-value" id="fund-sector">-</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">Industry</span>
                                <span class="metric-value" id="fund-industry">-</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">ROE</span>
                                <span class="metric-value" id="fund-roe">-</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">Profit Margin</span>
                                <span class="metric-value" id="fund-pm">-</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">Debt/Equity</span>
                                <span class="metric-value" id="fund-de">-</span>
                            </div>
                        </div>
                    </div>
                </div>

                <h3>🤖 AI Analysis & 10x Potential</h3>
                <div class="ai-insights">
                    <strong>AI 10x Score: <span id="ai-score">-</span>/10</strong>
                    <ul id="ai-reasoning"></ul>
                </div>

                <div class="grid-2">
                    <div class="ai-insights">
                        <strong>🐂 Bull Case</strong>
                        <ul id="bull-case"></ul>
                    </div>
                    <div class="ai-insights">
                        <strong>🐻 Bear Case</strong>
                        <ul id="bear-case"></ul>
                    </div>
                </div>

                <h3>🎯 Potential Catalysts</h3>
                <div class="ai-insights">
                    <ul id="catalysts"></ul>
                </div>

                <div class="chart-header">
                    <h3 class="chart-title">📈 Price Chart</h3>
                    <div class="chart-actions">
                        <div class="chart-range-group">
                            <button class="range-btn" data-range="1m">1M</button>
                            <button class="range-btn" data-range="6m">6M</button>
                            <button class="range-btn" data-range="1y">1Y</button>
                            <button class="range-btn" data-range="3y" aria-pressed="true">3Y</button>
                        </div>
                        <button id="chartToggle" onclick="toggleChartType()" class="chart-toggle">🕯️ Candlestick</button>
                    </div>
                </div>
                <div class="chart-container">
                    <canvas id="priceChart"></canvas>
                </div>

                <h3>📰 Recent News</h3>
                <div class="card">
                    <div id="newsDiv"></div>
                </div>
            </div>
        </div>
    </div>

    <!-- Statistics Modal -->
    <div id="statisticsModal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); z-index: 1000; overflow: auto;">
        <div style="background: #0f1419; margin: 20px auto; padding: 30px; border-radius: 8px; max-width: 1400px; border: 2px solid #1e88e5;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <h2 style="color: #1e88e5; margin: 0;">📊 Historical Statistics</h2>
                <button id="closeStatsBtn" onclick="closeStatistics()" style="background: #d32f2f; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 1em;">✕ Close</button>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px;">
                <div style="background: #1a1f2e; padding: 20px; border-radius: 8px; border: 1px solid #2a3f5f;">
                    <h3 style="color: #1e88e5; margin-top: 0;">% Above MA20 → Gains</h3>
                    <div style="height: 300px; position: relative;">
                        <canvas id="ma20GainChart"></canvas>
                    </div>
                </div>
                <div style="background: #1a1f2e; padding: 20px; border-radius: 8px; border: 1px solid #2a3f5f;">
                    <h3 style="color: #1e88e5; margin-top: 0;">% Above MA50 → Gains</h3>
                    <div style="height: 300px; position: relative;">
                        <canvas id="ma50GainChart"></canvas>
                    </div>
                </div>
            </div>
            <div style="background: #1a1f2e; padding: 20px; border-radius: 8px; border: 1px solid #2a3f5f;">
                <h3 style="color: #1e88e5; margin-top: 0;">Optimal MA Period Distribution</h3>
                <div style="height: 200px; position: relative;">
                    <canvas id="maPeriodChart"></canvas>
                </div>
            </div>
        </div>
    </div>

    <script>
        let priceChart = null;
        let currentChartType = 'line';
        let currentChartData = null;
        let currentTechnical = null;
        let currentRange = '3y'; // default range

        // Global click beacon (capture) to diagnose click handling
        try {
            document.addEventListener('click', function(ev){
                try {
                    const t = ev.target;
                    const tag = t && t.tagName || 'UNKNOWN';
                    const id = t && t.id || '';
                    const cls = t && t.className || '';
                    const info = `GLOBAL_CLICK tag=${tag} id=${id} class=${cls}`;
                    if (navigator.sendBeacon) { try { navigator.sendBeacon('/client_log', info); } catch(_){} }
                    console.log(info);
                } catch(_) {}
            }, true);
        } catch(_) {}

        // Optional: auto-open AAPL detail when hash is set
        try {
            if (location && location.hash === '#test-aapl') {
                setTimeout(function(){
                    if (window.viewStockDetail) { window.viewStockDetail('AAPL'); }
                }, 1000);
            }
        } catch(_) {}

        // Global JS error banner for quick diagnosis
        window.addEventListener('error', function(ev) {
            try {
                var msg = (ev && ev.message) ? ev.message : 'Unknown error';
                var loc = '';
                try { loc = (ev && ev.filename ? ev.filename : '') + ':' + (ev && ev.lineno ? ev.lineno : 0) + ':' + (ev && ev.colno ? ev.colno : 0); } catch(_) {}
                var banner = document.createElement('div');
                banner.style.cssText = 'position:fixed;top:0;left:0;right:0;background:#b00020;color:#fff;padding:10px;z-index:9999;font-weight:600';
                banner.textContent = 'Script error: ' + msg + (loc ? (' @ ' + loc) : '');
                document.body.appendChild(banner);
                window.lastErrorMessage = msg;
                window.lastErrorLocation = loc;
                console.error('Global error at', loc, msg);
                if (navigator.sendBeacon) {
                    try { navigator.sendBeacon('/client_log', 'JS_ERROR:' + msg + '|LOC|' + loc); } catch(_) {}
                }
            } catch(e) {}
        });

        window.addEventListener('unhandledrejection', function(ev) {
            try {
                var msg = (ev && ev.reason && ev.reason.message) ? ev.reason.message : 'Unhandled rejection';
                console.error('Unhandled rejection:', ev);
                if (navigator.sendBeacon) {
                    try { navigator.sendBeacon('/client_log', 'JS_UNHANDLED_REJECTION:' + msg); } catch(_) {}
                }
            } catch(e) {}
        });

        function initUI() {
            try {
                const statusEl = document.getElementById('uiStatus');
                const note = (msg) => { if (statusEl) statusEl.textContent = 'UI status: ' + msg; console.log(msg); };
                window.uiNote = note; // expose for quick debugging

                note('binding controls…');

                // Load saved settings from server and populate form fields
                fetch('/api/settings')
                    .then(r => r.json())
                    .then(settings => {
                        document.getElementById('convictionThreshold').value = settings.conviction_threshold || 4.0;
                        document.getElementById('rsiMin').value = settings.rsi_min || 20;
                        document.getElementById('rsiMax').value = settings.rsi_max || 80;
                        document.getElementById('volumeRatioMin').value = settings.volume_ratio_min || 0.8;
                        document.getElementById('maMinPeriod').value = settings.ma_min_period || 150;
                        document.getElementById('maMaxPeriod').value = settings.ma_max_period || 200;
                        
                        // Load and apply font size
                        const fontSize = settings.font_size || 1.0;
                        const fontSizeElement = document.getElementById('fontSize');
                        if (fontSizeElement) {
                            fontSizeElement.value = fontSize;
                            document.getElementById('fontSizeValue').textContent = Math.round(fontSize * 100) + '%';
                        }
                        document.documentElement.style.fontSize = (fontSize * 16) + 'px';
                        
                        console.log('Loaded saved settings:', settings);
                    })
                    .catch(err => {
                        console.log('No saved settings found, using defaults');
                    });

                const btn = document.getElementById('scanBtn');
                if (btn) {
                    btn.addEventListener('click', (e) => {
                        e.preventDefault();
                        startScan();
                        note('Scan button clicked');
                    });
                    // Fallback inline assignment in case addEventListener is overridden
                    btn.onclick = (e) => { e.preventDefault(); startScan(); note('Scan button clicked (inline fallback)'); };
                    // Pointer/touch fallback
                    btn.addEventListener('pointerdown', (e) => { e.preventDefault(); startScan(); note('Scan button clicked (pointerdown)'); });
                    console.log('Scan button bound');
                    note('Scan button bound');
                } else {
                    console.warn('Scan button not found');
                    note('Scan button not found');
                }

                const settingsBtn = document.getElementById('settingsBtn');
                if (settingsBtn) {
                    settingsBtn.addEventListener('click', (e) => {
                        e.preventDefault();
                        toggleSettings();
                        note('Settings button clicked');
                    });
                    settingsBtn.onclick = (e) => { e.preventDefault(); toggleSettings(); note('Settings button clicked (inline fallback)'); };
                    settingsBtn.addEventListener('pointerdown', (e) => { e.preventDefault(); toggleSettings(); note('Settings button clicked (pointerdown)'); });
                    console.log('Settings button bound');
                    note('Settings button bound');
                } else {
                    console.warn('Settings button not found');
                    note('Settings button not found');
                }
                // Bind TEST AAPL button without inline handlers
                try {
                    const testBtn = document.getElementById('testAaplBtn');
                    if (testBtn) {
                        testBtn.addEventListener('click', (e) => {
                            e.preventDefault();
                            console.log('TEST clicked');
                            if (!window.viewStockDetail) {
                                alert('viewStockDetail not available');
                                if (navigator.sendBeacon) { try { navigator.sendBeacon('/client_log','VIEW_STOCK_DETAIL_MISSING'); } catch(_){} }
                            } else {
                                window.viewStockDetail('AAPL');
                            }
                        });
                    }
                } catch(_){}

                // Event delegation as an extra safety net
                document.addEventListener('click', (ev) => {
                    if (ev.target && ev.target.id === 'scanBtn') {
                        ev.preventDefault();
                        startScan();
                        note('Scan button clicked (delegated)');
                    }
                    if (ev.target && ev.target.id === 'settingsBtn') {
                        ev.preventDefault();
                        toggleSettings();
                        note('Settings button clicked (delegated)');
                    }
                }, true);

                // Bind range buttons
                try {
                    document.querySelectorAll('.range-btn').forEach(btn => {
                        btn.addEventListener('click', (e) => {
                            e.preventDefault();
                            setRange(btn.dataset.range);
                        });
                    });
                } catch(e) { console.error('range button bind error', e); }
                
                // Bind font size slider for live preview
                try {
                    const fontSizeSlider = document.getElementById('fontSize');
                    if (fontSizeSlider) {
                        fontSizeSlider.addEventListener('input', (e) => {
                            const fontSize = parseFloat(e.target.value) || 1.0;
                            document.documentElement.style.fontSize = (fontSize * 16) + 'px';
                            const fontSizeValueEl = document.getElementById('fontSizeValue');
                            if (fontSizeValueEl) {
                                fontSizeValueEl.textContent = Math.round(fontSize * 100) + '%';
                            }
                        });
                    }
                } catch(e) { console.error('font size slider bind error', e); }

                // Preload current settings into panel
                loadAndDisplaySettings();

                // Live-update threshold without alert (debounced)
                try {
                    const thr = document.getElementById('convictionThreshold');
                    let thrTimer = null;
                    if (thr) {
                        thr.addEventListener('input', (e) => {
                            const v = parseFloat(e.target.value) || 4.0;
                            clearTimeout(thrTimer);
                            thrTimer = setTimeout(() => {
                                fetch('/api/settings', {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ conviction_threshold: v })
                                })
                                .then(() => { loadResults(); note('Threshold updated to ' + v); })
                                .catch(err => console.error('threshold update error', err));
                            }, 250);
                        });
                    }
                } catch(e) { console.error('threshold live update setup error', e); }

                note('ready');

                // Delegated click handler for cards (main and fallback)
                const resultsDiv = document.getElementById('resultsDiv');
                if (resultsDiv) {
                    resultsDiv.addEventListener('click', (ev) => {
                        const card = ev.target.closest('.opportunity-card');
                        if (!card) return;
                        ev.preventDefault();
                        const t = card.dataset.ticker || (card.querySelector('.ticker-large') ? card.querySelector('.ticker-large').textContent.trim() : '');
                        note('card clicked ' + t);
                        if (window.viewStockDetail && t) {
                            window.viewStockDetail(t);
                        }
                    });
                }
            } catch (e) {
                console.error('Initialization error:', e);
                const statusEl = document.getElementById('uiStatus');
                if (statusEl) statusEl.textContent = 'UI status: Initialization error: ' + e.message;
            }
        }

        // Run init immediately if DOM is already ready; otherwise wait
        try {
            if (document.readyState === 'complete' || document.readyState === 'interactive') {
                initUI();
            } else {
                document.addEventListener('DOMContentLoaded', initUI);
            }
        } catch (e) {
            console.error('Initial init binding error:', e);
        }

        // Absolute fallbacks: retry binding a few times in case something prevents the first run
        const reboundAttempts = [300, 800, 1500];
        reboundAttempts.forEach((ms, idx) => {
            setTimeout(() => {
                try {
                    const before = (window.__uiBoundTs || 0);
                    initUI();
                    window.__uiBoundTs = Date.now();
                    const statusEl = document.getElementById('uiStatus');
                    if (statusEl && statusEl.textContent && statusEl.textContent.includes('binding')) {
                        statusEl.textContent = 'UI status: ready (fallback #' + (idx+1) + ')';
                    }
                    console.log('Fallback init attempt', idx+1, 'prevTs', before, 'now', window.__uiBoundTs);
                } catch (e) {
                    console.error('Fallback init error:', e);
                }
            }, ms);
        });

        // (Auto start disabled) Removed auto-trigger test to keep manual control

        function toggleChartType() {
            try {
                if (navigator.sendBeacon) { navigator.sendBeacon('/client_log', 'TOGGLE_START:' + currentChartType); }
                console.log('🔄 Toggle chart type. Current:', currentChartType);
                
                if (currentChartType === 'line') {
                    currentChartType = 'candlestick';
                    document.getElementById('chartToggle').textContent = '📈 Line';
                    console.log('→ Switched to candlestick mode');
                    if (navigator.sendBeacon) { navigator.sendBeacon('/client_log', 'TOGGLE_TO_CANDLESTICK'); }
                } else {
                    currentChartType = 'line';
                    document.getElementById('chartToggle').textContent = '🕯️ Candlestick';
                    console.log('→ Switched to line mode');
                    if (navigator.sendBeacon) { navigator.sendBeacon('/client_log', 'TOGGLE_TO_LINE'); }
                }
                
                if (currentChartData && currentTechnical) {
                    console.log('📊 Rendering chart with type:', currentChartType);
                    if (navigator.sendBeacon) { navigator.sendBeacon('/client_log', 'TOGGLE_RENDER:' + currentChartType); }
                    renderChart(currentChartData, currentTechnical);
                } else {
                    console.warn('No chart data available. Data:', !!currentChartData, 'Tech:', !!currentTechnical);
                    if (navigator.sendBeacon) { navigator.sendBeacon('/client_log', 'TOGGLE_NO_DATA'); }
                }
            } catch (e) {
                console.error('Toggle error:', e);
                if (navigator.sendBeacon) { navigator.sendBeacon('/client_log', 'TOGGLE_ERROR:' + e.message); }
            }
        }
        
        // Make toggleChartType globally accessible for onclick
        window.toggleChartType = toggleChartType;

        function openStatistics() {
            const modal = document.getElementById('statisticsModal');
            if (!modal) return;
            modal.style.display = 'block';
            loadStatistics();
        }

        function closeStatistics() {
            const modal = document.getElementById('statisticsModal');
            if (!modal) return;
            modal.style.display = 'none';
        }

        // Close modal when clicking outside
        window.addEventListener('click', (e) => {
            const modal = document.getElementById('statisticsModal');
            if (e.target === modal) {
                modal.style.display = 'none';
            }
        });

        let statsCharts = { ma20: null, ma50: null, maPeriod: null };

        function loadStatistics() {
            fetch('/api/statistics')
                .then(r => r.json())
                .then(data => {
                    console.log('Statistics data loaded:', data);
                    renderStatistics(data);
                })
                .catch(err => {
                    console.error('Error loading statistics:', err);
                    alert('Error loading statistics: ' + err.message);
                });
        }

        function renderStatistics(data) {
            // Render MA20 Gains Chart
            renderMA20Chart(data.ma20_gains);
            // Render MA50 Gains Chart
            renderMA50Chart(data.ma50_gains);
            // Render MA Period Distribution Chart
            renderMAPeriodChart(data.ma_period_distribution);
        }

        function renderMA20Chart(ma20Data) {
            const ctx = document.getElementById('ma20GainChart');
            if (!ctx) return;

            if (statsCharts.ma20) statsCharts.ma20.destroy();

            const datasets = [
                {
                    label: 'S&P 500',
                    data: ma20Data.sp500 || [],
                    backgroundColor: 'rgba(30, 136, 229, 0.4)',
                    borderColor: 'rgba(30, 136, 229, 0.8)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Nasdaq 100',
                    data: ma20Data.nasdaq100 || [],
                    backgroundColor: 'rgba(255, 179, 0, 0.4)',
                    borderColor: 'rgba(255, 179, 0, 0.8)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Full 5000',
                    data: ma20Data.full5000 || [],
                    backgroundColor: 'rgba(0, 200, 83, 0.4)',
                    borderColor: 'rgba(0, 200, 83, 0.8)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                }
            ];

            statsCharts.ma20 = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ma20Data.labels || [],
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: { color: '#e0e0e0' }
                        }
                    },
                    scales: {
                        y: {
                            ticks: { color: '#999' },
                            grid: { color: '#2a3f5f' },
                            title: { text: 'Avg Gain %', color: '#999' }
                        },
                        x: {
                            ticks: { color: '#999' },
                            grid: { color: '#2a3f5f' },
                            title: { text: '% Above MA20', color: '#999' }
                        }
                    }
                }
            });
        }

        function renderMA50Chart(ma50Data) {
            const ctx = document.getElementById('ma50GainChart');
            if (!ctx) return;

            if (statsCharts.ma50) statsCharts.ma50.destroy();

            const datasets = [
                {
                    label: 'S&P 500',
                    data: ma50Data.sp500 || [],
                    backgroundColor: 'rgba(30, 136, 229, 0.4)',
                    borderColor: 'rgba(30, 136, 229, 0.8)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Nasdaq 100',
                    data: ma50Data.nasdaq100 || [],
                    backgroundColor: 'rgba(255, 179, 0, 0.4)',
                    borderColor: 'rgba(255, 179, 0, 0.8)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Full 5000',
                    data: ma50Data.full5000 || [],
                    backgroundColor: 'rgba(0, 200, 83, 0.4)',
                    borderColor: 'rgba(0, 200, 83, 0.8)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                }
            ];

            statsCharts.ma50 = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ma50Data.labels || [],
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: { color: '#e0e0e0' }
                        }
                    },
                    scales: {
                        y: {
                            ticks: { color: '#999' },
                            grid: { color: '#2a3f5f' },
                            title: { text: 'Avg Gain %', color: '#999' }
                        },
                        x: {
                            ticks: { color: '#999' },
                            grid: { color: '#2a3f5f' },
                            title: { text: '% Above MA50', color: '#999' }
                        }
                    }
                }
            });
        }

        function renderMAPeriodChart(maPeriodData) {
            const ctx = document.getElementById('maPeriodChart');
            if (!ctx) return;

            if (statsCharts.maPeriod) statsCharts.maPeriod.destroy();

            const datasets = [
                {
                    label: 'S&P 500',
                    data: maPeriodData.sp500 || [],
                    backgroundColor: 'rgba(30, 136, 229, 0.6)',
                    borderColor: 'rgba(30, 136, 229, 0.8)',
                    borderWidth: 1
                },
                {
                    label: 'Nasdaq 100',
                    data: maPeriodData.nasdaq100 || [],
                    backgroundColor: 'rgba(255, 179, 0, 0.6)',
                    borderColor: 'rgba(255, 179, 0, 0.8)',
                    borderWidth: 1
                },
                {
                    label: 'Full 5000',
                    data: maPeriodData.full5000 || [],
                    backgroundColor: 'rgba(0, 200, 83, 0.6)',
                    borderColor: 'rgba(0, 200, 83, 0.8)',
                    borderWidth: 1
                }
            ];

            statsCharts.maPeriod = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: maPeriodData.labels || [],
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'x',
                    plugins: {
                        legend: {
                            labels: { color: '#e0e0e0' }
                        }
                    },
                    scales: {
                        y: {
                            stacked: false,
                            ticks: { color: '#999' },
                            grid: { color: '#2a3f5f' },
                            title: { text: 'Normalized Frequency', color: '#999' }
                        },
                        x: {
                            ticks: { color: '#999' },
                            grid: { color: '#2a3f5f' },
                            title: { text: 'MA Period', color: '#999' }
                        }
                    }
                }
            });
        }

        function setRange(range) {
            currentRange = range;
            const btns = document.querySelectorAll('.range-btn');
            btns.forEach(b => b.setAttribute('aria-pressed', b.dataset.range === range ? 'true' : 'false'));
            if (currentChartData && currentTechnical) {
                renderChart(currentChartData, currentTechnical);
            }
        }

        let __lastToggleTs = 0;
        function toggleSettings() {
            const now = Date.now();
            if (now - __lastToggleTs < 150) return; // prevent double toggle from multiple handlers
            __lastToggleTs = now;
            const panel = document.getElementById('settingsPanel');
            if (!panel) return;
            if (panel.style.display === 'none') {
                loadAndDisplaySettings();
                panel.style.display = 'block';
            } else {
                panel.style.display = 'none';
            }
        }
        // Ensure global access for inline handlers or external testers
        window.toggleSettings = toggleSettings;

        function loadAndDisplaySettings() {
            fetch('/api/settings')
                .then(r => r.json())
                .then(settings => {
                    document.getElementById('convictionThreshold').value = settings.conviction_threshold;
                    document.getElementById('rsiMin').value = settings.rsi_min;
                    document.getElementById('rsiMax').value = settings.rsi_max;
                    document.getElementById('volumeRatioMin').value = settings.volume_ratio_min;
                    document.getElementById('maMinPeriod').value = settings.ma_min_period;
                    document.getElementById('maMaxPeriod').value = settings.ma_max_period;
                });
        }

        function loadAndApplySettings() {
            const settings = {
                conviction_threshold: parseFloat(document.getElementById('convictionThreshold').value) || 4.0,
                rsi_min: parseInt(document.getElementById('rsiMin').value) || 20,
                rsi_max: parseInt(document.getElementById('rsiMax').value) || 80,
                volume_ratio_min: parseFloat(document.getElementById('volumeRatioMin').value) || 0.8,
                ma_min_period: parseInt(document.getElementById('maMinPeriod').value) || 150,
                ma_max_period: parseInt(document.getElementById('maMaxPeriod').value) || 200,
                font_size: parseFloat(document.getElementById('fontSize').value) || 1.0,
                show_ma_20: true,  // Keep current defaults
                show_ma_50: true,
                show_ma_optimal_trend: true,
                ma_trend_min_period: 150,
                ma_trend_max_period: 200
            };

            fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings)
            })
            .then(r => r.json())
            .then(updated => {
                console.log('Settings saved:', updated);
                document.documentElement.style.fontSize = (settings.font_size * 16) + 'px';
                alert('Settings saved successfully!');
                // Immediately re-load results to apply conviction filter without re-scanning
                loadResults();
            })
            .catch(err => {
                console.error('Error saving settings:', err);
                alert('Failed to save settings: ' + err.message);
            });
        }

        document.getElementById('riskTolerance').addEventListener('input', (e) => {
            document.getElementById('riskValue').textContent = e.target.value + '/10';
        });

        function startScan() {
            const btn = document.getElementById('scanBtn');
            btn.disabled = true;
            
            const capital = document.getElementById('capital').value;
            const risk = document.getElementById('riskTolerance').value;
            const universe = document.getElementById('universeSize').value;

            document.getElementById('progressDiv').classList.add('active');
            document.getElementById('progressFill').style.width = '0%';
            document.getElementById('progressFill').textContent = '0%';
            // Provide immediate user feedback
            console.log('Starting scan...', { capital, risk, universe });
            const resultsDiv = document.getElementById('resultsDiv');
            resultsDiv.innerHTML = '<div class="empty">Starting scan… analyzing stocks with AI.</div>';

            fetch('/api/scan', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    capital: parseInt(capital),
                    risk_tolerance: parseInt(risk),
                    universe_size: parseInt(universe)
                })
            })
            .catch(err => {
                console.error('Error starting scan:', err);
                resultsDiv.innerHTML = '<div class="empty">Failed to start scan. Please check server and try again.</div>';
                document.getElementById('progressDiv').classList.remove('active');
                btn.disabled = false;
                const statusEl = document.getElementById('uiStatus');
                if (statusEl) statusEl.textContent = 'Scan start error: ' + err;
            });

            pollStatus(btn);
        }
        // Ensure global access for inline handlers or external testers
        window.startScan = startScan;

        function pollStatus(btn) {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    console.log('Status:', data);
                    const fill = document.getElementById('progressFill');
                    fill.style.width = data.progress + '%';
                    fill.textContent = data.progress + '%';

                    if (data.status === 'complete') {
                        console.log('Scan complete, loading results...');
                        setTimeout(() => {
                            loadResults();
                            document.getElementById('progressDiv').classList.remove('active');
                            btn.disabled = false;
                        }, 500);
                    } else {
                        setTimeout(() => pollStatus(btn), 500);
                    }
                })
                .catch(err => {
                    console.error('Error polling status:', err);
                    setTimeout(() => pollStatus(btn), 1000);
                });
        }

        function loadResults() {
            fetch('/api/results')
                .then(r => r.json())
                .then(data => {
                    console.log('Results loaded:', data);
                    
                    let results = data.results || data;
                    if (!Array.isArray(results)) results = [];
                    
                    const count = data.count !== undefined ? data.count : results.length;
                    const total = data.total !== undefined ? data.total : results.length;
                    const min_conviction = data.min_conviction !== undefined ? data.min_conviction : 4.0;
                    const minConv = Number(min_conviction);
                    
                    console.log(`Results loaded: ${count}/${total} candidates (min conviction: ${minConv})`);
                    
                    if (!results || results.length === 0) {
                        document.getElementById('resultsDiv').innerHTML = 
                            `<div class="empty">No candidates found with conviction >= ${minConv.toFixed(1)}. Try lowering the conviction threshold or running a new scan.</div>`;
                        return;
                    }

                    try {
                        const cardHtml = (opp) => {
                            const potentialX = opp.potential_x || ((opp.conviction / 10) * 10).toFixed(1);
                            const positionSize = opp.position_size || 2000;
                            const stopLoss = opp.stop_loss || (opp.price * 0.92);
                            const takeProfit = opp.take_profit || (opp.price * 2);
                            const reasonArr = opp.ai_reasoning || [];
                            const breakdown = opp.score_breakdown || {};
                            const breakdownHtml = Object.keys(breakdown).length ? `<div class="card-reasoning">${Object.entries(breakdown).filter(([k]) => k!=='total').map(([k,v]) => `<div class="reason-item">${k.replace(/_/g,' ')}: ${Number(v).toFixed(2)}</div>`).join('')}</div>` : '';
                            const reasonHtml = reasonArr.length ? `<div class="card-reasoning">${reasonArr.slice(0,2).map(r => `<div class="reason-item">${r}</div>`).join('')}</div>` : breakdownHtml;
                            const cats = (opp.catalysts && opp.catalysts.length > 0 ? opp.catalysts : [
                                opp.gain_1m > 15 ? '1-month momentum surge' : 'Positive momentum trend',
                                opp.gain_6m > 20 ? '6-month breakout' : 'Technical strength',
                                'Earnings upcoming'
                            ]).slice(0,3);
                            // Check for earnings catalysts for badge
                            const hasEarnings = (opp.catalysts || []).some(c => c.toLowerCase().includes('earning'));
                            const earningsBadge = hasEarnings ? `<span style="display: inline-block; background: #ff6b35; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; font-weight: 600; margin-left: 8px;">📅 EARNINGS SOON</span>` : '';
                            return `
                            <div class="opportunity-card" data-ticker="${opp.ticker}" role="button" aria-label="View ${opp.ticker} details" style="cursor: pointer;" tabindex="0">
                                <div class="card-header">
                                    <div class="ticker-section">
                                        <span class="ticker-large" style="cursor:pointer">${opp.ticker}</span>
                                        <span class="company-name">${opp.company_name || opp.ticker}</span>
                                        <span class="sector-tag">${opp.sector || 'Technology'}</span>
                                        ${earningsBadge}
                                    </div>
                                    <div class="conviction-section">
                                        <span class="conviction-score">${opp.conviction.toFixed(1)}/10</span>
                                        <span class="potential-label">${potentialX}x potential</span>
                                    </div>
                                    ${reasonHtml}
                                </div>
                                <div class="card-body">
                                    <div class="metrics-row">
                                        <div class="metric-box">
                                            <div class="metric-label">Current Price</div>
                                            <div class="metric-value primary">$${opp.price.toFixed(2)}</div>
                                        </div>
                                        <div class="metric-box">
                                            <div class="metric-label">${opp.optimal_ma || 200}-Day MA</div>
                                            <div class="metric-value">$${(opp.ma_value || opp.price * 0.95).toFixed(2)}</div>
                                        </div>
                                        <div class="metric-box">
                                            <div class="metric-label">Stop Loss</div>
                                            <div class="metric-value" style="color: #ff5252;">$${stopLoss.toFixed(2)}</div>
                                        </div>
                                        <div class="metric-box">
                                            <div class="metric-label">Target Price</div>
                                            <div class="metric-value" style="color: #00c853;">$${takeProfit.toFixed(2)}</div>
                                        </div>
                                        <div class="metric-box">
                                            <div class="metric-label">Position Size</div>
                                            <div class="metric-value success">$${positionSize.toLocaleString()}</div>
                                        </div>
                                    </div>
                                    <div class="analysis-text">
                                        ${opp.description || ('Price > ' + (opp.optimal_ma || 200) + '-MA, bullish momentum.')}
                                    </div>
                                    <div class="catalysts-section">
                                        <strong>Catalysts:</strong>
                                        <ul class="catalyst-list">${cats.map(c => `<li>${c}</li>`).join('')}</ul>
                                    </div>
                                    <div class="trade-setup">
                                        <button class="buy-btn" onclick="event.stopPropagation(); alert('Order placed: ${opp.ticker}');">BUY $${positionSize.toLocaleString()}</button>
                                        <button class="skip-btn" onclick="event.stopPropagation();">SKIP FOR NOW</button>
                                    </div>
                                </div>
                            </div>`;
                        };

                        const headerMsg = (() => {
                            let msg = '✓ Found ' + count + ' candidates';
                            if (count < total) msg += ' out of ' + total + ' (filtered at ' + minConv.toFixed(1) + '/10 conviction)';
                            return `
                                <div style="padding: 12px; background: #2a3f5f; border-radius: 6px; margin-bottom: 16px; border-left: 4px solid #1e88e5;">
                                    <p style="margin: 0; color: #1e88e5; font-weight: 600;">${msg}</p>
                                </div>`;
                        })();

                        const momentum = results.filter(r => r.is_momentum);
                        const tenx = results.filter(r => r.is_ten_x);
                        const others = results.filter(r => !r.is_momentum && !r.is_ten_x);

                        const colStyle = 'flex: 1; min-width: 400px;';
                        const html = headerMsg + `
                            <div style="display: flex; gap: 30px; flex-wrap: wrap;">
                                <div style="${colStyle}">
                                    <h2 style="color:#00c853; border-bottom: 2px solid #00c853; padding-bottom: 10px;">🌟 TenX Picks (${tenx.length})</h2>
                                    ${tenx.length > 0 ? tenx.map(cardHtml).join('') : '<div style="color:#666; padding:20px;">No TenX candidates found</div>'}
                                </div>
                                <div style="${colStyle}">
                                    <h2 style="color:#ffb300; border-bottom: 2px solid #ffb300; padding-bottom: 10px;">⚡ Momentum Picks (${momentum.length})</h2>
                                    ${momentum.length > 0 ? momentum.map(cardHtml).join('') : '<div style="color:#666; padding:20px;">No Momentum candidates found</div>'}
                                </div>
                            </div>
                            ${others.length > 0 ? `<div style="margin-top: 30px;"><h2 style="color:#999; border-bottom: 2px solid #2a3f5f; padding-bottom: 10px;">📎 Other Candidates (${others.length})</h2>${others.map(cardHtml).join('')}</div>` : ''}`;

                        const container = document.getElementById('resultsDiv');
                        container.innerHTML = html;
                        console.log('Results rendered (Momentum/TenX split)');

                        setTimeout(() => {
                            const cards = document.querySelectorAll('.opportunity-card');
                            console.log(`Found ${cards.length} cards to bind`);
                            cards.forEach(card => {
                                const ticker = card.getAttribute('data-ticker');
                                card.addEventListener('click', function(e) {
                                    if (e.target.closest('.buy-btn') || e.target.closest('.skip-btn')) return;
                                    e.preventDefault();
                                    if (window.viewStockDetail) viewStockDetail(ticker);
                                });
                                card.addEventListener('mouseenter', function(){ this.style.opacity = '0.8'; });
                                card.addEventListener('mouseleave', function(){ this.style.opacity = '1'; });
                            });
                        }, 0);

                        document.addEventListener('click', function(e){
                            try {
                                const t = e.target.closest && e.target.closest('.ticker-large');
                                if (!t) return;
                                const ticker = t.textContent && t.textContent.trim();
                                if (!ticker) return;
                                if (e.target.closest && (e.target.closest('.buy-btn') || e.target.closest('.skip-btn'))) return;
                                e.preventDefault();
                                if (window.viewStockDetail) window.viewStockDetail(ticker);
                            } catch(err) { console.error('ticker delegated handler error', err); }
                        }, true);
                    } catch (err) {
                        console.error('Error rendering results:', err);
                        document.getElementById('resultsDiv').innerHTML = 
                            '<div class="empty">Error rendering results: ' + err.message + '</div>';
                    }
                })
                .catch(err => {
                    console.error('Error loading results:', err);
                    document.getElementById('resultsDiv').innerHTML = 
                        '<div class="empty">Error loading results. Please try again.</div>';
                });
        }

            // Global delegated click handler as last-resort fallback (catches dynamically rendered cards)
        document.addEventListener('click', function(e) {
            try {
                const btn = e.target.closest('.buy-btn, .skip-btn');
                if (btn) return; // ignore clicks on action buttons

                const card = e.target.closest('.opportunity-card');
                if (!card) return;

                const ticker = card.getAttribute('data-ticker') || (card.querySelector('.ticker-large') ? card.querySelector('.ticker-large').textContent.trim() : null);
                if (!ticker) return;

                console.log('Delegated capture: opportunity-card clicked ->', ticker);
                // Small visual flash for immediate feedback
                card.style.transition = 'box-shadow 0.12s ease, transform 0.12s ease';
                card.style.transform = 'translateY(-3px)';
                setTimeout(() => { card.style.transform = ''; }, 120);

                // call viewStockDetail asynchronously to avoid interrupting other handlers
                setTimeout(() => {
                    try { viewStockDetail(ticker); } catch(err) { console.error('delegated viewStockDetail error', err); }
                }, 0);
            } catch (err) {
                console.error('delegation handler error', err);
            }
        }, false);

            // Debug panel removed to reduce script surface and avoid parse issues

        function displayStockDetail(data) {
            try {
                console.log('🔍 displayStockDetail called');
                console.log('Full data object:', JSON.stringify(data, null, 2));

                // Single, robust populate path
                forcePopulateDetail(data);
                hideLoadingOverlay();
                console.log('✅ displayStockDetail COMPLETE');
            } catch (err) {
                console.error('❌ ERROR in displayStockDetail:', err);
                console.error('Stack:', err.stack);
                alert('Error: ' + err.message);
                backToList();
            }
        }
        // Ensure global access for fallbacks
        window.displayStockDetail = displayStockDetail;

        function showLoadingOverlay(ticker) {
            const ov = document.getElementById('loadingOverlay');
            const tk = document.getElementById('loadingTicker');
            if (tk) tk.textContent = ticker ? `(${ticker})` : '';
            if (ov) ov.style.display = 'flex';
        }

        function hideLoadingOverlay() {
            const ov = document.getElementById('loadingOverlay');
            if (ov) ov.style.display = 'none';
        }

        function renderChart(chartData, technical) {
            const canvas = document.getElementById('priceChart');
            const ctx = canvas.getContext('2d');
            if (!chartData || !chartData.dates || !chartData.dates.length) return;

            // Force canvas to match container size before destroying old chart
            const container = canvas.parentElement;
            if (container && container.offsetWidth > 0 && container.offsetHeight > 0) {
                canvas.width = container.offsetWidth;
                canvas.height = container.offsetHeight;
            }

            if (priceChart) {
                priceChart.destroy();
            }

            // Range slicing
            let startIndex = 0;
            const total = chartData.dates.length;
            const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
            const sliceData = (arr) => (Array.isArray(arr) ? arr.slice(startIndex) : []);
            if (currentRange === '1m') {
                startIndex = clamp(total - 21, 0, total);
            } else if (currentRange === '6m') {
                startIndex = clamp(total - 126, 0, total);
            } else if (currentRange === '1y') {
                startIndex = clamp(total - 252, 0, total);
            } else {
                // Skip first 200 days to ensure all MAs (including MA200) are fully calculated
                startIndex = Math.min(200, total);
            }

            const toNum = (v) => {
                const n = Number(v);
                return Number.isFinite(n) ? n : null;
            };

            const labels = chartData.dates.slice(startIndex);
            let priceSeries = (chartData.prices || chartData.close || []).slice(startIndex).map(toNum);
            const ma20 = chartData.ma_20 ? chartData.ma_20.slice(startIndex).map(toNum) : [];
            const ma50 = chartData.ma_50 ? chartData.ma_50.slice(startIndex).map(toNum) : [];
            const ma150 = chartData.ma_150 ? chartData.ma_150.slice(startIndex).map(toNum) : [];
            const ma200 = chartData.ma_200 ? chartData.ma_200.slice(startIndex).map(toNum) : [];
            const maOptimalTrend = chartData.ma_optimal_trend ? chartData.ma_optimal_trend.slice(startIndex).map(toNum) : [];
            const maOptimalTrendPeriod = chartData.ma_optimal_trend_period || 175;
            const open = chartData.open ? chartData.open.slice(startIndex).map(toNum) : [];
            const high = chartData.high ? chartData.high.slice(startIndex).map(toNum) : [];
            const low = chartData.low ? chartData.low.slice(startIndex).map(toNum) : [];
            const close = chartData.close ? chartData.close.slice(startIndex).map(toNum) : [];
            const volumes = chartData.volumes ? chartData.volumes.slice(startIndex).map(toNum) : [];

            // Compute min/max for autoscale (filters out null/NaN)
            const allPrices = [...priceSeries, ...open, ...high, ...low, ...close].filter((v) => Number.isFinite(v));
            const minPrice = allPrices.length ? Math.min(...allPrices) : null;
            const maxPrice = allPrices.length ? Math.max(...allPrices) : null;

            // If priceSeries came back all nulls but we have close data, reuse close as series
            if (!priceSeries.some((v) => v !== null) && close.some((v) => v !== null)) {
                priceSeries = close;
            }

            // Guard against empty/invalid data to avoid blank charts
            const hasLineData = priceSeries.some((v) => v !== null);
            const hasCandleData = open.some((v) => v !== null) && high.some((v) => v !== null) && low.some((v) => v !== null) && close.some((v) => v !== null);
            if (!hasLineData && !hasCandleData) {
                console.warn('No valid chart data to render');
                if (priceChart) { priceChart.destroy(); priceChart = null; }
                const holder = document.getElementById('priceChart')?.parentElement;
                if (holder) {
                    holder.innerHTML = '<div style="color:#e0e0e0;padding:12px;">No chart data available.</div><canvas id="priceChart"></canvas>';
                }
                return;
            }

            const hasFinancial = !!(Chart?.registry?.getController?.('candlestick'));
            const wantsCandle = (currentChartType === 'candlestick');
            const datasets = [];
            let chartType = 'line';

            if (wantsCandle && open.length > 0 && hasCandleData) {
                console.log('🕯️ Candlestick requested - proper OHLC rendering');
                if (navigator.sendBeacon) { navigator.sendBeacon('/client_log', 'RENDER_CANDLESTICK'); }

                // Add dataset with OHLC data for tooltips
                datasets.push({
                    label: 'Candlesticks',
                    type: 'line',
                    data: labels.map((date, i) => ({
                        x: date,
                        y: close[i],
                        o: open[i],
                        h: high[i],
                        l: low[i],
                        c: close[i]
                    })),
                    borderColor: 'transparent',
                    backgroundColor: 'transparent',
                    pointRadius: 0,
                    yAxisID: 'y',
                    order: 10
                });
                
                chartType = 'line';
                console.log('✓ Candlestick rendering prepared:', open.length, 'candles');
            } else {
                if (wantsCandle && !hasCandleData) {
                    console.warn('⚠️ No OHLC data available, falling back to line chart');
                    currentChartType = 'line';
                    const toggleBtn = document.getElementById('chartToggle');
                    if (toggleBtn) { toggleBtn.textContent = '🕯️ Candlestick'; }
                    if (navigator.sendBeacon) { navigator.sendBeacon('/client_log', 'CANDLE_FALLBACK_NO_DATA'); }
                }
                // Line chart mode
                console.log('📈 Rendering line chart');
                datasets.push({
                    label: 'Price',
                    data: priceSeries,
                    borderColor: '#1e88e5',
                    backgroundColor: 'rgba(30, 136, 229, 0.1)',
                    borderWidth: 2,
                    tension: 0.1,
                    fill: true,
                    pointRadius: 0,
                    yAxisID: 'y'
                });
                chartType = 'line';
            }

            // Settings for which MAs to display (defaults to showing all)
            const showMA20 = true;
            const showMA50 = true;
            const showMAOptimalTrend = true;
            
            // Add MAs based on settings and chart type
            if (currentChartType === 'candlestick' && maOptimalTrend && maOptimalTrend.length > 0) {
                // For candlestick mode, show optimal trend MA
                if (showMAOptimalTrend) {
                    datasets.push({
                        label: `MA Trend (${maOptimalTrendPeriod})`,
                        data: labels.map((date, i) => ({x: date, y: maOptimalTrend[i]})),
                        borderColor: 'rgba(76, 175, 80, 0.8)',
                        borderWidth: 2,
                        borderDash: [],
                        fill: false,
                        pointRadius: 0,
                        type: 'line',
                        yAxisID: 'y',
                        order: 4
                    });
                }
                if (showMA50 && ma50 && ma50.length > 0) {
                    datasets.push({
                        label: 'MA 50',
                        data: labels.map((date, i) => ({x: date, y: ma50[i]})),
                        borderColor: 'rgba(255, 179, 0, 0.7)',
                        borderWidth: 2,
                        borderDash: [],
                        fill: false,
                        pointRadius: 0,
                        type: 'line',
                        yAxisID: 'y',
                        order: 5
                    });
                }
                if (showMA20 && ma20 && ma20.length > 0) {
                    datasets.push({
                        label: 'MA 20',
                        data: labels.map((date, i) => ({x: date, y: ma20[i]})),
                        borderColor: 'rgba(156, 39, 176, 0.6)',
                        borderWidth: 1,
                        borderDash: [],
                        fill: false,
                        pointRadius: 0,
                        type: 'line',
                        yAxisID: 'y',
                        order: 6
                    });
                }
            } else {
                // For line chart mode, show traditional MAs
                if (showMA20 && ma20 && ma20.length > 0) {
                    datasets.push({
                        label: 'MA 20',
                        data: labels.map((date, i) => ({x: date, y: ma20[i]})),
                        borderColor: 'rgba(156, 39, 176, 0.6)',
                        borderWidth: 1,
                        borderDash: [],
                        fill: false,
                        pointRadius: 0,
                        type: 'line',
                        yAxisID: 'y',
                        order: 6
                    });
                }
                if (showMA50 && ma50 && ma50.length > 0) {
                    datasets.push({
                        label: 'MA 50',
                        data: labels.map((date, i) => ({x: date, y: ma50[i]})),
                        borderColor: 'rgba(255, 179, 0, 0.7)',
                        borderWidth: 2,
                        borderDash: [],
                        fill: false,
                        pointRadius: 0,
                        type: 'line',
                        yAxisID: 'y',
                        order: 5
                    });
                }
                if (showMAOptimalTrend && maOptimalTrend && maOptimalTrend.length > 0) {
                    datasets.push({
                        label: `MA Trend (${maOptimalTrendPeriod})`,
                        data: labels.map((date, i) => ({x: date, y: maOptimalTrend[i]})),
                        borderColor: 'rgba(76, 175, 80, 0.8)',
                        borderWidth: 2,
                        borderDash: [],
                        fill: false,
                        pointRadius: 0,
                        type: 'line',
                        yAxisID: 'y',
                        order: 4
                    });
                }
            }

            // Add volume bars (semi-transparent)
            if (volumes && volumes.length > 0) {
                datasets.push({
                    label: 'Volume (M)',
                    data: labels.map((date, i) => ({x: date, y: volumes[i]})),
                    type: 'bar',
                    backgroundColor: volumes.map((vol, i) => {
                        // Color based on price movement (green if up, red if down)
                        const isUp = i > 0 && close[i] >= close[i-1];
                        return isUp ? 'rgba(38, 166, 154, 0.3)' : 'rgba(239, 83, 80, 0.3)';
                    }),
                    borderColor: 'transparent',
                    yAxisID: 'volume',
                    order: 20,
                    barPercentage: 0.9,
                    categoryPercentage: 1.0
                });
            }

            const annotations = {};
            if (technical && technical.support_levels) {
                technical.support_levels.forEach((level, i) => {
                    annotations[`support${i}`] = {
                        type: 'line',
                        yMin: level,
                        yMax: level,
                        borderColor: '#00c853',
                        borderWidth: 2,
                        borderDash: [10, 5],
                        label: {
                            display: true,
                            content: `Support: $${level.toFixed(2)}`,
                            position: 'end',
                            backgroundColor: 'rgba(0, 200, 83, 0.8)',
                            color: '#fff',
                            font: { size: 11 }
                        }
                    };
                });
            }

            if (technical && technical.resistance_levels) {
                technical.resistance_levels.forEach((level, i) => {
                    annotations[`resistance${i}`] = {
                        type: 'line',
                        yMin: level,
                        yMax: level,
                        borderColor: '#ff5252',
                        borderWidth: 2,
                        borderDash: [10, 5],
                        label: {
                            display: true,
                            content: `Resistance: $${level.toFixed(2)}`,
                            position: 'end',
                            backgroundColor: 'rgba(255, 82, 82, 0.8)',
                            color: '#fff',
                            font: { size: 11 }
                        }
                    };
                });
            }

            // Choose chart type and data config based on mode
            console.log('Creating chart with type:', chartType, ' financial:', hasFinancial, 'datasets:', datasets.length);

            // Always use category x-axis (labels) for proper bar positioning
            const chartDataConfig = { labels: labels, datasets: datasets };

            // Build scales outside the Chart options to avoid any ternary parse issues
            const ySuggestedMin = (minPrice != null && maxPrice != null) ? (minPrice * 0.99) : undefined;
            const ySuggestedMax = (minPrice != null && maxPrice != null) ? (maxPrice * 1.01) : undefined;

            const scalesConfig = (chartType === 'candlestick')
                ? {
                    x: {
                        type: 'time',
                        grid: { color: 'rgba(42, 63, 95, 0.3)', drawBorder: false },
                        ticks: { color: '#999', maxTicksLimit: 12 }
                    },
                    y: {
                        position: 'left',
                        grid: { color: 'rgba(42, 63, 95, 0.3)', drawBorder: false },
                        suggestedMin: ySuggestedMin,
                        suggestedMax: ySuggestedMax,
                        ticks: {
                            color: '#999',
                            callback: function(value) { return '$' + value.toFixed(0); }
                        }
                    },
                    volume: {
                        position: 'right',
                        grid: { display: false },
                        max: volumes.length > 0 ? Math.max(...volumes.filter(v => Number.isFinite(v))) * 4 : 100,
                        ticks: {
                            color: '#666',
                            callback: function(value) { return value.toFixed(0) + 'M'; }
                        }
                    }
                }
                : {
                    x: {
                        grid: { color: 'rgba(42, 63, 95, 0.3)', drawBorder: false },
                        ticks: { color: '#999', maxTicksLimit: 12 }
                    },
                    y: {
                        position: 'left',
                        grid: { color: 'rgba(42, 63, 95, 0.3)', drawBorder: false },
                        suggestedMin: ySuggestedMin,
                        suggestedMax: ySuggestedMax,
                        ticks: {
                            color: '#999',
                            callback: function(value) { return '$' + value.toFixed(0); }
                        }
                    },
                    volume: {
                        position: 'right',
                        grid: { display: false },
                        max: volumes.length > 0 ? Math.max(...volumes.filter(v => Number.isFinite(v))) * 4 : 100,
                        ticks: {
                            color: '#666',
                            callback: function(value) { return value.toFixed(0) + 'M'; }
                        }
                    }
                };

            // Custom candlestick plugin - draws proper OHLC candlesticks
            const candlestickPlugin = {
                id: 'candlestickPlugin',
                afterDraw(chart) {
                    if (currentChartType !== 'candlestick') return;
                    
                    // Get candlestick data from the chart dataset
                    const candleDataset = chart.data.datasets.find(d => d.label === 'Candlesticks');
                    if (!candleDataset || !candleDataset.data || candleDataset.data.length === 0) return;
                    
                    const ctx = chart.ctx;
                    const xScale = chart.scales.x;
                    const yScale = chart.scales.y;
                    const chartArea = chart.chartArea;
                    
                    ctx.save();
                    
                    // Calculate adaptive body width based on actual pixel spacing between points
                    const dataPoints = candleDataset.data;
                    const numCandles = dataPoints.length;
                    let pixelSpacing = 10; // default
                    if (numCandles > 1) {
                        const x0 = xScale.getPixelForValue(dataPoints[0].x);
                        const x1 = xScale.getPixelForValue(dataPoints[1].x);
                        pixelSpacing = Math.abs(x1 - x0);
                    }
                    const spacing = 2; // 2px spacing between candles
                    const calculatedWidth = pixelSpacing - spacing;
                    const bodyWidth = Math.max(3, Math.min(20, calculatedWidth)); // Range: 3px to 20px
                    
                    // Draw each candlestick (wick + body)
                    for (let i = 0; i < dataPoints.length; i++) {
                        const point = dataPoints[i];
                        const o = point.o, h = point.h, l = point.l, c = point.c;
                        
                        // Skip if any value is null/undefined
                        if (o == null || h == null || l == null || c == null) continue;
                        
                        // Pixel coordinates
                        const xPixel = xScale.getPixelForValue(point.x);
                        const yHigh = yScale.getPixelForValue(h);
                        const yLow = yScale.getPixelForValue(l);
                        const yOpen = yScale.getPixelForValue(o);
                        const yClose = yScale.getPixelForValue(c);
                        
                        const color = c >= o ? '#26a69a' : '#ef5350';  // Teal up, Red down
                        const wickWidth = 1;
                        
                        // Draw wick (thin vertical line from low to high)
                        ctx.strokeStyle = color;
                        ctx.lineWidth = wickWidth;
                        ctx.beginPath();
                        ctx.moveTo(xPixel, yHigh);
                        ctx.lineTo(xPixel, yLow);
                        ctx.stroke();
                        
                        // Draw body (thick rectangle from open to close)
                        const bodyTop = Math.min(yOpen, yClose);
                        const bodyBottom = Math.max(yOpen, yClose);
                        const bodyHeight = Math.max(bodyBottom - bodyTop, 1);  // Min 1px height
                        
                        ctx.fillStyle = color;
                        ctx.fillRect(xPixel - bodyWidth/2, bodyTop, bodyWidth, bodyHeight);
                        
                        // Body border
                        ctx.strokeStyle = color;
                        ctx.lineWidth = 1;
                        ctx.strokeRect(xPixel - bodyWidth/2, bodyTop, bodyWidth, bodyHeight);
                    }
                    
                    ctx.restore();
                    console.log('[Candlesticks] Rendered via custom plugin');
                }
            };

            // Register plugin with Chart.js
            if (!window.candlestickPluginRegistered) {
                Chart.register(candlestickPlugin);
                window.candlestickPluginRegistered = true;
            }

            priceChart = new Chart(ctx, {
                type: chartType,
                data: chartDataConfig,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { duration: 0 },  // Disable animation for instant render
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        candlestickPlugin: {},  // Enable our custom plugin
                        legend: {
                            display: true,
                            labels: { color: '#e0e0e0', font: { size: 12 }, usePointStyle: true }
                        },
                        tooltip: {
                            backgroundColor: 'rgba(26, 31, 46, 0.95)',
                            titleColor: '#fff',
                            bodyColor: '#e0e0e0',
                            borderColor: '#2a3f5f',
                            borderWidth: 1,
                            displayColors: true,
                            callbacks: {
                                label: function(context) {
                                    if (currentChartType === 'candlestick') {
                                        const v = context.raw || {};
                                        const o = (v.o != null) ? v.o.toFixed(2) : '—';
                                        const h = (v.h != null) ? v.h.toFixed(2) : '—';
                                        const l = (v.l != null) ? v.l.toFixed(2) : '—';
                                        const c = (v.c != null) ? v.c.toFixed(2) : '—';
                                        return `O:${o} H:${h} L:${l} C:${c}`;
                                    }
                                    let label = context.dataset.label || '';
                                    if (label) {
                                        label += ': ';
                                    }
                                    if (context.parsed.y !== null) {
                                        label += '$' + context.parsed.y.toFixed(2);
                                    }
                                    return label;
                                }
                            }
                        },
                        annotation: { annotations: annotations },
                        zoom: {
                            pan: { enabled: true, mode: 'x', modifierKey: 'shift' },
                            zoom: {
                                wheel: { enabled: true, modifierKey: null },
                                pinch: { enabled: true },
                                mode: 'x'
                            }
                        }
                    },
                    scales: scalesConfig
                }
            });
            console.log('Chart rendered:', currentChartType, 'range=', currentRange, 'points=', labels.length, 'datasets=', datasets.length);
        }

        // Robust populate: writes all fields and retries to avoid timing issues
        function forcePopulateDetail(data) {
            try {
                console.log('▶ forcePopulateDetail start', data && data.ticker, data);
                if (window.logOnPage) { window.logOnPage('forcePopulateDetail ' + (data && data.ticker ? data.ticker : 'unknown')); }
                const apply = (payload) => {
                    try {
                        const screen = payload.screen_data || {};
                        const tech = payload.technical_analysis || {};
                        const fund = payload.fundamentals || {};
                        const ai = payload.ai_analysis || {};
                        const chartData = payload.chart_data || {};
                        const n = (v) => (typeof v === 'number' ? v : (v != null ? Number(v) : NaN));
                        const set = (id, val) => {
                            const el = document.getElementById(id);
                            if (el) {
                                el.textContent = val;
                            } else {
                                console.warn('⚠ element not found:', id);
                            }
                        };

                        // Header
                        set('detailTicker', payload.ticker || '-');
                        set('detailCompanyName', fund.company_name || payload.ticker || '-');
                        const price = n(screen.price);
                        set('detailPrice', isFinite(price) ? '$' + price.toFixed(2) : '$-');

                        // Technicals
                        set('tech-price', isFinite(price) ? '$' + price.toFixed(2) : '-');
                        const maV = n(screen.ma_value);
                        // Use the chart's calculated optimal MA period if available
                        const optimalMaPeriod = chartData.ma_optimal_trend_period || screen.optimal_ma;
                        set('tech-ma', optimalMaPeriod ? `${optimalMaPeriod}-day @ $${isFinite(maV) ? maV.toFixed(2) : '-'}` : '-');
                        const rsi = n(tech.rsi);
                        set('tech-rsi', isFinite(rsi) ? rsi.toFixed(1) : '-');
                        const conv = n(screen.conviction);
                        set('tech-conviction', isFinite(conv) ? conv.toFixed(1) + '/10' : '-');
                        const g1 = n(screen.gain_1m), g3 = n(screen.gain_3m), g1y = n(screen.gain_1y);
                        set('returns-1m', isFinite(g1) ? g1.toFixed(2) + '%' : '-');
                        set('returns-3m', isFinite(g3) ? g3.toFixed(2) + '%' : '-');
                        set('returns-1y', isFinite(g1y) ? g1y.toFixed(2) + '%' : '-');

                        // If values still show '-', force rebuild of the metric cards
                        try {
                            const tp = document.getElementById('tech-price');
                            if (tp && (tp.textContent === '-' || tp.textContent === '')) {
                                const card = tp.closest('.card');
                                if (card) {
                                    const optimalMaPeriod = chartData.ma_optimal_trend_period || screen.optimal_ma;
                                    card.innerHTML = `
                                        <div class="metric-row"><span class="metric-label">Current Price</span><span class="metric-value">${isFinite(price)?('$'+price.toFixed(2)):'-'}</span></div>
                                        <div class="metric-row"><span class="metric-label">Optimal MA</span><span class="metric-value">${optimalMaPeriod? (optimalMaPeriod+'-day @ $'+(isFinite(maV)?maV.toFixed(2):'-')) : '-'}</span></div>
                                        <div class="metric-row"><span class="metric-label">RSI (14)</span><span class="metric-value">${isFinite(rsi)? rsi.toFixed(1) : '-'}</span></div>
                                        <div class="metric-row"><span class="metric-label">Volume Ratio</span><span class="metric-value">${isFinite(n(tech.volatility))? n(tech.volatility).toFixed(1)+'%' : '-'}</span></div>
                                        <div class="metric-row"><span class="metric-label">Conviction Score</span><span class="metric-value">${isFinite(conv)? conv.toFixed(1)+'/10' : '-'}</span></div>
                                        <div class="metric-row"><span class="metric-label">1M Returns</span><span class="metric-value">${isFinite(g1)? g1.toFixed(2)+'%' : '-'}</span></div>
                                        <div class="metric-row"><span class="metric-label">3M Returns</span><span class="metric-value">${isFinite(g3)? g3.toFixed(2)+'%' : '-'}</span></div>
                                        <div class="metric-row"><span class="metric-label">1Y Returns</span><span class="metric-value">${isFinite(g1y)? g1y.toFixed(2)+'%' : '-'}</span></div>
                                    `;
                                }
                            }
                        } catch (e) { console.warn('tech card rebuild error', e); }

                        // Fundamentals
                        const pe = n(fund.pe_ratio);
                        set('fund-pe', isFinite(pe) ? pe.toFixed(2) : (fund.pe_ratio || '-'));
                        const mc = n(fund.market_cap);
                        set('fund-mc', isFinite(mc) ? '$' + (mc / 1e9).toFixed(2) + 'B' : (fund.market_cap || '-'));
                        set('fund-sector', fund.sector || '-');
                        set('fund-industry', fund.industry || '-');
                        const roe = n(fund.roe);
                        set('fund-roe', isFinite(roe) ? (roe * 100).toFixed(2) + '%' : (fund.roe || '-'));
                        const pm = n(fund.profit_margin);
                        set('fund-pm', isFinite(pm) ? (pm * 100).toFixed(2) + '%' : (fund.profit_margin || '-'));
                        const de = n(fund.debt_to_equity);
                        set('fund-de', isFinite(de) ? de.toFixed(2) : (fund.debt_to_equity || '-'));

                        // Fundamentals fallback rebuild if still dashes
                        try {
                            const fpe = document.getElementById('fund-pe');
                            if (fpe && (fpe.textContent === '-' || fpe.textContent === '')) {
                                const fcard = fpe.closest('.card');
                                if (fcard) {
                                    fcard.innerHTML = `
                                        <div class="metric-row"><span class="metric-label">P/E Ratio</span><span class="metric-value">${isFinite(pe)? pe.toFixed(2) : '-'}</span></div>
                                        <div class="metric-row"><span class="metric-label">Market Cap</span><span class="metric-value">${isFinite(mc)? ('$'+(mc/1e9).toFixed(2)+'B') : '-'}</span></div>
                                        <div class="metric-row"><span class="metric-label">Sector</span><span class="metric-value">${fund.sector || '-'}</span></div>
                                        <div class="metric-row"><span class="metric-label">Industry</span><span class="metric-value">${fund.industry || '-'}</span></div>
                                        <div class="metric-row"><span class="metric-label">ROE</span><span class="metric-value">${isFinite(roe)? (roe*100).toFixed(2)+'%' : '-'}</span></div>
                                        <div class="metric-row"><span class="metric-label">Profit Margin</span><span class="metric-value">${isFinite(pm)? (pm*100).toFixed(2)+'%' : '-'}</span></div>
                                        <div class="metric-row"><span class="metric-label">Debt/Equity</span><span class="metric-value">${isFinite(de)? de.toFixed(2) : '-'}</span></div>
                                    `;
                                }
                            }
                        } catch (e) { console.warn('fund card rebuild error', e); }

                        // AI / Catalysts
                        const ten = n(ai.ten_x_score);
                        set('ai-score', isFinite(ten) ? ten.toFixed(1) : '-');
                        if (ai.ai_reasoning && Array.isArray(ai.ai_reasoning)) {
                            const el = document.getElementById('ai-reasoning');
                            if (el) el.innerHTML = ai.ai_reasoning.map((r) => `<li>${r}</li>`).join('');
                        }
                        if (ai.bull_case && Array.isArray(ai.bull_case)) {
                            const el = document.getElementById('bull-case');
                            if (el) el.innerHTML = ai.bull_case.map((r) => `<li>${r}</li>`).join('');
                        }
                        if (ai.bear_case && Array.isArray(ai.bear_case)) {
                            const el = document.getElementById('bear-case');
                            if (el) el.innerHTML = ai.bear_case.map((r) => `<li>${r}</li>`).join('');
                        }
                        if (ai.catalysts && Array.isArray(ai.catalysts)) {
                            const el = document.getElementById('catalysts');
                            if (el) el.innerHTML = ai.catalysts.map((r) => `<li>${r}</li>`).join('');
                        }

                        // News
                        if (payload.news && payload.news.length) {
                            const el = document.getElementById('newsDiv');
                            if (el) {
                                const newsHtml = payload.news.map((n) => {
                                    const link = n.link && n.link.length > 0 ? `<a href="${n.link}" target="_blank" style="color:#1e88e5;text-decoration:none;font-weight:600">${n.title}</a>` : `<span style="color:#e0e0e0">${n.title}</span>`;
                                    return `<div class="news-item" style="padding:10px 0;border-bottom:1px solid #2a3f5f;font-size:0.9em">${link}<div style="color:#999;font-size:0.85em;margin-top:4px">${n.source||'Source unknown'} • ${n.published||''}</div></div>`;
                                }).join('');
                                el.innerHTML = newsHtml || '<p style="color:#666">No recent news available</p>';
                            }
                        } else {
                            const el = document.getElementById('newsDiv');
                            if (el) el.innerHTML = '<p style="color:#666">No recent news available</p>';
                        }

                        // Chart - preserve current chart type
                        try {
                            if (payload.chart_data && payload.chart_data.dates && payload.chart_data.dates.length) {
                                currentChartData = payload.chart_data;
                                currentTechnical = tech;
                                // Don't render here - will render once after all retries complete
                            }
                        } catch (e) { console.error('forcePopulate chart error', e); }
                    } catch (e) {
                        console.error('apply populate error', e);
                    }
                };

                // apply immediately and with retries to avoid any timing issues
                apply(data);
                setTimeout(() => apply(data), 150);
                setTimeout(() => apply(data), 400);
                
                // Final apply with chart render after all DOM updates complete
                setTimeout(() => {
                    apply(data);
                    if (currentChartData && currentTechnical) {
                        renderChart(currentChartData, currentTechnical);
                    }
                }, 900);

                // Continuous enforcement loop for a short window to defeat any late overwrites
                try {
                    let ticks = 0;
                    if (window.detailPopulateIntervalId) {
                        clearInterval(window.detailPopulateIntervalId);
                        window.detailPopulateIntervalId = null;
                    }
                    window.detailPopulateIntervalId = setInterval(() => {
                        try { apply(data); } catch(e) { }
                        ticks++;
                        if (ticks >= 25) { // ~6s at 240ms
                            clearInterval(window.detailPopulateIntervalId);
                            window.detailPopulateIntervalId = null;
                            console.log('⏹ populate enforcement loop stopped');
                        }
                    }, 240);
                    console.log('▶ populate enforcement loop started');
                } catch(e) { console.warn('populate loop start error', e); }

                console.log('✅ forcePopulateDetail applied (multi-pass)');
            } catch (e) {
                console.error('forcePopulateDetail error', e);
            }
        }

        function backToList() {
            console.log('Switching back to list view');
            const detailView = document.getElementById('detailView');
            const listView = document.getElementById('listView');
            if (window.detailPopulateIntervalId) { try { clearInterval(window.detailPopulateIntervalId); window.detailPopulateIntervalId = null; } catch(_){} }
            
            if (detailView) {
                detailView.classList.remove('active');
                detailView.style.display = 'none';
            }
            if (listView) {
                listView.classList.add('active');
                listView.style.display = 'block';
            }
            
            window.scrollTo(0, 0);
        }
        // Ensure global access from inline handlers
        window.backToList = backToList;

        // Define viewStockDetail - SIMPLE, NO STUBS
        window.viewStockDetail = function(ticker) {
            console.log('viewStockDetail called:', ticker);
            if (navigator.sendBeacon) { try { navigator.sendBeacon('/client_log', 'VIEW_STOCK_DETAIL_CALL:'+ticker); } catch(_){} }
            
            const listView = document.getElementById('listView');
            const detailView = document.getElementById('detailView');
            
            if (listView) {
                listView.classList.remove('active');
                listView.style.display = 'none';
            }
            if (detailView) {
                detailView.classList.add('active');
                detailView.style.display = 'block';
            }
            
            window.scrollTo(0, 0);
            showLoadingOverlay(ticker);
            
            fetch('/api/stock/' + ticker)
                .then(r => r.json())
                .then(data => {
                    if (data.error) {
                        alert('Error: ' + data.error);
                        hideLoadingOverlay();
                        backToList();
                        return;
                    }
                    displayStockDetail(data);
                    // Defensive second-pass populate in case any block failed
                    setTimeout(() => { try { forcePopulateDetail(data); } catch(e) { console.error('forcePopulateDetail fail', e); } }, 0);
                    if (navigator.sendBeacon) { try { navigator.sendBeacon('/client_log', 'VIEW_STOCK_DETAIL_DONE:'+ticker); } catch(_){} }
                })
                .catch(err => {
                    alert('Error loading stock: ' + err.message);
                    if (navigator.sendBeacon) { try { navigator.sendBeacon('/client_log', 'VIEW_STOCK_DETAIL_FETCH_ERROR:'+err.message); } catch(_){} }
                    hideLoadingOverlay();
                    backToList();
                });
        };
        console.log('viewStockDetail defined:', typeof window.viewStockDetail);
    </script>
</body>
</html>'''


@app.route('/client_log', methods=['POST'])
def client_log():
    try:
        data = request.get_data(as_text=True)
        print(f"CLIENT_LOG: {data}")
        return ('', 204)
    except Exception as e:
        print(f"CLIENT_LOG error: {e}")
        return (str(e), 500)

@app.route('/test_candlestick')
def test_candlestick():
    return '''<!DOCTYPE html>
<html>
<head>
    <title>Candlestick Test</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/luxon@2.5.2"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1.2.0"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-financial@0.1.1"></script>
</head>
<body style="background: #1a1a1a; color: #fff; padding: 20px; font-family: monospace;">
    <h1>🕯️ Candlestick Plugin Validation</h1>
    <button id="testBtn" style="padding: 10px 20px; font-size: 16px; margin-bottom: 20px; cursor: pointer;">Test Rendering</button>
    <div id="result" style="margin-bottom: 20px; line-height: 1.6;"></div>
    <div style="width: 800px; height: 400px; background: #0f1419; padding: 20px; border-radius: 8px;">
        <canvas id="chart"></canvas>
    </div>
    
    <script>
        let testChart = null; // keep reference so we can destroy before reusing canvas

        document.getElementById('testBtn').addEventListener('click', function() {
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = '<div style="color: yellow;">🔄 Testing...</div>';
            
            try {
                if (typeof Chart === 'undefined') {
                    resultDiv.innerHTML = '<div style="color: red;">✗ Chart.js not loaded</div>';
                    return;
                }
                
                try {
                    const fin = window['chartjs-chart-financial'];
                    if (fin && Chart && Chart.register) {
                        const { CandlestickController, OhlcController, CandlestickElement, OhlcElement } = fin;
                        Chart.register(CandlestickController, OhlcController, CandlestickElement, OhlcElement);
                        resultDiv.innerHTML += '<div style="color: green;">✓ Controllers registered manually</div>';
                    }
                } catch (regErr) {
                    resultDiv.innerHTML += '<div style="color: orange;">⚠ Registration error: ' + regErr.message + '</div>';
                }

                resultDiv.innerHTML += '<div style="color: green;">✓ Chart.js ' + Chart.version + ' loaded</div>';
                const controllersObj = (Chart.registry && Chart.registry.controllers && Chart.registry.controllers.items) ? Chart.registry.controllers.items : {};
                const controllerNames = Object.keys(controllersObj);
                resultDiv.innerHTML += '<div style="color: #888;">Controllers: ' + (controllerNames.length ? controllerNames.join(', ') : '(empty)') + '</div>';

                const hasCandle = !!(Chart.registry && Chart.registry.getController && Chart.registry.getController('candlestick'));
                if (hasCandle) {
                    resultDiv.innerHTML += '<div style="color: green;">✓ Candlestick controller registered!</div>';
                } else {
                    resultDiv.innerHTML += '<div style="color: red;">✗ Candlestick controller NOT found</div>';
                    return;
                }
                
                const ohlcData = [
                    { x: '2024-01-01', o: 100, h: 110, l: 95, c: 105 },
                    { x: '2024-01-02', o: 105, h: 115, l: 103, c: 108 },
                    { x: '2024-01-03', o: 108, h: 112, l: 102, c: 103 },
                    { x: '2024-01-04', o: 103, h: 108, l: 98, c: 106 },
                    { x: '2024-01-05', o: 106, h: 118, l: 105, c: 115 }
                ];
                
                const ctx = document.getElementById('chart').getContext('2d');

                // Destroy any existing chart instance on this canvas to avoid reuse errors
                try {
                    if (testChart) {
                        testChart.destroy();
                        testChart = null;
                    } else if (Chart.getChart) {
                        const existing = Chart.getChart('chart');
                        if (existing) { existing.destroy(); }
                    }
                } catch (destroyErr) {
                    console.warn('Chart destroy warning', destroyErr);
                }

                testChart = new Chart(ctx, {
                    type: 'candlestick',
                    data: {
                        datasets: [{
                            label: 'OHLC Test Data',
                            data: ohlcData
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: false }
                        }
                    }
                });
                
                resultDiv.innerHTML += '<div style="color: lime; font-weight: bold; font-size: 18px;">✓✓✓ SUCCESS! Candlestick chart rendered!</div>';
                
            } catch (e) {
                resultDiv.innerHTML += '<div style="color: red;">✗ Error: ' + e.message + '</div>';
                console.error(e);
            }
        });
        
        setTimeout(() => document.getElementById('testBtn').click(), 500);
    </script>
</body>
</html>'''

@app.route('/detail_ssr/<ticker>')
def detail_ssr(ticker):
        """True server-side rendering - all values populated on server."""
        try:
                hunter = TenXHunter()
                analysis = hunter.get_detailed_analysis(ticker.upper())
                analysis = clean_nan_values(analysis)
                
                # Extract data with safe access
                screen = analysis.get('screen_data', {})
                tech = analysis.get('technical_analysis', {})
                fund = analysis.get('fundamentals', {})
                news = analysis.get('news', [])
                chart_data = analysis.get('chart_data', {})
                
                # Helper functions for formatting
                def fmt_price(v, prefix='$'):
                    if v is None or (isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf'))):
                        return prefix + '-'
                    try:
                        return prefix + str(float(v))[:8]
                    except:
                        return prefix + '-'
                
                def fmt_pct(v):
                    if v is None or (isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf'))):
                        return '-'
                    try:
                        return str(float(v))[:6] + '%'
                    except:
                        return '-'
                
                def fmt_num(v, decimals=2):
                    if v is None or (isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf'))):
                        return '-'
                    try:
                        return str(round(float(v), decimals))[:8]
                    except:
                        return '-'
                
                # Format all values on server
                price = screen.get('price', 0)
                price_str = fmt_price(price)
                
                optimal_ma = screen.get('optimal_ma', '-')
                ma_value = screen.get('ma_value', 0)
                ma_str = f"{optimal_ma}-day @ {fmt_price(ma_value)}" if optimal_ma != '-' else '-'
                
                rsi = tech.get('rsi', None)
                rsi_str = fmt_num(rsi, 1) if rsi else '-'
                
                conviction = screen.get('conviction', None)
                conviction_str = f"{fmt_num(conviction, 1)}/10" if conviction else '-'
                
                # Returns
                gain_1m = fmt_pct(screen.get('gain_1m'))
                gain_3m = fmt_pct(screen.get('gain_3m'))
                gain_1y = fmt_pct(screen.get('gain_1y'))
                
                # Fundamentals
                pe = fmt_num(fund.get('pe_ratio'), 2) if fund.get('pe_ratio') else '-'
                market_cap = fund.get('market_cap')
                mc_str = '-'
                if market_cap:
                    try:
                        mc_val = float(market_cap) / 1e9
                        mc_str = f"${mc_val:.2f}B"
                    except:
                        mc_str = '-'
                
                sector = fund.get('sector', '-')
                industry = fund.get('industry', '-')
                
                roe = fund.get('roe')
                roe_str = '-'
                if roe:
                    try:
                        roe_str = f"{float(roe)*100:.2f}%"
                    except:
                        roe_str = '-'
                
                profit_margin = fund.get('profit_margin')
                pm_str = '-'
                if profit_margin:
                    try:
                        pm_str = f"{float(profit_margin)*100:.2f}%"
                    except:
                        pm_str = '-'
                
                de = fmt_num(fund.get('debt_to_equity'), 2)
                
                # Company name
                company_name = fund.get('company_name') or screen.get('company_name') or analysis.get('ticker', 'N/A')
                
                # News HTML
                news_html = ''
                if news:
                    for article in news[:5]:
                        link = article.get('link', '#')
                        title = article.get('title', 'No title')
                        source = article.get('source', '')
                        news_html += f'<div><a href="{link}" target="_blank">{title}</a> <span style="color:#999">{source}</span></div>'
                else:
                    news_html = '<em>No recent news</em>'
                
                import json
                data_json = json.dumps(analysis)
                
                html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Detail - TICKER_PLACEHOLDER</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body{font-family:Segoe UI,Roboto,Arial;background:#0f1419;color:#e0e0e0;margin:0;padding:20px}
        .card{background:#1a1f2e;padding:16px;border-radius:8px;margin:10px 0;border:1px solid #2a3f5f}
        .grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
        .metric-row{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #2a3f5f}
        .metric-label{color:#999}
        .metric-value{font-weight:700}
        .chart-container{height:420px;border:1px solid #2a3f5f;border-radius:8px;padding:10px;background:#0f1419}
        a{color:#1e88e5;text-decoration:none}
    </style>
</head>
<body>
    <h1>Detail - <span id="detailTicker">TICKER_PLACEHOLDER</span></h1>
    <div class="card">
        <div style="display:flex;justify-content:space-between">
            <div>
                <div id="detailCompanyName">COMPANY_PLACEHOLDER</div>
            </div>
            <div id="detailPrice">PRICE_PLACEHOLDER</div>
        </div>
    </div>

    <div class="grid-2">
        <div class="card">
            <h3>Technical Analysis</h3>
            <div class="metric-row"><span class="metric-label">Current Price</span><span class="metric-value" id="tech-price">TECH_PRICE_PLACEHOLDER</span></div>
            <div class="metric-row"><span class="metric-label">Optimal MA</span><span class="metric-value" id="tech-ma">MA_PLACEHOLDER</span></div>
            <div class="metric-row"><span class="metric-label">RSI (14)</span><span class="metric-value" id="tech-rsi">RSI_PLACEHOLDER</span></div>
            <div class="metric-row"><span class="metric-label">Conviction</span><span class="metric-value" id="tech-conviction">CONVICTION_PLACEHOLDER</span></div>
            <div class="metric-row"><span class="metric-label">1M Returns</span><span class="metric-value" id="returns-1m">GAIN_1M_PLACEHOLDER</span></div>
            <div class="metric-row"><span class="metric-label">3M Returns</span><span class="metric-value" id="returns-3m">GAIN_3M_PLACEHOLDER</span></div>
            <div class="metric-row"><span class="metric-label">1Y Returns</span><span class="metric-value" id="returns-1y">GAIN_1Y_PLACEHOLDER</span></div>
        </div>
        <div class="card">
            <h3>Fundamentals</h3>
            <div class="metric-row"><span class="metric-label">P/E</span><span class="metric-value" id="fund-pe">PE_PLACEHOLDER</span></div>
            <div class="metric-row"><span class="metric-label">Market Cap</span><span class="metric-value" id="fund-mc">MC_PLACEHOLDER</span></div>
            <div class="metric-row"><span class="metric-label">Sector</span><span class="metric-value" id="fund-sector">SECTOR_PLACEHOLDER</span></div>
            <div class="metric-row"><span class="metric-label">Industry</span><span class="metric-value" id="fund-industry">INDUSTRY_PLACEHOLDER</span></div>
            <div class="metric-row"><span class="metric-label">ROE</span><span class="metric-value" id="fund-roe">ROE_PLACEHOLDER</span></div>
            <div class="metric-row"><span class="metric-label">Profit Margin</span><span class="metric-value" id="fund-pm">PM_PLACEHOLDER</span></div>
            <div class="metric-row"><span class="metric-label">Debt/Equity</span><span class="metric-value" id="fund-de">DE_PLACEHOLDER</span></div>
        </div>
    </div>

    <div class="card"><h3>News</h3><div id="newsDiv">NEWS_PLACEHOLDER</div></div>

    <div class="card"><h3>Price Chart</h3><div class="chart-container"><canvas id="priceChart"></canvas></div></div>

    <script>
        const data = DATA_JSON_PLACEHOLDER;
        const screen = data.screen_data || {};
        const tech = data.technical_analysis || {};
        const fund = data.fundamentals || {};
        const chart_data = data.chart_data || {};
        
        // Chart rendering
        (function render(){
            try{
                const ctx = document.getElementById('priceChart').getContext('2d');
                const datasets=[];
                
                if (chart_data.dates && chart_data.close) {
                    datasets.push({ 
                        label:'Price', 
                        data:chart_data.close, 
                        borderColor:'#1e88e5', 
                        backgroundColor:'rgba(30,136,229,0.1)', 
                        borderWidth:2, 
                        tension:0.1, 
                        fill:true, 
                        pointRadius:0, 
                        yAxisID:'y' 
                    });
                }
                
                if (chart_data.ma_50 && chart_data.ma_50.length) {
                    datasets.push({ 
                        label:'MA 50', 
                        data:chart_data.ma_50, 
                        borderColor:'#ffb300', 
                        borderWidth:1.5, 
                        borderDash:[5,5], 
                        fill:false, 
                        pointRadius:0, 
                        yAxisID:'y' 
                    });
                }
                
                if (chart_data.ma_200 && chart_data.ma_200.length) {
                    datasets.push({ 
                        label:'MA 200', 
                        data:chart_data.ma_200, 
                        borderColor:'#e53935', 
                        borderWidth:1.5, 
                        borderDash:[5,5], 
                        fill:false, 
                        pointRadius:0, 
                        yAxisID:'y' 
                    });
                }
                
                new Chart(ctx, { 
                    type: 'line', 
                    data:{ 
                        labels:chart_data.dates||[], 
                        datasets 
                    }, 
                    options:{ 
                        responsive:true, 
                        maintainAspectRatio:false, 
                        plugins:{ 
                            legend:{ 
                                display:true, 
                                labels:{ color:'#e0e0e0', usePointStyle:true } 
                            }, 
                            tooltip:{ 
                                backgroundColor:'rgba(26,31,46,0.95)', 
                                titleColor:'#fff', 
                                bodyColor:'#e0e0e0' 
                            } 
                        }, 
                        scales:{ 
                            x:{ ticks:{ color:'#999' }, grid:{ color:'rgba(42,63,95,0.3)' } }, 
                            y:{ position:'right', ticks:{ color:'#999' }, grid:{ color:'rgba(42,63,95,0.3)' } } 
                        } 
                    }
                });
            }catch(e){ 
                console.error('Chart error', e); 
            }
        })();
    </script>
</body>
</html>'''
                
                # Replace all placeholders with actual values
                html = html.replace('TICKER_PLACEHOLDER', ticker.upper())
                html = html.replace('COMPANY_PLACEHOLDER', company_name)
                html = html.replace('PRICE_PLACEHOLDER', price_str)
                html = html.replace('TECH_PRICE_PLACEHOLDER', price_str)
                html = html.replace('MA_PLACEHOLDER', ma_str)
                html = html.replace('RSI_PLACEHOLDER', rsi_str)
                html = html.replace('CONVICTION_PLACEHOLDER', conviction_str)
                html = html.replace('GAIN_1M_PLACEHOLDER', gain_1m)
                html = html.replace('GAIN_3M_PLACEHOLDER', gain_3m)
                html = html.replace('GAIN_1Y_PLACEHOLDER', gain_1y)
                html = html.replace('PE_PLACEHOLDER', pe)
                html = html.replace('MC_PLACEHOLDER', mc_str)
                html = html.replace('SECTOR_PLACEHOLDER', sector)
                html = html.replace('INDUSTRY_PLACEHOLDER', industry)
                html = html.replace('ROE_PLACEHOLDER', roe_str)
                html = html.replace('PM_PLACEHOLDER', pm_str)
                html = html.replace('DE_PLACEHOLDER', de)
                html = html.replace('NEWS_PLACEHOLDER', news_html)
                html = html.replace('DATA_JSON_PLACEHOLDER', data_json)
                
                return html
        except Exception as e:
                import traceback
                error_msg = f"Error rendering SSR detail: {str(e)}"
                traceback.print_exc()
                return error_msg, 500

@app.route('/direct_test')
def direct_test():
    return open('direct_candle_test.html').read()

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 10x Hunter API Server")
    print("="*70)
    print("\n✓ Setup complete!")
    print("✓ Open browser: http://localhost:5000")
    print("✓ Click 'Run Weekly Screen' to scan real stocks")
    print("\n" + "="*70 + "\n")
    app.run(debug=True, port=5000)
