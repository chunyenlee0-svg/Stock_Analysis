import os
import json
import datetime
import base64
import io
import re
import time
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from google import genai
from google.genai import types
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from jinja2 import Template
from pydantic import BaseModel, Field
from typing import List

# 全域設定：強制設定 requests 的預設連線超時時間為 15 秒，避免 yfinance 或其它 API 請求無限期卡死
original_session_request = requests.Session.request
def patched_session_request(self, method, url, *args, **kwargs):
    if 'timeout' not in kwargs or kwargs['timeout'] is None:
        kwargs['timeout'] = 15
    return original_session_request(self, method, url, *args, **kwargs)
requests.Session.request = patched_session_request

# ==========================================
# 1. 數據獲取與統計計算模組
# ==========================================

def fetch_stock_data(symbol):
    """
    獲取台股與全球指標歷史數據（過去兩年）
    """
    special_mappings = {
        'XAUD': 'GC=F',
        'NTBD': 'USDTWD=X',
        'DINI': 'DX-Y.NYB'
    }
    
    if symbol in special_mappings:
        ticker_symbol = special_mappings[symbol]
        print(f"偵測到全球指標標的 {symbol}，使用 YFinance 代號 {ticker_symbol} 下載...")
    elif not (symbol.endswith('.TW') or symbol.endswith('.TWO')):
        ticker_symbol = f"{symbol}.TW"
    else:
        ticker_symbol = symbol
        
    print(f"正在下載 {ticker_symbol} 的歷史數據...")
    df = yf.download(ticker_symbol, period="2y")
    
    if df.empty and symbol not in special_mappings:
        if not (symbol.endswith('.TW') or symbol.endswith('.TWO')):
            ticker_symbol = f"{symbol}.TWO"
            print(f"上市找不到，嘗試下載上櫃 {ticker_symbol}...")
            df = yf.download(ticker_symbol, period="2y")
            
    if df.empty:
        raise ValueError(f"無法取得股票代號 {symbol} 的數據。")
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    return df, ticker_symbol

def fetch_stock_news(ticker_symbol):
    """
    從 yfinance 獲取股票相關最新新聞
    """
    ticker = yf.Ticker(ticker_symbol)
    news = ticker.news
    formatted_news = []
    if news:
        for item in news[:5]:
            formatted_news.append({
                'title': item.get('title', '無標題'),
                'publisher': item.get('publisher', '未知媒體'),
                'link': item.get('link', '#'),
                'summary': item.get('summary', '')
            })
    return formatted_news

def calculate_technical_indicators(df):
    """
    計算移動平均線 (MA) 與常用的技術指標 (KD, RSI, MACD)
    """
    df = df.copy()
    
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA120'] = df['Close'].rolling(window=120).mean()
    df['MA240'] = df['Close'].rolling(window=240).mean()
    
    low_9 = df['Low'].rolling(window=9).min()
    high_9 = df['High'].rolling(window=9).max()
    divisor = high_9 - low_9
    rsv = ((df['Close'] - low_9) / divisor * 100).fillna(50)
    
    k_values = []
    d_values = []
    current_k = 50.0
    current_d = 50.0
    for val in rsv:
        current_k = (2/3) * current_k + (1/3) * val
        current_d = (2/3) * current_d + (1/3) * current_k
        k_values.append(current_k)
        d_values.append(current_d)
        
    df['K'] = k_values
    df['D'] = d_values
    
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi_series = 100 - (100 / (1 + rs))
        return rsi_series.fillna(50)
        
    df['RSI6'] = calculate_rsi(df['Close'], 6)
    df['RSI14'] = calculate_rsi(df['Close'], 14)
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = ema12 - ema26
    df['DEM'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['OSC'] = df['DIF'] - df['DEM']
    
    # 5日均量
    df['Volume_5'] = df['Volume'].rolling(window=5).mean().fillna(df['Volume'])
    
    # 布林通道 (Bollinger Bands)
    df['BB_Middle'] = df['MA20']
    std20 = df['Close'].rolling(window=20).std().fillna(0)
    df['BB_Upper'] = df['BB_Middle'] + 2 * std20
    df['BB_Lower'] = df['BB_Middle'] - 2 * std20
    
    # 乖離率 (BIAS)
    df['BIAS5'] = (((df['Close'] - df['MA5']) / df['MA5']) * 100).fillna(0)
    df['BIAS10'] = (((df['Close'] - df['MA10']) / df['MA10']) * 100).fillna(0)
    df['BIAS20'] = (((df['Close'] - df['MA20']) / df['MA20']) * 100).fillna(0)
    
    return df

def calculate_interval_statistics(df):
    """
    計算個股最近一週、一月、一季、半年、一年的價格和成交量的平均、中位、最高及最低等數值
    """
    stats = {}
    today = df.index[-1]
    
    intervals = {
        '1週': datetime.timedelta(days=7),
        '1月': datetime.timedelta(days=30),
        '1季': datetime.timedelta(days=90),
        '半年': datetime.timedelta(days=180),
        '1年': datetime.timedelta(days=365)
    }
    
    for name, delta in intervals.items():
        start_date = today - delta
        sub_df = df[df.index >= start_date]
        
        if len(sub_df) == 0:
            continue
            
        stats[name] = {
            'price_mean': float(sub_df['Close'].mean()),
            'price_median': float(sub_df['Close'].median()),
            'price_max': float(sub_df['Close'].max()),
            'price_min': float(sub_df['Close'].min()),
            'volume_mean': float(sub_df['Volume'].mean()),
            'volume_median': float(sub_df['Volume'].median()),
            'volume_max': float(sub_df['Volume'].max()),
            'volume_min': float(sub_df['Volume'].min())
        }
    return stats

# ==========================================
# 2. K線圖繪製模組
# ==========================================

def generate_kline_chart_base64(df, name):
    """
    使用 mplfinance 繪製 K 線圖，並將其轉換成 base64 字串以嵌入 HTML
    """
    plot_df = df.tail(60).copy()
    
    mc = mpf.make_marketcolors(
        up='red',
        down='green',
        edge='inherit',
        wick='inherit',
        volume='inherit'
    )
    s = mpf.make_mpf_style(
        base_mpf_style='charles',
        marketcolors=mc,
        gridstyle=':',
        y_on_right=False
    )
    
    buf = io.BytesIO()
    
    mpf.plot(
        plot_df,
        type='candle',
        style=s,
        volume=True,
        mav=(5, 10, 20),
        title=f"\n{name} (Last 60 Days) K-Line",
        ylabel='Price (TWD)',
        ylabel_lower='Volume',
        savefig=dict(fname=buf, dpi=120, bbox_inches='tight'),
        figscale=1.1
    )
    
    buf.seek(0)
    img_bytes = buf.read()
    buf.close()
    plt.close('all')
    
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    return base64_str

# ==========================================
# 3. 規則引導與 AI 分析模組 (含 API 限流退避與指標規則備用機制)
# ==========================================

def generate_rule_based_analysis(name, ticker_symbol, latest_data):
    """
    當 Gemini API 每日配額用盡或連線失敗時，自動啟動技術指標邏輯分析作為備用方案，
    確保報表中的每一檔股票都一定有分析結果，不會出現 API 錯誤訊息。
    """
    latest_close = latest_data['Close']
    k_val = latest_data['K']
    d_val = latest_data['D']
    rsi6 = latest_data['RSI6']
    rsi14 = latest_data['RSI14']
    osc = latest_data['OSC']
    
    ma5 = latest_data['MA5']
    ma10 = latest_data['MA10']
    ma20 = latest_data['MA20']
    
    # 均線多空排列
    if ma5 > ma10 > ma20:
        ma_status = "均線系統呈多頭排列，股價短期偏強"
        ma_signal = "bullish"
    elif ma5 < ma10 < ma20:
        ma_status = "均線系統呈空頭排列，股價短期偏弱"
        ma_signal = "bearish"
    else:
        ma_status = "均線呈糾結狀態，處於區間整理階段"
        ma_signal = "neutral"
        
    # KD 交叉與區域
    kd_cross = "黃金交叉（K值大於D值）" if k_val > d_val else "死亡交叉（K值小於D值）"
    if k_val > 80:
        kd_zone = "進入超買區（K > 80），應留意高檔獲利回吐壓力"
    elif k_val < 20:
        kd_zone = "進入超賣區（K < 20），短期下檔有支撐，不宜過度殺低"
    else:
        kd_zone = "處於中性整理區間"
        
    # MACD 動能
    macd_status = "MACD 柱狀圖翻紅（OSC 大於 0），多頭動能擴張" if osc > 0 else "MACD 柱狀圖翻綠（OSC 小於 0），空頭動能主導"
    
    # 技術面綜合研判與建議
    advice = "觀望"
    reason = "目前技術指標多空互見。股價處於區間震盪狀態，建議維持觀望，等待突破關鍵均線後再行布局。"
    entry = f"{latest_close * 0.98:.2f}"
    profit = f"{latest_close * 1.06:.2f}"
    stop = f"{latest_close * 0.95:.2f}"
    
    # 買入訊號判定 (KD黃金交叉 + MACD翻紅 + 均線偏多)
    if k_val > d_val and osc > 0 and latest_close > ma20:
        advice = "買入"
        reason = "技術面轉強：KD呈現黃金交叉，且MACD多頭動能轉強，股價站穩月線（MA20）之上，短期有機會發動攻勢。"
        entry = f"{latest_close:.2f}"
        profit = f"{latest_close * 1.08:.2f}"
        stop = f"{latest_close * 0.95:.2f}"
    # 賣出訊號判定 (KD死亡交叉 + MACD翻綠 + 均線偏空)
    elif k_val < d_val and osc < 0 and latest_close < ma20:
        advice = "賣出"
        reason = "技術面轉弱：KD交叉向下且指標偏弱，MACD動能柱翻綠擴大，股價跌破月線支撐，技術面呈現修正趨勢。"
        entry = f"{latest_close * 0.90:.2f}"
        profit = f"{latest_close * 0.99:.2f}"
        stop = f"{latest_close * 1.02:.2f}"
        
    fallback_report = f"""### 一、當日表現分析 (技術指標與布林乖離規則生成)
今日該股收盤價為 **{latest_close:.2f}** 元。
目前 {ma_status}。在技術指標方面，KD 指標目前呈現 **{kd_cross}**，且 {kd_zone}。
RSI-6 目前數值為 **{rsi6:.1f}**。同時，{macd_status}。
此外，布林通道上限為 **{latest_data['BB_Upper']:.2f}** 元，下限為 **{latest_data['BB_Lower']:.2f}** 元，股價目前處於【{"布林超買區" if latest_close > latest_data['BB_Upper'] else ("布林超賣區" if latest_close < latest_data['BB_Lower'] else "布林通道內部整合")}】。
短期 5 日乖離率為 **{latest_data['BIAS5']:.2f}%**，20 日乖離率為 **{latest_data['BIAS20']:.2f}%**。
綜合技術指標狀態，目前多空力量呈現【{"多方佔優" if advice == "買入" else ("空方佔優" if advice == "賣出" else "多空拉鋸整理")}】。

### 二、明日開盤預測與動能評估
基於短線指標與月線多空位置判定，預測明日走勢以【{"偏多突破" if advice == "買入" else ("震盪偏弱" if advice == "賣出" else "區間平盤整理")}】機率較高。
上檔壓力參考 {latest_close * 1.03:.2f} 元，下檔短期防守支撐參考 {latest_close * 0.97:.2f} 元。

### 三、籌碼面與資券分析 (技術規則推估)
今日成交量為 {latest_data['Volume']:,.0f} 股。從量能變動來看，目前量能較 5 日均量【{"放大" if latest_data['Volume'] > latest_data['Volume_5'] else "萎縮"}】。
由於 Yahoo Finance 未提供即時融資券及三大法人每日買賣超，在此根據量價特徵與月線位置，推估目前主力籌碼呈現【{"偏向逢低進駐" if advice == "買入" else ("偏向逢高調節" if advice == "賣出" else "觀望縮量調節")}】趨勢。

### 四、買賣與目標價建議
- 交易建議：{advice}
- 建議理由：{reason}
- 進場參考價：{entry} 元
- 停利目標價：{profit} 元
- 停損參考價：{stop} 元

### 五、短期風險提示
防範技術指標短期與布林軌道邊界區隔鈍化，以及成交量不足之誘多陷阱。後續若量能無法溫和放大，仍須防範價格跌破均線支撐。
*(本欄位為技術指標規則自動判定生成)*
"""
    return fallback_report

def chunk_list(lst, n):
    """將 list 切分成大小為 n 的 chunk"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class SingleStockAnalysis(BaseModel):
    symbol: str = Field(description="股票代號，例如 '2330'")
    analysis_today: str = Field(description="當日表現與技術面分析。請在此段落詳細分析當日股價表現、均線多空排列、KD/RSI/MACD指標強弱，並融合【布林通道】（是否壓縮、觸及上下軌）與【乖離率】（是否過大或安全），以及【形態學】分析特徵（如突破、頭部、底部、整理等）。")
    prediction_tomorrow: str = Field(description="明日開盤預測與動能評估。結合今日收盤表現、市場消息與指標動能，評估明日開盤的走勢預測（開高/平盤/開低）以及背後的邏輯與市場心理。")
    chips_and_margin: str = Field(description="籌碼面與資券動態分析。請結合成交量變動、相關新聞中提及的三大法人買賣超動向，以及融資融券變動趨勢進行多空力道解讀。")
    advice: str = Field(description="交易建議。必須是 '買入'、'賣出' 或 '觀望' 中的一個，嚴禁填寫其他詞彙")
    reason: str = Field(description="建議理由：簡述交易策略與上述技術面、形態面、籌碼面結合的原因。")
    entry_price: str = Field(description="進場參考價。必須是具體價格數字（例如 '800' 或 '800.50'），絕對不可填寫 'N/A'、'無'、'--'、'未提供' 或空白。如果是觀望，請給出拉回合理支撐點的具體價格數字。如果是賣出，請給出未來回補或重新買進的具體價格數字。")
    target_price: str = Field(description="停利目標價。必須是具體價格數字（例如 '900'），絕對不可填寫 'N/A'、'無'、'--'、'未提供' 或空白。如果是觀望或賣出，請給出反彈壓力位或前高的具體價格數字。")
    stop_loss_price: str = Field(description="停損參考價。必須是具體價格數字（例如 '750'），絕對不可填寫 'N/A'、'無'、'--' 或空白。如果是觀望，請給出下檔關鍵支撐防守價。如果是賣出，請給出回補或反彈壓力突破強行停損的具體價格數字。")
    risk_warning: str = Field(description="短期風險提示。說明當前可能面臨的技術面修正、基本面或市場消息面風險。")

class BatchStockAnalysisResponse(BaseModel):
    analyses: List[SingleStockAnalysis]

def analyze_batch_with_gemini(api_key, batch_stocks):
    """
    使用單一 API 呼叫，批次分析多檔股票，並以結構化 JSON 格式回傳
    """
    if not api_key or api_key.startswith("YOUR_"):
        print("Gemini API Key 未設定，將對此批次個股採用規則技術分析...")
        return None
        
    client = genai.Client(api_key=api_key)
    
    prompt = """你是一位專業的台股投資分析師與量化交易專家。請針對以下多檔個股的盤後數據、歷史統計、技術指標及最新市場消息進行深入分析。

請為以下每檔股票撰寫一份詳細的專業報告。
請使用繁體中文（台灣習慣用語），並回傳符合 schema 定義的結構化 JSON 資料。

【特別分析要求】
1. 必須加入「形態學分析」：識別如雙底 (W底)、雙頂 (M頭)、頭肩、整理平台突破、K線組合等經典形態特徵。
2. 必須加入「籌碼面與融資融券分析」：根據市場新聞中提及的三大法人買賣超動態、成交量異常放大/縮小，以及融資融券變動趨勢進行推估分析。
3. ⚠️ 重要：【交易建議價格規範】
   不論交易建議是「買入」、「賣出」還是「觀望」，都【嚴禁】在 entry_price, target_price, stop_loss_price 填寫 "N/A"、"無"、"--"、"未提供" 或空白。
   - 若建議為「觀望」：必須提供拉回合理支撐位時的「進場參考價」、突破前高的「停利目標價」、跌破關鍵支撐的「停損參考價」。
   - 若建議為「賣出」：必須提供未來跌深欲重新佈局或反彈回補時的「進場參考價」、反彈壓力位的「停利目標價」、關鍵壓力突破需強行停損的「停損參考價」。
   - 請基於今日收盤價與支撐壓力位，給出具體的「數值價格」（例如 "800" 或 "800.5"），不可使用文字模糊帶過，必須是純數字。

"""
    for item in batch_stocks:
        prompt += f"--- 股票：{item['name']} ({item['symbol']}) ---\n"
        prompt += f"- 分析日期：{item['date']}\n"
        prompt += f"- 今日價格與量：開盤 {item['latest_open']:.2f} 元, 收盤 {item['latest_close']:.2f} 元, 最高 {item['latest_high']:.2f} 元, 最低 {item['latest_low']:.2f} 元, 成交量 {item['latest_volume']:,.0f} 股\n"
        prompt += f"- 技術指標：MA5={item['ma5']:.2f}, MA10={item['ma10']:.2f}, MA20={item['ma20']:.2f}, MA60={item['ma60']:.2f}\n"
        prompt += f"  布林通道：上限 {item['bb_upper']:.2f}, 中軌 {item['bb_middle']:.2f}, 下限 {item['bb_lower']:.2f}\n"
        prompt += f"  乖離率 (BIAS)：5日 {item['bias5']:.2f}%, 10日 {item['bias10']:.2f}%, 20日 {item['bias20']:.2f}%\n"
        prompt += f"  KD: K={item['k_val']:.2f}, D={item['d_val']:.2f} | RSI: RSI-6={item['rsi6']:.2f}, RSI-14={item['rsi14']:.2f}\n"
        prompt += f"  MACD: DIF={item['dif']:.2f}, DEM={item['dem']:.2f}, OSC={item['osc']:.2f}\n"
        prompt += f"- 歷史區間統計對比：\n{item['stats_text']}\n"
        prompt += f"- 相關市場新聞：\n{item['news_text']}\n\n"
        
    for attempt in range(5):
        try:
            print(f"正在對此批次個股進行 AI 智慧分析 (個股數: {len(batch_stocks)})...")
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=BatchStockAnalysisResponse,
                )
            )
            try:
                res_data = json.loads(response.text)
                return res_data.get('analyses', [])
            except Exception as je:
                print(f"解析 Gemini 回傳的 JSON 失敗: {je}")
                print(f"回傳原始內容: {response.text}")
                
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "503" in err_str or "quota" in err_str.lower() or "limit" in err_str.lower():
                sleep_time = (attempt + 1) * 20
                print(f"Gemini API 限流或忙碌，將於 {sleep_time} 秒後重試批次分析...")
                time.sleep(sleep_time)
            else:
                print(f"Gemini API 批次呼叫發生嚴重錯誤: {e}")
                break
                
    return None

def analyze_with_gemini(api_key, name, ticker_symbol, latest_data, stats_data, news_list):
    """
    呼叫 Gemini API 進行專業的盤後分析與預測（限流重試，失敗則自動退避至規則分析）
    """
    if not api_key or api_key.startswith("YOUR_"):
        print(f"[{name}] Gemini API Key 未設定，啟動規則技術分析...")
        return generate_rule_based_analysis(name, ticker_symbol, latest_data)
        
    client = genai.Client(api_key=api_key)
    
    latest_date = latest_data.name.strftime('%Y-%m-%d')
    latest_close = latest_data['Close']
    latest_open = latest_data['Open']
    latest_high = latest_data['High']
    latest_low = latest_data['Low']
    latest_volume = latest_data['Volume']
    
    k_val = latest_data['K']
    d_val = latest_data['D']
    rsi6 = latest_data['RSI6']
    rsi14 = latest_data['RSI14']
    dif = latest_data['DIF']
    dem = latest_data['DEM']
    osc = latest_data['OSC']
    
    ma5 = latest_data['MA5']
    ma10 = latest_data['MA10']
    ma20 = latest_data['MA20']
    ma60 = latest_data['MA60']
    
    news_text = ""
    if news_list:
        for idx, item in enumerate(news_list):
            news_text += f"{idx+1}. {item.get('title')} (發布: {item.get('publisher')})\n   連結: {item.get('link')}\n"
    else:
        news_text = "目前無相關即時新聞。\n"
        
    stats_text = ""
    for name_key, s in stats_data.items():
        stats_text += f"- {name_key}:\n"
        stats_text += f"  收盤價: 平均 {s['price_mean']:.2f}, 中位 {s['price_median']:.2f}, 最高 {s['price_max']:.2f}, 最低 {s['price_min']:.2f}\n"
        stats_text += f"  成交量: 平均 {s['volume_mean']:,.0f}, 中位 {s['volume_median']:,.0f}, 最高 {s['volume_max']:,.0f}, 最低 {s['volume_min']:,.0f}\n"

    prompt = f"""
你是一位專業的台股投資分析師與量化交易專家。請針對以下個股的盤後數據、歷史統計、技術指標及最新市場消息進行深入分析。

【個股基本數據】
- 個股名稱：{name} ({ticker_symbol})
- 分析日期：{latest_date}
- 今日開盤價：{latest_open:.2f} 元
- 今日最高價：{latest_high:.2f} 元
- 今日最低價：{latest_low:.2f} 元
- 今日收盤價：{latest_close:.2f} 元
- 今日成交量：{latest_volume:,.0f} 股

【均線與技術指標】
- 均線系統：MA5={ma5:.2f}, MA10={ma10:.2f}, MA20={ma20:.2f}, MA60={ma60:.2f}
- KD 指標：K={k_val:.2f}, D={d_val:.2f}
- RSI 指標：RSI-6={rsi6:.2f}, RSI-14={rsi14:.2f}
- MACD 指標：DIF={dif:.2f}, DEM={dem:.2f}, OSC={osc:.2f}

【歷史區間數據對比】
{stats_text}

【相關市場新聞】
{news_text}

請基於上述資料進行盤後綜合分析，並撰寫一份詳細的專業報告。
請嚴格按照以下四個部分進行輸出，並使用繁體中文（台灣習慣用語）：

### 一、當日表現分析
[請從基本市況、均線排列、技術指標多空訊號以及市場新聞這幾個維度，詳細解析該股今日的表現與多空力量對比]

### 二、明日開盤預測與動能評估
[請結合今日收盤表現、市場消息與指標動能，評估明日開盤的走勢預測（開高/平盤/開低）以及背後的邏輯與市場心理]

### 三、買賣與目標價建議
- 交易建議：[買入 / 賣出 / 觀望]
- 建議理由：[請簡述原因]
- 進場參考價：[具體數值或合理區間] 元
- 停利目標價：[具體數值] 元
- 停損參考價：[具體數值] 元

### 四、短期風險提示
[說明當前可能面臨的技術面修正、基本面或市場消息面風險]
"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt
            )
            return response.text
        except Exception as e:
            err_str = str(e)
            # 判斷是否為頻率超限 (429) 或伺服器忙碌 (503) 等配額問題
            if "429" in err_str or "503" in err_str or "quota" in err_str.lower() or "limit" in err_str.lower():
                sleep_time = (attempt + 1) * 15
                print(f"[{name}] Gemini API 限流或忙碌，將於 {sleep_time} 秒後重試...")
                time.sleep(sleep_time)
            else:
                # 其它致命錯誤直接跳出
                print(f"[{name}] Gemini API 發生錯誤 ({e})，啟動規則技術分析...")
                break
                
    # 重試皆失敗或配額已用盡時，自動降級為規則技術分析
    print(f"[{name}] Gemini API 配額用盡或呼叫失敗，已自動生成規則技術指標報告替代...")
    return generate_rule_based_analysis(name, ticker_symbol, latest_data)

def extract_ai_recommendation(ai_text):
    """
    從分析回應中提取交易建議 (買入/賣出/觀望) 以及目標區間描述
    """
    advice = "觀望"
    targets = "未設定"
    
    lines = ai_text.split('\n')
    for line in lines:
        if '交易建議' in line:
            for m in ['買入', '買進', '賣出', '觀望']:
                if m in line:
                    advice = m
                    break
                    
    entry = ""
    profit = ""
    stop = ""
    for line in lines:
        if '進場參考價' in line:
            entry = line.split('：')[-1].split(':')[-1].strip().replace('元', '').strip()
        elif '停利目標價' in line:
            profit = line.split('：')[-1].split(':')[-1].strip().replace('元', '').strip()
        elif '停損參考價' in line:
            stop = line.split('：')[-1].split(':')[-1].strip().replace('元', '').strip()
            
    if entry or profit:
        targets = f"{entry} ➔ {profit} (損:{stop})"
    else:
        for line in lines:
            if ('目標' in line or '參考' in line) and ('元' in line):
                targets = line.replace('- ', '').strip()
                if len(targets) > 25:
                    targets = targets[:25] + "..."
                break
                
    return advice, targets

# ==========================================
# 4. Google Drive 儲存模組 (服務帳戶)
# ==========================================

def get_gdrive_service():
    """
    建立 Google Drive API 服務物件。
    支援兩種驗證方式（優先採用個人 OAuth2 授權以解決個人雲端硬碟容量限制問題，若無則採用服務帳戶）：
    """
    scopes = ['https://www.googleapis.com/auth/drive']
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    
    creds = None
    
    # ----------------------------------------
    # 方式一：個人 OAuth2 授權 (解決個人帳戶 0 空間問題)
    # ----------------------------------------
    # 1. 檢查 GitHub Actions 環境變數
    token_env = os.environ.get('GDRIVE_OAUTH_TOKEN')
    if token_env:
        try:
            creds_info = json.loads(token_env)
            creds = Credentials.from_authorized_user_info(creds_info, scopes)
            print("成功從環境變數載入個人 OAuth2 憑證。")
        except Exception as e:
            print(f"解析環境變數 GDRIVE_OAUTH_TOKEN 錯誤: {e}")
            
    # 2. 檢查本機 token.json
    if not creds and os.path.exists('token.json'):
        try:
            creds = Credentials.from_authorized_user_file('token.json', scopes)
            print("成功從本機 token.json 載入個人 OAuth2 憑證。")
        except Exception as e:
            print(f"載入本機 token.json 錯誤: {e}")
            
    # 3. 如果憑證過期但有 refresh_token，則自動刷新
    if creds and creds.expired and creds.refresh_token:
        try:
            print("個人 OAuth2 憑證已過期，正在自動刷新...")
            creds.refresh(Request())
            # 保存刷新後的憑證
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
            print("憑證刷新成功，已更新 token.json。")
        except Exception as e:
            print(f"刷新 OAuth2 憑證失敗: {e}")
            creds = None
            
    # 4. 如果本機無憑證，但有 credentials.json，啟動瀏覽器互動登入 (僅限本機)
    if not creds:
        if os.path.exists('credentials.json'):
            try:
                from google_auth_oauthlib.flow import InstalledAppFlow
                print("未偵測到 token.json，正在啟動瀏覽器進行個人 OAuth2 授權登入...")
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', scopes)
                creds = flow.run_local_server(port=0)
                with open('token.json', 'w') as token:
                    token.write(creds.to_json())
                print("授權成功！已為您儲存 token.json，下次執行無須再次登入。")
            except Exception as e:
                print(f"執行 OAuth2 瀏覽器登入流程失敗: {e}")
                
    if creds:
        try:
            service = build('drive', 'v3', credentials=creds)
            return service
        except Exception as e:
            print(f"建立 OAuth2 服務物件失敗: {e}")

    # ----------------------------------------
    # 方式二：服務帳戶 (Service Account)
    # ----------------------------------------
    print("嘗試使用 Google 服務帳戶憑證驗證...")
    creds_dict = None
    sa_env = os.environ.get('GDRIVE_SERVICE_ACCOUNT')
    if sa_env:
        try:
            creds_dict = json.loads(sa_env)
        except Exception as e:
            print(f"解析環境變數 GDRIVE_SERVICE_ACCOUNT 錯誤: {e}")
            
    if not creds_dict and os.path.exists('service_account.json'):
        try:
            with open('service_account.json', 'r', encoding='utf-8') as f:
                creds_dict = json.load(f)
        except Exception as e:
            print(f"讀取本機 service_account.json 錯誤: {e}")
            
    if creds_dict:
        try:
            creds_sa = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
            service = build('drive', 'v3', credentials=creds_sa)
            return service
        except Exception as e:
            print(f"建立服務帳戶驗證失敗: {e}")
            
    print("❌ 錯誤：所有 Google Drive 驗證方式皆失敗，將略過雲端硬碟上傳。")
    return None

def upload_html_to_gdrive(service, file_path, folder_id, file_name):
    """
    將生成的 HTML 報告上傳到雲端硬碟特定資料夾，並設定公開檢視與回傳連結
    """
    if not service:
        return None
        
    file_metadata = {
        'name': file_name,
        'mimeType': 'text/html'
    }
    if folder_id:
        file_metadata['parents'] = [folder_id]
        
    media = MediaFileUpload(file_path, mimetype='text/html', resumable=True)
    
    try:
        query = f"name = '{file_name}' and trashed = false"
        if folder_id:
            query += f" and '{folder_id}' in parents"
            
        results = service.files().list(q=query, fields="files(id)").execute()
        files = results.get('files', [])
        
        if files:
            file_id = files[0]['id']
            print(f"找到同名檔案 (ID: {file_id})，正在覆蓋更新...")
            file = service.files().update(
                fileId=file_id,
                media_body=media,
                fields='id, webViewLink'
            ).execute()
        else:
            print("正在建立新檔案...")
            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, webViewLink'
            ).execute()
            file_id = file.get('id')
            
        web_view_link = file.get('webViewLink')
        
        try:
            permission = {
                'type': 'anyone',
                'role': 'reader'
            }
            service.permissions().create(
                fileId=file_id,
                body=permission
            ).execute()
        except Exception as pe:
            print(f"設定雲端硬碟檔案公開權限失敗: {pe}")
            
        return web_view_link
    except Exception as e:
        print(f"Google 雲端硬碟上傳失敗: {e}")
        return None

# ==========================================
# 5. 通知與發送模組 (Gmail, LINE)
# ==========================================

def send_gmail_summary(gmail_user, gmail_password, recipient, subject, html_content):
    """
    發送單一彙總的電子報郵件
    """
    if not gmail_user or not gmail_password or gmail_user.startswith("YOUR_") or gmail_password.startswith("YOUR_"):
        print("Gmail 使用者或應用程式密碼未正確設定。跳過電子郵件發送。")
        return False
        
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = gmail_user
    msg['To'] = recipient
    
    msg_html = MIMEText(html_content, 'html', 'utf-8')
    msg.attach(msg_html)
        
    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(gmail_user, gmail_password)
        server.sendmail(gmail_user, [recipient], msg.as_string())
        server.close()
        print(f"彙總電子郵件成功發送至 {recipient}")
        return True
    except Exception as e:
        print(f"電子郵件發送失敗: {e}")
        return False

def send_line_summary_notification(token, user_id, date_str, stats_summary, web_view_link):
    """
    使用 LINE Messaging API 推送單一彙總訊息 (支援以逗號分隔發送給多個使用者)
    """
    if not token or not user_id or token.startswith("YOUR_") or user_id.startswith("YOUR_"):
        print("LINE Channel Token 或 User ID 未設定。跳過 LINE 發送。")
        return False
        
    user_ids = [uid.strip() for uid in user_id.split(',') if uid.strip()]
    if not user_ids:
        print("沒有有效的 LINE User ID，跳過發送。")
        return False
        
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    buy_list = ", ".join(stats_summary['buys']) if stats_summary['buys'] else "無"
    sell_list = ", ".join(stats_summary['sells']) if stats_summary['sells'] else "無"
    
    line_text = f"📊 台股盤後智慧分析總報告 ({date_str})\n" \
                f"今日已為您完成 {stats_summary['total']} 檔個股分析！\n\n" \
                f"📈 AI 交易策略摘要：\n" \
                f"• 建議買入 📈：{buy_list}\n" \
                f"• 建議賣出 📉：{sell_list}\n" \
                f"• 建議觀望 👁️：共 {len(stats_summary['holds'])} 檔\n\n"
                
    if web_view_link:
        line_text += f"🔗 雲端硬碟分頁總報告連結：\n{web_view_link}"
    else:
        line_text += "（本次未成功上傳雲端硬碟報告，請確認 service_account.json 設定）"
        
    success = True
    for uid in user_ids:
        payload = {
            "to": uid,
            "messages": [
                {
                    "type": "text",
                    "text": line_text
                }
            ]
        }
        try:
            res = requests.post(url, headers=headers, json=payload)
            if res.status_code == 200:
                print(f"LINE 彙總訊息推送成功！(使用者: {uid})")
            else:
                print(f"LINE 推送失敗 ({res.status_code}) to {uid}: {res.text}")
                success = False
        except Exception as e:
            print(f"LINE 發送異常 to {uid}: {e}")
            success = False
            
    return success

# ==========================================
# 6. HTML 模板定義
# ==========================================

# 整合型分頁 HTML 模板 (包含 Sidebar, Dashboard 首頁與各股詳情分頁)
CONSOLIDATED_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>台股盤後智慧分析總報告 - {{ date }}</title>
    <style>
        :root {
            --bg: #F0F2F5;
            --sidebar-bg: #1E293B;
            --sidebar-text: #94A3B8;
            --sidebar-hover-bg: #334155;
            --sidebar-active-bg: #2563EB;
            --sidebar-active-text: #FFFFFF;
            --card-bg: #FFFFFF;
            --text-primary: #1E293B;
            --text-secondary: #64748B;
            --border-color: #E2E8F0;
            --up-color: #EF4444; /* 紅漲 */
            --down-color: #10B981; /* 綠跌 */
            --buy-bg: #FEE2E2;
            --buy-text: #EF4444;
            --sell-bg: #D1FAE5;
            --sell-text: #10B981;
            --hold-bg: #F1F5F9;
            --hold-text: #64748B;
        }
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Microsoft JhengHei", sans-serif;
            background-color: var(--bg);
            color: var(--text-primary);
            margin: 0;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }
        
        /* 側邊導覽欄 */
        .sidebar {
            width: 260px;
            background-color: var(--sidebar-bg);
            color: var(--sidebar-text);
            display: flex;
            flex-direction: column;
            height: 100%;
            flex-shrink: 0;
        }
        .sidebar-header {
            padding: 24px;
            font-size: 18px;
            font-weight: 800;
            color: #F8FAFC;
            border-bottom: 1px solid #334155;
            letter-spacing: 0.5px;
        }
        .sidebar-menu {
            flex: 1;
            overflow-y: auto;
            padding: 15px 12px;
        }
        .menu-group-title {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #475569;
            padding: 10px 16px;
            font-weight: bold;
        }
        .menu-item {
            display: flex;
            align-items: center;
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 4px;
            cursor: pointer;
            color: var(--sidebar-text);
            font-size: 14px;
            transition: all 0.2s ease;
            text-decoration: none;
        }
        .menu-item:hover {
            background-color: var(--sidebar-hover-bg);
            color: #F8FAFC;
        }
        .menu-item.active {
            background-color: var(--sidebar-active-bg);
            color: var(--sidebar-active-text);
            font-weight: bold;
        }
        
        /* 主內容區域 */
        .main-content {
            flex: 1;
            overflow-y: auto;
            padding: 30px;
            height: 100%;
        }
        
        .tab-pane {
            display: none;
        }
        .tab-pane.active {
            display: block;
            animation: fadeIn 0.3s ease-in-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* 卡片與通用組件 */
        .header-section {
            margin-bottom: 25px;
        }
        .header-section h1 {
            margin: 0;
            font-size: 28px;
            font-weight: 800;
            color: #0F172A;
        }
        .header-section p {
            margin: 5px 0 0 0;
            color: var(--text-secondary);
            font-size: 14px;
        }
        
        .card {
            background: var(--card-bg);
            border-radius: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            border: 1px solid var(--border-color);
            padding: 24px;
            margin-bottom: 24px;
        }
        .card-title {
            font-size: 18px;
            font-weight: 700;
            color: #0F172A;
            margin-top: 0;
            margin-bottom: 20px;
            border-left: 4px solid var(--sidebar-active-bg);
            padding-left: 12px;
        }
        
        /* 儀表板首頁卡片 */
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 24px;
        }
        .stat-card {
            background: var(--card-bg);
            border-radius: 12px;
            border: 1px solid var(--border-color);
            padding: 20px;
            text-align: center;
        }
        .stat-label {
            font-size: 12px;
            color: var(--text-secondary);
            font-weight: bold;
            margin-bottom: 5px;
        }
        .stat-value {
            font-size: 28px;
            font-weight: 800;
            color: #0F172A;
        }
        
        /* 表格樣式 */
        .table-responsive {
            width: 100%;
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            text-align: right;
        }
        th, td {
            padding: 14px 16px;
            border-bottom: 1px solid var(--border-color);
        }
        th {
            background-color: #F8FAFC;
            color: #475569;
            font-weight: 700;
        }
        td.text-left, th.text-left {
            text-align: left;
        }
        .stock-row {
            cursor: pointer;
            transition: background 0.15s ease;
        }
        .stock-row:hover {
            background-color: #F8FAFC;
        }
        
        .font-bold { font-weight: bold; }
        .up-color { color: var(--up-color); }
        .down-color { color: var(--down-color); }
        
        /* 標籤 */
        .badge {
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: bold;
            display: inline-block;
        }
        .badge-buy { background-color: var(--buy-bg); color: var(--buy-text); }
        .badge-sell { background-color: var(--sell-bg); color: var(--sell-text); }
        .badge-hold { background-color: var(--hold-bg); color: var(--hold-text); }
        
        /* 個股詳情數據網格 */
        .grid-3 {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .data-item {
            padding: 12px 16px;
            background-color: #F8FAFC;
            border-radius: 8px;
            border: 1px solid #EDF2F7;
        }
        .data-label {
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }
        .data-value {
            font-size: 22px;
            font-weight: 700;
        }
        
        /* K線圖 */
        .chart-container {
            text-align: center;
            margin: 15px 0;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        /* AI 分析文章 */
        .analysis-content {
            font-size: 15px;
            line-height: 1.7;
        }
        .analysis-content h3 {
            color: var(--sidebar-active-bg);
            font-size: 16px;
            margin-top: 25px;
            margin-bottom: 10px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 6px;
        }
        .analysis-content ul {
            padding-left: 20px;
        }
        .analysis-content li {
            margin-bottom: 8px;
        }
        
        .btn-detail {
            background-color: #EFF6FF;
            color: var(--sidebar-active-bg);
            border: 1px solid #BFDBFE;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-detail:hover {
            background-color: var(--sidebar-active-bg);
            color: white;
            border-color: var(--sidebar-active-bg);
        }
        
        /* 響應式調整 */
        @media (max-width: 768px) {
            body { flex-direction: column; overflow: auto; }
            .sidebar { width: 100%; height: auto; }
            .sidebar-menu { display: flex; overflow-x: auto; padding: 10px; }
            .menu-group-title { display: none; }
            .menu-item { margin-bottom: 0; margin-right: 8px; flex-shrink: 0; }
            .main-content { padding: 15px; height: auto; overflow: visible; }
        }
    </style>
</head>
<body>
    <!-- 側邊欄 -->
    <div class="sidebar">
        <div class="sidebar-header">📈 台股智慧理財</div>
        <div class="sidebar-menu">
            <a class="menu-item active" id="menu-dashboard" onclick="switchTab('dashboard')">📊 總覽儀表板</a>
            <div class="menu-group-title">追蹤個股</div>
            {% for s in snapshots %}
            <a class="menu-item" id="menu-tab-{{ s.symbol }}" onclick="switchTab('tab-{{ s.symbol }}')">🔍 {{ s.symbol }} {{ s.name }}</a>
            {% endfor %}
        </div>
    </div>
    
    <!-- 主內容區 -->
    <div class="main-content">
        
        <!-- 儀表板首頁分頁 -->
        <div id="dashboard" class="tab-pane active">
            <div class="header-section">
                <h1>台股盤後分析儀表板</h1>
                <p>分析日期：{{ date }} | 已追蹤 {{ total_stocks }} 檔股票</p>
            </div>
            
            <div class="stat-grid">
                <div class="stat-card" style="border-top: 4px solid var(--up-color);">
                    <div class="stat-label">建議買入</div>
                    <div class="stat-value up-color">{{ stat_summary.buys|length }}</div>
                </div>
                <div class="stat-card" style="border-top: 4px solid var(--down-color);">
                    <div class="stat-label">建議賣出</div>
                    <div class="stat-value down-color">{{ stat_summary.sells|length }}</div>
                </div>
                <div class="stat-card" style="border-top: 4px solid var(--text-secondary);">
                    <div class="stat-label">建議觀望</div>
                    <div class="stat-value" style="color: var(--text-secondary);">{{ stat_summary.holds|length }}</div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">今日個股表現與交易建議總覽</div>
                <div class="table-responsive">
                    <table>
                        <thead>
                            <tr>
                                <th class="text-left">股票名稱</th>
                                <th>最新收盤價</th>
                                <th>本日漲跌</th>
                                <th>本日漲跌幅</th>
                                <th>KD 指標</th>
                                <th>RSI6 / RSI14</th>
                                <th>AI 交易建議</th>
                                <th>交易目標區間</th>
                                <th>操作</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for s in snapshots %}
                            <tr class="stock-row" onclick="switchTab('tab-{{ s.symbol }}')">
                                <td class="text-left font-bold" style="color: var(--sidebar-active-bg);">
                                    {{ s.symbol }} {{ s.name }}
                                </td>
                                <td class="font-bold">{{ "%.2f"|format(s.latest_close) }} 元</td>
                                <td class="{% if s.change >= 0 %}up-color{% else %}down-color{% endif %}">
                                    {{ "%+.2f"|format(s.change) }} 元
                                </td>
                                <td class="{% if s.change >= 0 %}up-color{% else %}down-color{% endif %} font-bold">
                                    {{ "%+.2f"|format(s.change_percent) }}%
                                </td>
                                <td>K:{{ "%.1f"|format(s.k_val) }} | D:{{ "%.1f"|format(s.d_val) }}</td>
                                <td>{{ "%.1f"|format(s.rsi6) }} / {{ "%.1f"|format(s.rsi14) }}</td>
                                <td>
                                    {% if '買入' in s.ai_advice or '買進' in s.ai_advice %}
                                    <span class="badge badge-buy">買入</span>
                                    {% elif '賣出' in s.ai_advice %}
                                    <span class="badge badge-sell">賣出</span>
                                    {% else %}
                                    <span class="badge badge-hold">觀望</span>
                                    {% endif %}
                                </td>
                                <td class="font-bold" style="font-size: 13px;">{{ s.ai_targets }}</td>
                                <td><button class="btn-detail">詳細報告</button></td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <!-- 各股分頁 -->
        {% for s in snapshots %}
        <div id="tab-{{ s.symbol }}" class="tab-pane">
            <div class="header-section">
                <h1>{{ s.name }} ({{ s.symbol }}) 盤後分析</h1>
                <p>分析日期：{{ s.date }}</p>
            </div>
            
            <!-- 數據小卡 -->
            <div class="grid-3">
                <div class="data-item">
                    <div class="data-label">最新收盤價</div>
                    <div class="data-value {% if s.change >= 0 %}up-color{% else %}down-color{% endif %}">
                        {{ "%.2f"|format(s.latest_close) }} 元 ({{ "%+.2f"|format(s.change_percent) }}%)
                    </div>
                </div>
                <div class="data-item">
                    <div class="data-label">開盤 / 最高 / 最低</div>
                    <div class="data-value" style="font-size: 16px;">
                        {{ "%.2f"|format(s.latest_open) }} / {{ "%.2f"|format(s.latest_high) }} / {{ "%.2f"|format(s.latest_low) }} 元
                    </div>
                </div>
                <div class="data-item">
                    <div class="data-label">當日成交量</div>
                    <div class="data-value" style="font-size: 16px; color: #1E293B;">
                        {{ "{:,.0f}".format(s.latest_volume) }} 股
                    </div>
                </div>
                <div class="data-item">
                    <div class="data-label">KD 指標狀態</div>
                    <div class="data-value" style="font-size: 16px; color: var(--sidebar-active-bg);">
                        K: {{ "%.2f"|format(s.k_val) }} | D: {{ "%.2f"|format(s.d_val) }}
                    </div>
                </div>
            </div>
            
            <!-- K 線圖 -->
            <div class="card">
                <div class="card-title">K 線走勢圖 (近 60 個交易日)</div>
                <div class="chart-container">
                    <img src="data:image/png;base64,{{ s.kline_base64 }}" alt="{{ s.name }} K-Line">
                </div>
            </div>
            
            <!-- 歷史區間表格 -->
            <div class="card">
                <div class="card-title">歷史統計對比</div>
                <div class="table-responsive">
                    <table>
                        <thead>
                            <tr>
                                <th class="text-left">區間</th>
                                <th>價格平均</th>
                                <th>價格中位</th>
                                <th>價格最高</th>
                                <th>價格最低</th>
                                <th>成交量平均</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for name, item in s.stats.items() %}
                            <tr>
                                <td class="text-left" style="font-weight: 700;">{{ name }}</td>
                                <td>{{ "%.2f"|format(item.price_mean) }}</td>
                                <td>{{ "%.2f"|format(item.price_median) }}</td>
                                <td class="up-color">{{ "%.2f"|format(item.price_max) }}</td>
                                <td class="down-color">{{ "%.2f"|format(item.price_min) }}</td>
                                <td>{{ "{:,.0f}".format(item.volume_mean) }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- AI 報告 -->
            <div class="card">
                <div class="card-title">AI 專業盤後分析報告</div>
                <div class="analysis-content">
                    {{ s.gemini_analysis_html }}
                </div>
            </div>
        </div>
        {% endfor %}
        
    </div>
    
    <script>
        function switchTab(tabId) {
            const panes = document.querySelectorAll('.tab-pane');
            panes.forEach(pane => pane.classList.remove('active'));
            
            const target = document.getElementById(tabId);
            if (target) {
                target.classList.add('active');
            }
            
            const menuItems = document.querySelectorAll('.menu-item');
            menuItems.forEach(item => item.classList.remove('active'));
            
            if (tabId === 'dashboard') {
                document.getElementById('menu-dashboard').classList.add('active');
            } else {
                const activeMenu = document.getElementById('menu-' + tabId);
                if (activeMenu) {
                    activeMenu.classList.add('active');
                }
            }
            
            document.querySelector('.main-content').scrollTop = 0;
        }

        // 當從信件直接開啟個股連結時，讀取網址的 Hash (#tab-XXXX) 自動切換分頁
        function checkHash() {
            const hash = window.location.hash;
            if (hash && hash.startsWith('#tab-')) {
                const tabId = hash.substring(1);
                switchTab(tabId);
            } else if (hash === '#dashboard' || hash === '') {
                switchTab('dashboard');
            }
        }

        window.addEventListener('DOMContentLoaded', checkHash);
        window.addEventListener('hashchange', checkHash);
    </script>
</body>
</html>
"""

# Gmail 電子報 HTML 模板 (包含 Summary 彙總表，超連結至雲端硬碟各股分頁)
GMAIL_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>台股盤後智慧分析彙總報告</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #F3F4F6; margin: 0; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
        .header { background: linear-gradient(135deg, #1A365D, #2B6CB0); color: white; padding: 25px; border-radius: 8px; text-align: center; margin-bottom: 25px; }
        .header h2 { margin: 0 0 10px 0; font-size: 24px; }
        .summary-table { width: 100%; border-collapse: collapse; margin-bottom: 25px; font-size: 14px; }
        .summary-table th, .summary-table td { padding: 12px; border-bottom: 1px solid #e5e7eb; text-align: right; }
        .summary-table th { background-color: #F8FAFC; color: #475569; text-align: right; font-weight: bold; }
        .summary-table td.text-left, .summary-table th.text-left { text-align: left; }
        .up-color { color: #EF4444; font-weight: bold; }
        .down-color { color: #10B981; font-weight: bold; }
        .badge { padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; display: inline-block; }
        .badge-buy { background-color: #FEE2E2; color: #EF4444; }
        .badge-sell { background-color: #D1FAE5; color: #10B981; }
        .badge-hold { background-color: #F1F5F9; color: #64748B; }
        .btn { display: block; width: 260px; margin: 30px auto 10px auto; padding: 14px 24px; background-color: #2563EB; color: white !important; text-align: center; text-decoration: none; font-weight: bold; border-radius: 6px; box-shadow: 0 4px 6px rgba(37,99,235,0.2); }
        .footer-note { font-size: 12px; color: #64748B; text-align: center; margin-top: 30px; border-top: 1px solid #E2E8F0; padding-top: 20px; }
        .link-text { text-decoration: none; color: #2563EB; font-weight: bold; }
        .link-text:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>台股盤後智慧分析總報告</h2>
            <p style="margin: 0; font-size: 14px; opacity: 0.9;">分析日期：{{ date }} | 已追蹤個股共 {{ total_stocks }} 檔</p>
        </div>
        
        <table class="summary-table">
            <thead>
                <tr>
                    <th class="text-left">股票</th>
                    <th>收盤價</th>
                    <th>漲跌幅</th>
                    <th>KD 指標</th>
                    <th>AI 建議</th>
                    <th>交易參考區間</th>
                    <th>個股詳情</th>
                </tr>
            </thead>
            <tbody>
                {% for s in snapshots %}
                <tr>
                    <td class="text-left" style="font-weight: bold; color: #1E293B;">
                        {% if web_view_link %}
                        <a href="{{ web_view_link }}#tab-{{ s.symbol }}" target="_blank" class="link-text">
                            {{ s.symbol }} {{ s.name }}
                        </a>
                        {% else %}
                        {{ s.symbol }} {{ s.name }}
                        {% endif %}
                    </td>
                    <td style="font-weight: bold;">{{ "%.2f"|format(s.latest_close) }} 元</td>
                    <td class="{% if s.change >= 0 %}up-color{% else %}down-color{% endif %}">
                        {{ "%+.2f"|format(s.change_percent) }}%
                    </td>
                    <td>K:{{ "%.1f"|format(s.k_val) }} | D:{{ "%.1f"|format(s.d_val) }}</td>
                    <td>
                        {% if '買入' in s.ai_advice or '買進' in s.ai_advice %}
                            <span class="badge badge-buy">買入</span>
                        {% elif '賣出' in s.ai_advice %}
                            <span class="badge badge-sell">賣出</span>
                        {% else %}
                            <span class="badge badge-hold">觀望</span>
                        {% endif %}
                    </td>
                    <td style="font-weight: bold; font-size: 13px;">{{ s.ai_targets }}</td>
                    <td>
                        {% if web_view_link %}
                        <a href="{{ web_view_link }}#tab-{{ s.symbol }}" target="_blank" class="link-text">🔍 閱讀報告</a>
                        {% else %}
                        --
                        {% endif %}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        
        {% if web_view_link %}
        <a href="{{ web_view_link }}" class="btn">開啟雲端分頁總報告</a>
        {% endif %}
        
        <div class="footer-note">
            💡 <b>個股連結功能：</b>點選表格中的股票名稱或「🔍 閱讀報告」，可直接連線至雲端硬碟並自動切換至該股的 K 線圖與詳細盤後分析報告。<br>
            所有分析內容僅供參考，投資人應獨立判斷並自負投資風險。
        </div>
    </div>
</body>
</html>
"""

def markdown_to_html(md_text):
    """
    簡單將 Markdown 文字轉換為 HTML 格式，以便在網頁排版中正確呈現
    """
    html = re.sub(r'^### (.*)', r'<h3>\1</h3>', md_text, flags=re.MULTILINE)
    html = re.sub(r'^- (.*)', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'((?:<li>.*</li>\s*)+)', r'<ul>\1</ul>', html)
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    html = html.replace('\n', '<br>')
    return html

# ==========================================
# 7. 主排程與協調模組
# ==========================================

def load_config():
    """
    讀取設定檔並與環境變數合併
    """
    config = {}
    if os.path.exists('config.json'):
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            print(f"讀取 config.json 失敗: {e}")
            
    keys = [
        'GEMINI_API_KEY', 'GMAIL_USER', 'GMAIL_APP_PASSWORD',
        'LINE_CHANNEL_ACCESS_TOKEN', 'LINE_USER_ID', 'GDRIVE_FOLDER_ID'
    ]
    for key in keys:
        val = os.environ.get(key)
        if val:
            config[key] = val
            
    return config

def main():
    print("====== 台股盤後整合分析系統啟動 ======")
    config = load_config()
    
    stock_list_path = '台股名單.csv'
    if not os.path.exists(stock_list_path):
        print(f"找不到台股名單：{stock_list_path}")
        return
        
    import csv
    stocks = []
    
    try:
        with open(stock_list_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_clean = {k.strip().replace('\ufeff', ''): v.strip() for k, v in row.items() if k is not None}
                if '代號' in row_clean and '名稱' in row_clean:
                    stocks.append({'代號': row_clean['代號'], '名稱': row_clean['名稱']})
    except Exception:
        try:
            with open(stock_list_path, 'r', encoding='big5') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_clean = {k.strip().replace('\ufeff', ''): v.strip() for k, v in row.items() if k is not None}
                    if '代號' in row_clean and '名稱' in row_clean:
                        stocks.append({'代號': row_clean['代號'], '名稱': row_clean['名稱']})
        except Exception as e:
            print(f"無法解析股票清單 CSV: {e}")
            return
            
    gdrive_service = get_gdrive_service()
    
    # 儲存所有個股的數據快照
    all_snapshots = []
    
    # 用於 LINE 摘要多空統計
    stat_summary = {
        'total': 0,
        'buys': [],
        'sells': [],
        'holds': []
    }
    
    # ----------------------------------------
    # 第一階段：本地數據下載與準備
    # ----------------------------------------
    prepared_stocks = []
    for idx, row in enumerate(stocks):
        symbol = str(row['代號']).strip()
        name = str(row['名稱']).strip()
        print(f"\n[{idx+1}/{len(stocks)}] 下載與準備數據: {symbol} {name}")
        
        try:
            # A. 下載與計算
            df, ticker_symbol = fetch_stock_data(symbol)
            df = calculate_technical_indicators(df)
            stats = calculate_interval_statistics(df)
            news = fetch_stock_news(ticker_symbol)
            
            latest_row = df.iloc[-1]
            prev_row = df.iloc[-2]
            
            change = latest_row['Close'] - prev_row['Close']
            change_percent = (change / prev_row['Close']) * 100
            
            # B. 繪製 K 線圖
            kline_base64 = generate_kline_chart_base64(df, symbol)
            
            # 格式化統計數據文字
            stats_text = ""
            for name_key, s in stats.items():
                stats_text += f"- {name_key}:\n"
                stats_text += f"  收盤價: 平均 {s['price_mean']:.2f}, 中位 {s['price_median']:.2f}, 最高 {s['price_max']:.2f}, 最低 {s['price_min']:.2f}\n"
                stats_text += f"  成交量: 平均 {s['volume_mean']:,.0f}, 中位 {s['volume_median']:,.0f}, 最高 {s['volume_max']:,.0f}, 最低 {s['volume_min']:,.0f}\n"
                
            # 格式化新聞文字
            news_text = ""
            if news:
                for n_idx, item in enumerate(news):
                    news_text += f"{n_idx+1}. {item.get('title')} (發布: {item.get('publisher')})\n"
            else:
                news_text = "目前無相關即時新聞。\n"
                
            prepared_stocks.append({
                'symbol': symbol,
                'name': name,
                'ticker_symbol': ticker_symbol,
                'latest_row': latest_row,
                'prev_row': prev_row,
                'change': change,
                'change_percent': change_percent,
                'latest_open': float(latest_row['Open']),
                'latest_close': float(latest_row['Close']),
                'latest_high': float(latest_row['High']),
                'latest_low': float(latest_row['Low']),
                'latest_volume': int(latest_row['Volume']),
                'ma5': float(latest_row['MA5']),
                'ma10': float(latest_row['MA10']),
                'ma20': float(latest_row['MA20']),
                'ma60': float(latest_row['MA60']),
                'bb_upper': float(latest_row['BB_Upper']),
                'bb_middle': float(latest_row['BB_Middle']),
                'bb_lower': float(latest_row['BB_Lower']),
                'bias5': float(latest_row['BIAS5']),
                'bias10': float(latest_row['BIAS10']),
                'bias20': float(latest_row['BIAS20']),
                'k_val': float(latest_row['K']),
                'd_val': float(latest_row['D']),
                'rsi6': float(latest_row['RSI6']),
                'rsi14': float(latest_row['RSI14']),
                'dif': float(latest_row['DIF']),
                'dem': float(latest_row['DEM']),
                'osc': float(latest_row['OSC']),
                'date': latest_row.name.strftime('%Y-%m-%d'),
                'stats': stats,
                'stats_text': stats_text,
                'news': news,
                'news_text': news_text,
                'kline_base64': kline_base64
            })
            
            # yfinance 下載完畢後稍作延遲，友善請求
            time.sleep(0.5)
            
        except Exception as e:
            print(f"準備個股 {symbol} {name} 數據時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            
    # ----------------------------------------
    # 第二階段：批次呼叫 Gemini 進行分析與拼裝
    # ----------------------------------------
    api_key = config.get('GEMINI_API_KEY')
    batch_size = 6
    batches = list(chunk_list(prepared_stocks, batch_size))
    
    for b_idx, batch in enumerate(batches):
        print(f"\n[批次 {b_idx+1}/{len(batches)}] 開始批次處理 AI 分析 (共 {len(batch)} 檔個股)...")
        
        ai_results = None
        if api_key and not api_key.startswith("YOUR_"):
            ai_results = analyze_batch_with_gemini(api_key, batch)
            
        # 建立 Symbol 對應 AI 回傳結果的對照表
        ai_map = {}
        if ai_results:
            for res in ai_results:
                symbol_key = str(res.get('symbol', '')).strip()
                
                # 取得今日收盤價，供後備計算
                latest_close_val = None
                for item in batch:
                    if item['symbol'] == symbol_key:
                        latest_close_val = float(item['latest_close'])
                        break
                
                # 清理與驗證價格（嚴禁 N/A, 無, --, 空白等，若出現則以收盤價為準進行後備計算）
                for price_key in ['entry_price', 'target_price', 'stop_loss_price']:
                    val = str(res.get(price_key, '')).strip().upper()
                    is_invalid = not val or any(x in val for x in ['N/A', '無', '--', 'NONE', 'NULL', '未提供', 'NAN'])
                    if is_invalid and latest_close_val is not None:
                        advice_val = res.get('advice', '觀望')
                        if price_key == 'entry_price':
                            calc_val = latest_close_val * 0.98 if advice_val == '觀望' else (latest_close_val * 0.90 if advice_val == '賣出' else latest_close_val)
                            res[price_key] = f"{calc_val:.2f}"
                        elif price_key == 'target_price':
                            calc_val = latest_close_val * 1.06 if advice_val == '觀望' else (latest_close_val * 0.99 if advice_val == '賣出' else latest_close_val * 1.08)
                            res[price_key] = f"{calc_val:.2f}"
                        elif price_key == 'stop_loss_price':
                            calc_val = latest_close_val * 0.95 if advice_val == '觀望' else (latest_close_val * 1.02 if advice_val == '賣出' else latest_close_val * 0.95)
                            res[price_key] = f"{calc_val:.2f}"
                
                ai_map[symbol_key] = res
                
        # 處理批次內的每一檔個股
        for item in batch:
            symbol = item['symbol']
            name = item['name']
            latest_row = item['latest_row']
            stats = item['stats']
            kline_base64 = item['kline_base64']
            change = item['change']
            change_percent = item['change_percent']
            
            try:
                ai_info = ai_map.get(symbol)
                
                if ai_info:
                    # 重新拼裝成原系統所使用的 Markdown 結構，以相容後續的 HTML 生成
                    ai_raw_analysis = f"""### 一、當日表現分析
{ai_info.get('analysis_today', '')}

### 二、明日開盤預測與動能評估
{ai_info.get('prediction_tomorrow', '')}

### 三、籌碼面與資券分析
{ai_info.get('chips_and_margin', '')}

### 四、買賣與目標價建議
- 交易建議：{ai_info.get('advice', '觀望')}
- 建議理由：{ai_info.get('reason', '')}
- 進場參考價：{ai_info.get('entry_price', '')} 元
- 停利目標價：{ai_info.get('target_price', '')} 元
- 停損參考價：{ai_info.get('stop_loss_price', '')} 元

### 五、短期風險提示
{ai_info.get('risk_warning', '')}"""
                    ai_advice = ai_info.get('advice', '觀望')
                    
                    # 目標價字串拼裝 (格式如: 2325.50 ➔ 2517.50 (損:2256.25))
                    entry_p = ai_info.get('entry_price', '')
                    target_p = ai_info.get('target_price', '')
                    stop_p = ai_info.get('stop_loss_price', '')
                    ai_targets = f"{entry_p} ➔ {target_p} (損:{stop_p})"
                else:
                    # 降級成規則技術分析
                    print(f"[{symbol} {name}] 未取得 AI 分析結果，降級至規則技術分析...")
                    ai_raw_analysis = generate_rule_based_analysis(name, item['ticker_symbol'], latest_row)
                    ai_advice, ai_targets = extract_ai_recommendation(ai_raw_analysis)
                    
                ai_analysis_html = markdown_to_html(ai_raw_analysis)
                
                # 分類個股策略
                if '買入' in ai_advice or '買進' in ai_advice:
                    stat_summary['buys'].append(f"{symbol} {name}")
                elif '賣出' in ai_advice:
                    stat_summary['sells'].append(f"{symbol} {name}")
                else:
                    stat_summary['holds'].append(f"{symbol} {name}")
                    
                # D. 建立並保存個股數據快照 (qa.py 會用到)
                snapshot = {
                    'symbol': symbol,
                    'name': name,
                    'date': item['date'],
                    'latest_close': float(item['latest_close']),
                    'change': float(change),
                    'change_percent': float(change_percent),
                    'latest_open': float(item['latest_open']),
                    'latest_high': float(item['latest_high']),
                    'latest_low': float(item['latest_low']),
                    'latest_volume': int(item['latest_volume']),
                    'k_val': float(item['k_val']),
                    'd_val': float(item['d_val']),
                    'rsi6': float(item['rsi6']),
                    'rsi14': float(item['rsi14']),
                    'stats': stats,
                    'ai_raw_analysis': ai_raw_analysis,
                    'ai_advice': ai_advice,
                    'ai_targets': ai_targets,
                    'kline_base64': kline_base64,
                    'gemini_analysis_html': ai_analysis_html
                }
                all_snapshots.append(snapshot)
                stat_summary['total'] += 1
                
                snapshot_filename = f"{symbol}_snapshot.json"
                with open(snapshot_filename, 'w', encoding='utf-8') as sf:
                    # 儲存不含大圖的快照以節省容量
                    light_snapshot = snapshot.copy()
                    if 'kline_base64' in light_snapshot:
                        del light_snapshot['kline_base64']
                    json.dump(light_snapshot, sf, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"處理個股 {symbol} {name} 時發生錯誤: {e}")
                import traceback
                traceback.print_exc()
                
        # 批次之間也稍作延遲，防止極端限流
        if b_idx < len(batches) - 1:
            time.sleep(15)
            
    if not all_snapshots:
        print("沒有成功分析任何股票，結束程序。")
        return
        
    # E. 生成整合型 HTML 報表
    print("\n所有個股數據與 AI 分析收集完畢，開始生成整合型分頁報告...")
    date_str = datetime.date.today().strftime('%Y-%m-%d')
    t_consolidated = Template(CONSOLIDATED_HTML_TEMPLATE)
    rendered_consolidated_html = t_consolidated.render(
        snapshots=all_snapshots,
        date=date_str,
        total_stocks=len(all_snapshots),
        stat_summary=stat_summary
    )
    
    report_filename = f"temp_consolidated_report.html"
    with open(report_filename, 'w', encoding='utf-8') as hf:
        hf.write(rendered_consolidated_html)
        
    # F. 上傳整合型報告至 Google Drive
    drive_folder_id = config.get('GDRIVE_FOLDER_ID')
    gdrive_filename = f"{date_str}_台股盤後智慧分析總報告.html"
    print("正在上傳分頁總報告至 Google 雲端硬碟...")
    web_view_link = upload_html_to_gdrive(
        gdrive_service,
        report_filename,
        drive_folder_id,
        gdrive_filename
    )
    if web_view_link:
        print(f"雲端總報告上傳成功: {web_view_link}")
    else:
        print("雲端總報告上傳失敗或被略過。")
        
    # G. 發送單一彙總 Gmail 電子報 (信中股票列可直接跳轉至雲端個股分頁)
    recipient_email = config.get('GMAIL_USER')
    gmail_pass = config.get('GMAIL_APP_PASSWORD')
    if recipient_email and not recipient_email.startswith("YOUR_"):
        t_gmail = Template(GMAIL_HTML_TEMPLATE)
        rendered_gmail_html = t_gmail.render(
            snapshots=all_snapshots,
            date=date_str,
            total_stocks=len(all_snapshots),
            web_view_link=web_view_link
        )
        subject = f"【台股盤後總報告】{date_str} - 已完成 {len(all_snapshots)} 檔個股分析摘要"
        print("正在發送彙總電子郵件...")
        send_gmail_summary(
            gmail_user=recipient_email,
            gmail_password=gmail_pass,
            recipient=recipient_email,
            subject=subject,
            html_content=rendered_gmail_html
        )
        
    # H. 發送單一彙總 LINE 推送訊息
    line_token = config.get('LINE_CHANNEL_ACCESS_TOKEN')
    line_user_id = config.get('LINE_USER_ID')
    if line_token and line_user_id:
        print("正在發送 LINE 彙總推送通知...")
        send_line_summary_notification(
            token=line_token,
            user_id=line_user_id,
            date_str=date_str,
            stats_summary=stat_summary,
            web_view_link=web_view_link
        )
        
    # 清理臨時文件
    if os.path.exists(report_filename):
        os.remove(report_filename)
        
    print("\n====== 整合式盤後自動分析任務全部結束 ======")

if __name__ == '__main__':
    main()
