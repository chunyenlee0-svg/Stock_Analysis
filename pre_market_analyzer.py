import os
import json
import datetime
import time
import csv
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd
import yfinance as yf
from google import genai
from google.genai import types
from google.oauth2 import service_account
from pydantic import BaseModel, Field
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from jinja2 import Template

# ==========================================
# 1. 基礎配置與設定檔載入
# ==========================================

def load_config():
    """
    載入設定檔並與環境變數合併
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

# ==========================================
# 2. 數據獲取與統計計算模組
# ==========================================

def fetch_global_markets():
    """
    下載前一晚美股指數與重要台股 ADR 收盤數據
    """
    tickers = {
        '^DJI': '道瓊工業指數',
        '^GSPC': '標普 500 指數',
        '^IXIC': '那斯達克指數',
        '^SOX': '費城半導體指數',
        'TSM': '台積電 ADR',
        'UMC': '聯電 ADR',
        'ASX': '日月光 ADR'
    }
    
    results = {}
    text_block = ""
    print("正在下載美股大盤及 ADR 數據...")
    for symbol, name in tickers.items():
        try:
            df = yf.download(symbol, period='5d')
            # 處理 yfinance 可能返回的 MultiIndex 欄位
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            if not df.empty and len(df) >= 2:
                latest_close = float(df['Close'].iloc[-1])
                prev_close = float(df['Close'].iloc[-2])
                change = latest_close - prev_close
                change_percent = (change / prev_close) * 100
                results[symbol] = f"{latest_close:.2f} ({change_percent:+.2f}%)"
                text_block += f"- {name} ({symbol}): 收盤價 {latest_close:.2f}, 漲跌幅 {change_percent:+.2f}%\n"
            else:
                results[symbol] = "N/A"
                text_block += f"- {name} ({symbol}): 無歷史數據\n"
        except Exception as e:
            print(f"下載全球指標 {symbol} 失敗: {e}")
            results[symbol] = "N/A"
            text_block += f"- {name} ({symbol}): 下載錯誤 {e}\n"
            
    return results, text_block

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
                'publisher': item.get('publisher', '未知媒體')
            })
    return formatted_news

# ==========================================
# 3. Gemini 智慧盤前分析模組
# ==========================================

class PreMarketSingleAnalysis(BaseModel):
    symbol: str = Field(description="股票代號，例如 '2330'")
    open_prediction: str = Field(description="今日開盤預測。必須是 '開高'、'開低' 或 '平盤' 之一，嚴禁填寫其他詞彙")
    prediction_reason: str = Field(description="預測理由。請詳細結合美股、費半指數、台股相關 ADR 漲跌，以及該股最新消息進行多空解讀。")
    pre_market_strategy: str = Field(description="今日盤前操作策略建議。給予具體交易策略方向（例如：拉回支撐布局、跌深反彈回補、開盤觀望不追高等）。")
    key_levels: str = Field(description="今日看點與關鍵點位（例如：觀察某均線支撐、某壓力位，或開盤跳空缺口等）。")

class BatchPreMarketResponse(BaseModel):
    analyses: list[PreMarketSingleAnalysis]

def analyze_pre_market_batch(api_key, global_text, batch_stocks):
    """
    呼叫 Gemini 進行批次盤前分析
    """
    if not api_key or api_key.startswith("YOUR_"):
        print("Gemini API Key 未設定，將採用規則盤前分析...")
        return None
        
    client = genai.Client(api_key=api_key)
    
    prompt = f"""你是一位專業的台股投資分析師與量化交易專家。請針對昨日美股與相關 ADR 的表現、最新市場消息，以及以下個股昨日技術指標狀態與最新消息，進行今日【盤前開盤預測與操作策略分析】。

【昨日全球市場與 ADR 表現】
{global_text}

請為以下每檔股票撰寫一份詳細的盤前專業預測報告。
請使用繁體中文（台灣習慣用語），並回傳符合 schema 定義的結構化 JSON 資料。

"""
    for item in batch_stocks:
        prompt += f"--- 股票：{item['name']} ({item['symbol']}) ---\n"
        prompt += f"- 昨日收盤價：{item['yesterday_close']:.2f} 元 (昨日漲跌幅: {item['yesterday_change_percent']:+.2f}%)\n"
        prompt += f"- 昨日技術指標狀態：{item['yesterday_tech']}\n"
        prompt += f"- 個股最新消息：\n{item['news_text']}\n\n"
        
    for attempt in range(5):
        try:
            print(f"正在對此批次個股進行盤前 AI 分析 (個股數: {len(batch_stocks)})...")
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=BatchPreMarketResponse,
                )
            )
            try:
                res_data = json.loads(response.text)
                return res_data.get('analyses', [])
            except Exception as je:
                print(f"解析 Gemini 盤前 JSON 失敗: {je}")
                print(f"原始內容: {response.text}")
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "503" in err_str or "quota" in err_str.lower() or "limit" in err_str.lower():
                sleep_time = (attempt + 1) * 20
                print(f"Gemini API 限流 ({err_str})，將於 {sleep_time} 秒後重試盤前批次分析...")
                time.sleep(sleep_time)
            else:
                print(f"Gemini API 批次呼叫嚴重錯誤: {e}")
                break
                
    return None

def generate_rule_based_pre_market(name, symbol, yesterday_close, yesterday_change_percent, global_data):
    """
    規則盤前分析 (當 API 失敗時的 Fallback 兜底方案)
    """
    # 讀取費半指數與 ADR 漲跌來做最基礎的硬規則預測
    sox_perf = 0.0
    tsm_perf = 0.0
    
    sox_str = global_data.get('^SOX', 'N/A')
    tsm_str = global_data.get('TSM', 'N/A')
    
    if '(' in sox_str and '%' in sox_str:
        try:
            sox_perf = float(sox_str.split('(')[-1].replace('%)', '').replace('+', ''))
        except: pass
    if '(' in tsm_str and '%' in tsm_str:
        try:
            tsm_perf = float(tsm_str.split('(')[-1].replace('%)', '').replace('+', ''))
        except: pass
        
    # 規則研判
    if tsm_perf > 1.0 or (tsm_perf >= 0 and sox_perf > 1.5):
        prediction = "開高"
        reason = f"昨日美股科技股與半導體族群走強，費半指數收漲 {sox_perf:+.2f}%，且台積電 ADR 大漲 {tsm_perf:+.2f}%，為今日台股電子權值股開盤提供強勁支撐與上漲動能。"
        strategy = "今日開盤預估偏多，若有跳空開高，短線不宜過度追高，可待開盤震盪拉回守住平盤或短均線支撐時再行分批布局。"
        levels = f"今日上檔壓力參考昨日收盤價之上 1.5% - 2.0% 位置；下檔防守支撐參考昨日收盤價 {yesterday_close:.2f} 元。"
    elif tsm_perf < -1.0 or (tsm_perf <= 0 and sox_perf < -1.5):
        prediction = "開低"
        reason = f"昨日美股表現低迷，費半指數下跌 {sox_perf:+.2f}%，且台積電 ADR 收跌 {tsm_perf:+.2f}%，預估將壓抑今日電子權值股表現，盤前面臨下修賣壓。"
        strategy = "預估今日開盤偏弱，盤前不宜盲目搶反彈。建議開盤後先觀察半小時，確認止跌訊號並守住關鍵均線（如 5日線或月線）後再考慮低吸。"
        levels = f"今日下檔關鍵防守支撐參考昨日收盤價之下 1.5% - 2.0% 位置；上檔壓力參考昨日收盤價 {yesterday_close:.2f} 元。"
    else:
        prediction = "平盤"
        reason = f"昨日美股四大指數與 ADR 呈現窄幅震盪整理，費半收 {sox_perf:+.2f}%，台積電 ADR 收 {tsm_perf:+.2f}%，無重大消息刺激，預期今日個股開盤以平盤附近震盪開出為主。"
        strategy = "今日預期以區間整理為主，短線操作建議不追高不殺低，採取逢低買進、逢高調節的區間策略。"
        levels = f"今日看點為昨日最高價與最低價區間，上檔壓力位 {yesterday_close * 1.01:.2f} 元，下檔支撐位 {yesterday_close * 0.99:.2f} 元。"
        
    return {
        'symbol': symbol,
        'open_prediction': prediction,
        'prediction_reason': reason,
        'pre_market_strategy': strategy,
        'key_levels': levels
    }

# ==========================================
# 4. Google Drive 儲存模組
# ==========================================

def get_gdrive_service():
    scopes = ['https://www.googleapis.com/auth/drive']
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    
    creds = None
    token_env = os.environ.get('GDRIVE_OAUTH_TOKEN')
    if token_env:
        try:
            creds_info = json.loads(token_env)
            creds = Credentials.from_authorized_user_info(creds_info, scopes)
        except Exception as e:
            print(f"解析環境變數 GDRIVE_OAUTH_TOKEN 錯誤: {e}")
            
    if not creds and os.path.exists('token.json'):
        try:
            creds = Credentials.from_authorized_user_file('token.json', scopes)
        except Exception as e:
            print(f"載入本機 token.json 錯誤: {e}")
            
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        except Exception as e:
            print(f"刷新 OAuth2 憑證失敗: {e}")
            creds = None
            
    if creds:
        try:
            return build('drive', 'v3', credentials=creds)
        except Exception as e:
            print(f"建立 Drive 服務失敗: {e}")
            
    # Fallback to Service Account
    sa_env = os.environ.get('GDRIVE_SERVICE_ACCOUNT')
    creds_dict = None
    if sa_env:
        try:
            creds_dict = json.loads(sa_env)
        except: pass
    if not creds_dict and os.path.exists('service_account.json'):
        try:
            with open('service_account.json', 'r', encoding='utf-8') as f:
                creds_dict = json.load(f)
        except: pass
    if creds_dict:
        try:
            creds_sa = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
            return build('drive', 'v3', credentials=creds_sa)
        except: pass
        
    return None

def upload_html_to_gdrive(service, file_path, folder_id, file_name):
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
            file = service.files().update(fileId=file_id, media_body=media, fields='id, webViewLink').execute()
        else:
            print("正在建立新檔案...")
            file = service.files().create(body=file_metadata, media_body=media, fields='id, webViewLink').execute()
            file_id = file.get('id')
            
        web_view_link = file.get('webViewLink')
        try:
            service.permissions().create(fileId=file_id, body={'type': 'anyone', 'role': 'reader'}).execute()
        except: pass
        return web_view_link
    except Exception as e:
        print(f"Google 雲端硬碟上傳失敗: {e}")
        return None

# ==========================================
# 5. Gmail 與 LINE 發送模組
# ==========================================

def send_gmail_summary(gmail_user, gmail_password, recipient, subject, html_content):
    if not gmail_user or not gmail_password or gmail_user.startswith("YOUR_"):
        print("Gmail 帳號或密碼未設定。跳過電子郵件發送。")
        return False
        
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = gmail_user
    msg['To'] = recipient
    
    part_html = MIMEText(html_content, 'html', 'utf-8')
    msg.attach(part_html)
    
    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(gmail_user, gmail_password)
        server.sendmail(gmail_user, [recipient], msg.as_string())
        server.close()
        print(f"盤前電子郵件成功發送至 {recipient}")
        return True
    except Exception as e:
        print(f"盤前電子郵件發送失敗: {e}")
        return False

def send_line_summary_notification(token, user_id, date_str, stats_summary, global_data, web_view_link):
    if not token or not user_id or token.startswith("YOUR_") or user_id.startswith("YOUR_"):
        print("LINE Channel Token 或 User ID 未設定。跳過 LINE 發送。")
        return False
        
    user_ids = [uid.strip() for uid in user_id.split(',') if uid.strip()]
    if not user_ids:
        return False
        
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    high_list = ", ".join(stats_summary['highs']) if stats_summary['highs'] else "無"
    low_list = ", ".join(stats_summary['lows']) if stats_summary['lows'] else "無"
    
    line_text = f"☀️ 台股盤前開盤智慧預測 ({date_str})\n" \
                f"今日已為您完成 {stats_summary['total']} 檔個股預測！\n\n" \
                f"🌎 美股及 ADR 收盤表現：\n" \
                f"• 費半指數 📈: {global_data.get('^SOX', 'N/A')}\n" \
                f"• 台積電 ADR: {global_data.get('TSM', 'N/A')}\n" \
                f"• 聯電 ADR: {global_data.get('UMC', 'N/A')}\n\n" \
                f"🔮 今日台股開盤預測：\n" \
                f"• 預估開高 📈：{high_list}\n" \
                f"• 預估開低 📉：{low_list}\n" \
                f"• 預估平盤 👁️：共 {len(stats_summary['flats'])} 檔\n\n"
                
    if web_view_link:
        line_text += f"🔗 盤前雲端報告連結：\n{web_view_link}"
    else:
        line_text += "（本次未成功上傳雲端硬碟報告，請確認 token 授權狀態）"
        
    success = True
    for uid in user_ids:
        payload = {
            "to": uid,
            "messages": [{"type": "text", "text": line_text}]
        }
        try:
            res = requests.post(url, headers=headers, json=payload)
            if res.status_code == 200:
                print(f"LINE 盤前訊息推送成功！(使用者: {uid})")
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

PRE_MARKET_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>台股盤前智慧預測總報告 - {{ date }}</title>
    <style>
        :root {
            --bg: #F0F4F8;
            --sidebar-bg: #0F172A;
            --sidebar-text: #94A3B8;
            --sidebar-hover-bg: #1E293B;
            --sidebar-active-bg: #3B82F6;
            --sidebar-active-text: #FFFFFF;
            --card-bg: #FFFFFF;
            --text-primary: #1E293B;
            --text-secondary: #64748B;
            --border-color: #E2E8F0;
            --up-color: #EF4444;
            --down-color: #10B981;
            --high-bg: #FEE2E2;
            --high-text: #EF4444;
            --low-bg: #D1FAE5;
            --low-text: #10B981;
            --flat-bg: #F1F5F9;
            --flat-text: #64748B;
        }
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg);
            color: var(--text-primary);
            margin: 0;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }
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
            border-bottom: 1px solid #1E293B;
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
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
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
        .grid-3 {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 24px;
        }
        .global-card {
            background: var(--card-bg);
            border-radius: 12px;
            border: 1px solid var(--border-color);
            padding: 16px;
            text-align: center;
        }
        .global-label {
            font-size: 12px;
            color: var(--text-secondary);
            font-weight: bold;
            margin-bottom: 5px;
        }
        .global-value {
            font-size: 20px;
            font-weight: 800;
            color: #0F172A;
        }
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
        .badge {
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: bold;
            display: inline-block;
        }
        .badge-high { background-color: var(--high-bg); color: var(--high-text); }
        .badge-low { background-color: var(--low-bg); color: var(--low-text); }
        .badge-flat { background-color: var(--flat-bg); color: var(--flat-text); }
        .btn-detail {
            background-color: #EFF6FF;
            color: var(--sidebar-active-bg);
            border: 1px solid #BFDBFE;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: bold;
            cursor: pointer;
        }
        .detail-item {
            padding: 16px;
            background-color: #F8FAFC;
            border-radius: 8px;
            border: 1px solid #EDF2F7;
            margin-bottom: 15px;
        }
        .detail-label {
            font-size: 13px;
            font-weight: bold;
            color: var(--sidebar-active-bg);
            margin-bottom: 6px;
        }
        .detail-value {
            font-size: 15px;
            line-height: 1.6;
        }
        @media (max-width: 768px) {
            body { flex-direction: column; overflow: auto; }
            .sidebar { width: 100%; height: auto; }
            .sidebar-menu { display: flex; overflow-x: auto; padding: 10px; }
            .menu-group-title { display: none; }
            .menu-item { margin-bottom: 0; margin-right: 8px; flex-shrink: 0; }
            .main-content { padding: 15px; height: auto; }
        }
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="sidebar-header">☀️ 盤前智慧預測</div>
        <div class="sidebar-menu">
            <a class="menu-item active" id="menu-dashboard" onclick="switchTab('dashboard')">📊 盤前儀表板</a>
            <div class="menu-group-title">預測個股</div>
            {% for s in snapshots %}
            <a class="menu-item" id="menu-tab-{{ s.symbol }}" onclick="switchTab('tab-{{ s.symbol }}')">🔍 {{ s.symbol }} {{ s.name }}</a>
            {% endfor %}
        </div>
    </div>
    
    <div class="main-content">
        <div id="dashboard" class="tab-pane active">
            <div class="header-section">
                <h1>全球市場與 ADR 盤前彙總</h1>
                <p>分析日期：{{ date }} | 已完成 {{ total_stocks }} 檔個股開盤預測</p>
            </div>
            
            <div class="grid-3">
                <div class="global-card" style="border-top: 4px solid #3B82F6;">
                    <div class="global-label">費城半導體 (SOX)</div>
                    <div class="global-value">{{ global_data['^SOX'] }}</div>
                </div>
                <div class="global-card" style="border-top: 4px solid var(--up-color);">
                    <div class="global-label">台積電 ADR (TSM)</div>
                    <div class="global-value">{{ global_data['TSM'] }}</div>
                </div>
                <div class="global-card" style="border-top: 4px solid var(--down-color);">
                    <div class="global-label">聯電 ADR (UMC)</div>
                    <div class="global-value">{{ global_data['UMC'] }}</div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">今日開盤預測與操作策略總覽</div>
                <div class="table-responsive">
                    <table>
                        <thead>
                            <tr>
                                <th class="text-left">股票名稱</th>
                                <th>昨日收盤價</th>
                                <th>開盤預測</th>
                                <th class="text-left">操作策略方向</th>
                                <th>看點</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for s in snapshots %}
                            <tr class="stock-row" onclick="switchTab('tab-{{ s.symbol }}')">
                                <td class="text-left font-bold" style="color: var(--sidebar-active-bg);">
                                    {{ s.symbol }} {{ s.name }}
                                </td>
                                <td class="font-bold">{{ "%.2f"|format(s.yesterday_close) }} 元</td>
                                <td>
                                    {% if s.open_prediction == '開高' %}
                                    <span class="badge badge-high">開高 📈</span>
                                    {% elif s.open_prediction == '開低' %}
                                    <span class="badge badge-low">開低 📉</span>
                                    {% else %}
                                    <span class="badge badge-flat">平盤 👁️</span>
                                    {% endif %}
                                </td>
                                <td class="text-left" style="font-size: 13px;">{{ s.pre_market_strategy }}</td>
                                <td><button class="btn-detail">詳細</button></td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        {% for s in snapshots %}
        <div id="tab-{{ s.symbol }}" class="tab-pane">
            <div class="header-section">
                <h1>{{ s.name }} ({{ s.symbol }}) 盤前智慧預測</h1>
                <p>昨日收盤價：{{ "%.2f"|format(s.yesterday_close) }} 元 ({{ "%+.2f"|format(s.yesterday_change_percent) }}%)</p>
            </div>
            
            <div class="card">
                <div class="card-title">開盤研判與策略</div>
                
                <div class="detail-item">
                    <div class="detail-label">開盤預測</div>
                    <div class="detail-value">
                        {% if s.open_prediction == '開高' %}
                        <span class="badge badge-high">開高 📈</span>
                        {% elif s.open_prediction == '開低' %}
                        <span class="badge badge-low">開低 📉</span>
                        {% else %}
                        <span class="badge badge-flat">平盤 👁️</span>
                        {% endif %}
                    </div>
                </div>
                
                <div class="detail-item">
                    <div class="detail-label">預測理由 (研判邏輯)</div>
                    <div class="detail-value">{{ s.prediction_reason }}</div>
                </div>
                
                <div class="detail-item">
                    <div class="detail-label">今日盤前操作策略</div>
                    <div class="detail-value">{{ s.pre_market_strategy }}</div>
                </div>
                
                <div class="detail-item">
                    <div class="detail-label">今日關鍵看點</div>
                    <div class="detail-value">{{ s.key_levels }}</div>
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
            if (target) target.classList.add('active');
            
            const menuItems = document.querySelectorAll('.menu-item');
            menuItems.forEach(item => item.classList.remove('active'));
            
            if (tabId === 'dashboard') {
                document.getElementById('menu-dashboard').classList.add('active');
            } else {
                const activeMenu = document.getElementById('menu-' + tabId);
                if (activeMenu) activeMenu.classList.add('active');
            }
            document.querySelector('.main-content').scrollTop = 0;
        }
        function checkHash() {
            const hash = window.location.hash;
            if (hash && hash.startsWith('#tab-')) {
                switchTab(hash.substring(1));
            } else {
                switchTab('dashboard');
            }
        }
        window.addEventListener('DOMContentLoaded', checkHash);
        window.addEventListener('hashchange', checkHash);
    </script>
</body>
</html>
"""

PRE_MARKET_GMAIL_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>台股盤前智慧預測速報</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #F8FAFC; margin: 0; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
        .header { background: linear-gradient(135deg, #0F172A, #1E293B); color: white; padding: 25px; border-radius: 8px; text-align: center; margin-bottom: 25px; }
        .header h2 { margin: 0 0 10px 0; font-size: 24px; }
        .summary-table { width: 100%; border-collapse: collapse; margin-bottom: 25px; font-size: 14px; }
        .summary-table th, .summary-table td { padding: 12px; border-bottom: 1px solid #e5e7eb; text-align: right; }
        .summary-table th { background-color: #F8FAFC; color: #475569; text-align: right; font-weight: bold; }
        .summary-table td.text-left, .summary-table th.text-left { text-align: left; }
        .badge { padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; display: inline-block; }
        .badge-high { background-color: #FEE2E2; color: #EF4444; }
        .badge-low { background-color: #D1FAE5; color: #10B981; }
        .badge-flat { background-color: #F1F5F9; color: #64748B; }
        .btn { display: block; width: 260px; margin: 30px auto 10px auto; padding: 14px 24px; background-color: #3B82F6; color: white !important; text-align: center; text-decoration: none; font-weight: bold; border-radius: 6px; }
        .footer-note { font-size: 12px; color: #64748B; text-align: center; margin-top: 30px; border-top: 1px solid #E2E8F0; padding-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>台股盤前智慧預測報告</h2>
            <p style="margin: 0; font-size: 14px; opacity: 0.9;">預測日期：{{ date }} | 已完成 {{ total_stocks }} 檔個股預測</p>
        </div>
        
        <table class="summary-table">
            <thead>
                <tr>
                    <th class="text-left">股票</th>
                    <th>昨日收盤</th>
                    <th>開盤預測</th>
                    <th class="text-left">盤前操作策略</th>
                </tr>
            </thead>
            <tbody>
                {% for s in snapshots %}
                <tr>
                    <td class="text-left" style="font-weight: bold; color: #1E293B;">
                        {% if web_view_link %}
                        <a href="{{ web_view_link }}#tab-{{ s.symbol }}" target="_blank" style="text-decoration: none; color: #3B82F6;">
                            {{ s.symbol }} {{ s.name }}
                        </a>
                        {% else %}
                        {{ s.symbol }} {{ s.name }}
                        {% endif %}
                    </td>
                    <td style="font-weight: bold;">{{ "%.2f"|format(s.yesterday_close) }} 元</td>
                    <td>
                        {% if s.open_prediction == '開高' %}
                        <span class="badge badge-high">開高</span>
                        {% elif s.open_prediction == '開低' %}
                        <span class="badge badge-low">開低</span>
                        {% else %}
                        <span class="badge badge-flat">平盤</span>
                        {% endif %}
                    </td>
                    <td class="text-left" style="color: #475569; font-size: 13px;">{{ s.pre_market_strategy }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        
        {% if web_view_link %}
        <a href="{{ web_view_link }}" class="btn" target="_blank">開啟雲端分頁總報告</a>
        {% endif %}
        
        <div class="footer-note">
            <p>※ 本報告內容由 AI 智慧分析生成，僅供投資參考，投資人應獨立判斷並自負投資風險。</p>
            <p>本信件為系統自動發送，請勿直接回信。</p>
        </div>
    </div>
</body>
</html>
"""

# ==========================================
# 7. 主程序進入點
# ==========================================

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    print("====== 台股盤前智慧預測系統啟動 ======")
    config = load_config()
    
    stock_list_path = '台股名單.csv'
    if not os.path.exists(stock_list_path):
        print(f"找不到台股名單：{stock_list_path}")
        return
        
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
            
    # 下載美股大盤與 ADR 數據
    global_results, global_text = fetch_global_markets()
    
    prepared_stocks = []
    for idx, row in enumerate(stocks):
        symbol = str(row['代號']).strip()
        name = str(row['名稱']).strip()
        
        # 盤前跳過全球 macro tickers (XAUD, NTBD, DINI) 不做個股預測，但會呈現在大盤彙總中
        if symbol in ['XAUD', 'NTBD', 'DINI']:
            continue
            
        print(f"\n[{idx+1}/{len(stocks)}] 下載與準備盤前數據: {symbol} {name}")
        
        # 優先讀取昨日盤後分析快照 snapshot
        snapshot_filename = f"{symbol}_snapshot.json"
        yesterday_close = 0.0
        yesterday_change_percent = 0.0
        yesterday_tech = "無快照指標"
        
        if os.path.exists(snapshot_filename):
            try:
                with open(snapshot_filename, 'r', encoding='utf-8') as sf:
                    snap_data = json.load(sf)
                    yesterday_close = float(snap_data.get('latest_close', 0.0))
                    yesterday_change_percent = float(snap_data.get('change_percent', 0.0))
                    
                    k_val = snap_data.get('k_val', 50.0)
                    d_val = snap_data.get('d_val', 50.0)
                    rsi6 = snap_data.get('rsi6', 50.0)
                    rsi14 = snap_data.get('rsi14', 50.0)
                    ai_advice = snap_data.get('ai_advice', '觀望')
                    yesterday_tech = f"收盤 {yesterday_close:.2f} 元 (昨日建議: {ai_advice}，KD: K={k_val:.1f}, D={d_val:.1f}，RSI-6={rsi6:.1f})"
            except Exception as e:
                print(f"讀取昨日快照 {snapshot_filename} 失敗: {e}")
                
        # 兜底：如果找不到快照，從 yfinance 下載昨日最後價格
        if yesterday_close == 0.0:
            try:
                ticker_symbol = f"{symbol}.TW"
                df = yf.download(ticker_symbol, period='5d')
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if not df.empty and len(df) >= 2:
                    yesterday_close = float(df['Close'].iloc[-1])
                    prev_close = float(df['Close'].iloc[-2])
                    yesterday_change_percent = ((yesterday_close - prev_close) / prev_close) * 100
                    yesterday_tech = f"收盤 {yesterday_close:.2f} 元，均線支撐月線附近。"
                else:
                    # 試上櫃
                    ticker_symbol = f"{symbol}.TWO"
                    df = yf.download(ticker_symbol, period='5d')
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    if not df.empty and len(df) >= 2:
                        yesterday_close = float(df['Close'].iloc[-1])
                        prev_close = float(df['Close'].iloc[-2])
                        yesterday_change_percent = ((yesterday_close - prev_close) / prev_close) * 100
                        yesterday_tech = f"收盤 {yesterday_close:.2f} 元，上櫃整理。"
            except Exception as e:
                print(f"下載 {symbol} 昨日價格失敗: {e}")
                continue
                
        # 獲取新聞
        ticker_symbol = f"{symbol}.TW" if not symbol.endswith('.TW') and not symbol.endswith('.TWO') else symbol
        news_list = fetch_stock_news(ticker_symbol)
        news_text = ""
        if news_list:
            for n_idx, item in enumerate(news_list):
                news_text += f"{n_idx+1}. {item.get('title')} (發布: {item.get('publisher')})\n"
        else:
            news_text = "無即時新聞。\n"
            
        prepared_stocks.append({
            'symbol': symbol,
            'name': name,
            'ticker_symbol': ticker_symbol,
            'yesterday_close': yesterday_close,
            'yesterday_change_percent': yesterday_change_percent,
            'yesterday_tech': yesterday_tech,
            'news_text': news_text
        })
        time.sleep(0.5)
        
    # 批次呼叫 Gemini 分析
    api_key = config.get('GEMINI_API_KEY')
    batch_size = 6
    batches = list(chunk_list(prepared_stocks, batch_size))
    
    all_predictions = []
    stat_summary = {
        'total': 0,
        'highs': [],
        'lows': [],
        'flats': []
    }
    
    for b_idx, batch in enumerate(batches):
        print(f"\n[盤前批次 {b_idx+1}/{len(batches)}] 開始處理 AI 開盤預測 (共 {len(batch)} 檔)...")
        
        ai_results = None
        if api_key and not api_key.startswith("YOUR_"):
            ai_results = analyze_pre_market_batch(api_key, global_text, batch)
            
        ai_map = {}
        if ai_results:
            for res in ai_results:
                symbol_key = str(res.get('symbol', '')).strip()
                ai_map[symbol_key] = res
                
        for item in batch:
            symbol = item['symbol']
            name = item['name']
            
            ai_info = ai_map.get(symbol)
            if ai_info:
                prediction = ai_info.get('open_prediction', '平盤')
                reason = ai_info.get('prediction_reason', '')
                strategy = ai_info.get('pre_market_strategy', '')
                levels = ai_info.get('key_levels', '')
            else:
                # 規則兜底
                print(f"[{symbol} {name}] 未取得 AI 盤前預測，使用規則推估兜底...")
                rule_res = generate_rule_based_pre_market(name, symbol, item['yesterday_close'], item['yesterday_change_percent'], global_results)
                prediction = rule_res['open_prediction']
                reason = rule_res['prediction_reason']
                strategy = rule_res['pre_market_strategy']
                levels = rule_res['key_levels']
                
            pred_item = {
                'symbol': symbol,
                'name': name,
                'yesterday_close': item['yesterday_close'],
                'yesterday_change_percent': item['yesterday_change_percent'],
                'open_prediction': prediction,
                'prediction_reason': reason,
                'pre_market_strategy': strategy,
                'key_levels': levels
            }
            all_predictions.append(pred_item)
            stat_summary['total'] += 1
            
            # 分類
            if prediction == '開高':
                stat_summary['highs'].append(f"{symbol} {name}")
            elif prediction == '開低':
                stat_summary['lows'].append(f"{symbol} {name}")
            else:
                stat_summary['flats'].append(f"{symbol} {name}")
                
        # 批次延遲防限流
        if b_idx < len(batches) - 1:
            time.sleep(25)
            
    if not all_predictions:
        print("沒有成功生成任何盤前預測，結束。")
        return
        
    # 生成 HTML 報表
    print("\n盤前預測收集完畢，開始生成整合型盤前報告...")
    date_str = datetime.date.today().strftime('%Y-%m-%d')
    t_html = Template(PRE_MARKET_HTML_TEMPLATE)
    rendered_html = t_html.render(
        snapshots=all_predictions,
        date=date_str,
        total_stocks=len(all_predictions),
        global_data=global_results
    )
    
    report_filename = "temp_pre_market_report.html"
    with open(report_filename, 'w', encoding='utf-8') as hf:
        hf.write(rendered_html)
        
    # 上傳至 Google Drive
    gdrive_service = get_gdrive_service()
    drive_folder_id = config.get('GDRIVE_FOLDER_ID')
    gdrive_filename = f"{date_str}_台股盤前智慧預測報告.html"
    print("正在上傳盤前總報告至 Google 雲端硬碟...")
    web_view_link = upload_html_to_gdrive(
        gdrive_service,
        report_filename,
        drive_folder_id,
        gdrive_filename
    )
    if web_view_link:
        print(f"雲端盤前報告上傳成功: {web_view_link}")
    else:
        print("雲端盤前報告上傳失敗。")
        
    # 發送 Gmail
    recipient_email = config.get('GMAIL_USER')
    gmail_pass = config.get('GMAIL_APP_PASSWORD')
    if recipient_email and not recipient_email.startswith("YOUR_"):
        t_gmail = Template(PRE_MARKET_GMAIL_TEMPLATE)
        rendered_gmail_html = t_gmail.render(
            snapshots=all_predictions,
            date=date_str,
            total_stocks=len(all_predictions),
            global_data=global_results,
            web_view_link=web_view_link
        )
        subject = f"【台股盤前預測速報】{date_str} - 已完成 {len(all_predictions)} 檔個股今日開盤分析"
        print("正在發送盤前電子郵件...")
        send_gmail_summary(
            gmail_user=recipient_email,
            gmail_password=gmail_pass,
            recipient=recipient_email,
            subject=subject,
            html_content=rendered_gmail_html
        )
        
    # 發送 LINE
    line_token = config.get('LINE_CHANNEL_ACCESS_TOKEN')
    line_user_id = config.get('LINE_USER_ID')
    if line_token and line_user_id:
        print("正在發送 LINE 盤前推送通知...")
        send_line_summary_notification(
            token=line_token,
            user_id=line_user_id,
            date_str=date_str,
            stats_summary=stat_summary,
            global_data=global_results,
            web_view_link=web_view_link
        )
        
    # 清理臨時文件
    if os.path.exists(report_filename):
        os.remove(report_filename)
        
    print("\n====== 盤前智慧預測分析任務全部結束 ======")

if __name__ == '__main__':
    main()
