import os
import json
import glob
from google import genai
from google.genai import types

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
            
    keys = ['GEMINI_API_KEY']
    for key in keys:
        val = os.environ.get(key)
        if val:
            config[key] = val
            
    return config

def select_stock_snapshot():
    """
    列出當前目錄下所有的股票數據快照，並讓使用者選擇
    """
    snapshot_files = glob.glob('*_snapshot.json')
    if not snapshot_files:
        print("❌ 找不到任何股票分析快照檔案！")
        print("💡 請先執行 `python3 analyzer.py` 來下載並分析股票，系統會自動生成快照檔案。")
        return None
        
    print("\n=== 請選擇您想詢問的個股分析 ===")
    valid_snapshots = []
    
    for idx, filepath in enumerate(snapshot_files):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                valid_snapshots.append((filepath, data))
                print(f"[{idx + 1}] {data['symbol']} {data['name']} (分析日期: {data['date']})")
        except Exception as e:
            print(f"無法讀取快照檔案 {filepath}: {e}")
            
    if not valid_snapshots:
        print("❌ 沒有可用的有效快照檔案。")
        return None
        
    while True:
        try:
            choice = input(f"\n請輸入選單編號 (1-{len(valid_snapshots)})，或輸入 'q' 退出: ").strip()
            if choice.lower() == 'q':
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(valid_snapshots):
                return valid_snapshots[idx][1]
            else:
                print(f"⚠️ 請輸入 1 到 {len(valid_snapshots)} 之間的數字。")
        except ValueError:
            print("⚠️ 請輸入有效的數字。")

def start_qa_session(config, snapshot):
    """
    開啟與 Gemini 的互動式 QA 對話
    """
    api_key = config.get('GEMINI_API_KEY')
    if not api_key or api_key.startswith("YOUR_"):
        print("❌ 錯誤：未偵測到有效的 GEMINI_API_KEY。")
        print("💡 請在 `config.json` 中設定您的 API Key，或設定環境變數 `GEMINI_API_KEY`。")
        return
        
    print(f"\n正在連線 AI 程式，初始化 {snapshot['name']} ({snapshot['symbol']}) 的問答環境...")
    
    # 建立系統指導語
    system_instruction = f"""
你是一位專業的台股智慧理財助理。使用者剛剛閱讀了個股「{snapshot['name']} ({snapshot['symbol']})」在 {snapshot['date']} 的盤後分析報告。
以下是該股當日的完整數據快照與 AI 盤後分析結論，作為你回答問題的上下文脈絡 (Context)：

【個股當日基本數據】
- 收盤價：{snapshot['latest_close']} 元 (今日漲跌: {snapshot['change']:.2f} 元, 漲跌幅: {snapshot['change_percent']:+.2f}%)
- 開/高/低：{snapshot['latest_open']} / {snapshot['latest_high']} / {snapshot['latest_low']} 元
- 成交量：{snapshot['latest_volume']:,} 股
- 技術指標：K={snapshot['k_val']:.2f}, D={snapshot['d_val']:.2f}, RSI6={snapshot['rsi6']:.2f}, RSI14={snapshot['rsi14']:.2f}

【歷史統計對比（區間包含：1週、1月、1季、半年、1年）】
{json.dumps(snapshot['stats'], ensure_ascii=False, indent=2)}

【原 AI 盤後分析報告內容】
{snapshot['ai_raw_analysis']}

請遵循以下規範回答使用者的追問：
1. 請以親切、客觀、專業的態度回答。
2. 優先基於提供的個股數據快照與原分析報告內容進行回答。
3. 若使用者問及指標的學術定義（例如：「什麼是KD黃金交叉？」），請用深入淺出的方式向其解釋，並對照當前的數值做具體解說。
4. 提供投資分析時，請加註免責聲明，提醒使用者「所有內容僅供參考，投資人應獨立判斷並自負投資風險」。
5. 請一律使用繁體中文（台灣習慣用語，例如：個股、收盤價、黃金交叉等）進行回答。
"""

    try:
        client = genai.Client(api_key=api_key)
        chat = client.chats.create(
            model="gemini-flash-latest",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7
            )
        )
        
        print("\n==================================================")
        print(f"🤖 AI 投資助理已就位！您現在可以開始詢問關於 {snapshot['name']} 的問題了。")
        print("輸入 'exit' 或 'quit' 可退出對話，輸入 'clear' 可清除畫面。")
        print("==================================================")
        
        while True:
            try:
                user_question = input("\n👤 您：").strip()
                if not user_question:
                    continue
                if user_question.lower() in ['exit', 'quit']:
                    print("👋 感謝使用，祝您投資順利！")
                    break
                if user_question.lower() == 'clear':
                    os.system('clear' if os.name != 'nt' else 'cls')
                    continue
                    
                print("🤖 正在思考中...")
                response = chat.send_message(user_question)
                print(f"\n🤖 AI：\n{response.text}")
                
            except KeyboardInterrupt:
                print("\n👋 對話已強制終止。")
                break
            except Exception as chat_err:
                print(f"\n❌ 對話過程中發生錯誤: {chat_err}")
                
    except Exception as init_err:
        print(f"❌ 初始化 AI 會話失敗: {init_err}")

def main():
    print("====== 台股盤後分析 - 互動問答系統 ======")
    config = load_config()
    
    # 選擇股票快照
    snapshot = select_stock_snapshot()
    if snapshot:
        start_qa_session(config, snapshot)

if __name__ == '__main__':
    main()
