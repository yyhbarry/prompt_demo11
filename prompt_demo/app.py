from flask import Flask, request, jsonify
from flask_cors import CORS
from zhipuai import ZhipuAI

# --- 初始化 ---

# 初始化 Flask 应用
# __name__是当前模块名, static_folder指定静态文件目录, static_url_path是访问静态文件的URL前缀
app = Flask(__name__, static_folder='static', static_url_path='')
# 允许所有来源的跨域请求，方便本地开发
CORS(app)

# --- 配置信息 (API Key 硬编码) ---
ZHIPU_API_KEY = "885a748aaebb476983a8929931055a4e.gn8cI6KTZQJZp2ep"
GLM_MODEL_NAME = "glm-4"  # 使用 glm-4 模型，它在理解复杂指令和生成长文本方面更强大

# 检查API密钥是否存在 (虽然是硬编码，但保留此检查以防万一)
if not ZHIPU_API_KEY:
    raise ValueError("ZHIPU_API_KEY 未在代码中设置。")

# 初始化智谱AI客户端
try:
    client = ZhipuAI(api_key=ZHIPU_API_KEY)
    print("智谱AI客户端初始化成功。")
except Exception as e:
    print(f"智谱AI客户端初始化失败: {e}")
    client = None

# --- API 核心调用函数 ---
def get_glm_response(user_prompt: str):
    """
    接收用户提示，调用智谱AI模型并返回其响应。
    这是从您原始代码中提取并改造的核心功能。
    """
    if not client:
        return "错误：智谱AI客户端未初始化。"

    messages = [
        {"role": "user", "content": user_prompt}
    ]
    try:
        print(f"\n--- 正在向GLM ({GLM_MODEL_NAME}) 发送提示 ---")
        print(f"Prompt: {user_prompt[:200]}...") # 打印提示词前200个字符

        response = client.chat.completions.create(
            model=GLM_MODEL_NAME,
            messages=messages,
            temperature=0.7,  # 对于创造性/解释性任务，可以稍微提高温度
            max_tokens=2048
        )

        if response.choices and response.choices[0].message:
            content = response.choices[0].message.content
            print("--- GLM API 调用成功，已收到回复 ---")
            return content.strip()
        else:
            print("GLM API调用成功，但未返回有效内容。")
            return "抱歉，AI未能生成有效的回复。"
    except Exception as e:
        print(f"调用GLM API时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return f"调用AI服务时发生错误: {e}"

# --- Flask API 路由 ---
@app.route('/')
def serve_index():
    # 提供前端页面
    return app.send_static_file('index.html')

@app.route('/api/ask', methods=['POST'])
def handle_ask_request():
    """
    处理前端发送过来的提问请求。
    """
    # 检查请求是否为JSON格式
    if not request.is_json:
        return jsonify({"error": "请求必须是JSON格式"}), 400

    data = request.get_json()
    prompt = data.get('prompt')

    # 检查'prompt'字段是否存在
    if not prompt:
        return jsonify({"error": "请求中缺少'prompt'字段"}), 400

    # 调用AI模型获取回复
    ai_response = get_glm_response(prompt)

    # 将回复返回给前端
    return jsonify({"response": ai_response})

# --- 启动服务器 ---
if __name__ == '__main__':
    # host='0.0.0.0' 让局域网内的其他设备（如手机）也能访问
    # debug=True 方便开发调试，生产环境应设为 False
    # 注意：端口号从 5000 改为了 5006，请确保您的访问地址正确
    app.run(host='0.0.0.0', port=5006, debug=True)