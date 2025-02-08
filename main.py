import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
from typing import Union
from openai import OpenAI
import os
import tempfile


class DataAnalyzer:
    def __init__(self, api_key: str, api_type: str):
        self.api_key = api_key
        self.api_type = api_type
        self.df = None

    def load_data(self, file: Union[str, bytes]) -> bool:
        """加载数据文件"""
        try:
            if file.name.endswith('.xlsx'):
                self.df = pd.read_excel(file)
            elif file.name.endswith('.csv'):
                self.df = pd.read_csv(file)
            return True
        except Exception as e:
            st.error(f"文件加载错误: {str(e)}")
            return False

    def process_query(self, query: str) -> str:
        """处理用户查询"""
        if self.df is None:
            return "请先上传数据文件"

        system_prompt = f"""你是一个数据分析助手。当前数据集包含以下列：
{', '.join(self.df.columns)}
数据形状：{self.df.shape}

你的任务是：
1. 理解用户的数据分析需求
2. 生成相应的Python代码
3. 返回格式化的Markdown代码块，其中包含可执行的Python代码

注意：
- 使用 'df' 作为数据框的变量名
- 对于可视化，使用 plotly.express
- 代码应该可以直接在Streamlit环境中运行
- 如果需要显示图表，使用 st.plotly_chart()
"""

        try:
            if self.api_type == "DeepSeek":
                return self._call_deepseek(system_prompt, query)
            elif self.api_type == "Qwen":
                return self._call_qwen(system_prompt, query)
            elif self.api_type == "Kimi":
                return self._call_kimi(system_prompt, query)
            elif self.api_type == "豆包":
                return self._call_doubao(system_prompt, query)
            elif self.api_type == "智谱":
                return self._call_zhipu(system_prompt, query)
            else:
                return "不支持的API类型"
        except Exception as e:
            return f"API调用失败: {str(e)}"

    def _call_deepseek(self, system_prompt, query):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 0.1
        }
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        return response.json()["choices"][0]["message"]["content"]

    #
    # def _call_qwen(self, system_prompt, query):
    #     headers = {
    #         "Content-Type": "application/json",
    #         "Authorization": f"Bearer {self.api_key}"
    #     }
    #     data = {
    #         "model": "qwen-max",
    #         "messages": [
    #             {"role": "system", "content": system_prompt},
    #             {"role": "user", "content": query}
    #         ],
    #         "temperature": 0.1
    #     }
    #     response = requests.post(
    #         "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
    #         headers=headers,
    #         json=data
    #     )
    #     return response.json()["output"]["choices"][0]["message"]["content"]

    def _call_qwen(self, system_prompt, query):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": "deepseek-r1-distill-qwen-7b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 0.1
        }
        response = requests.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            headers=headers,
            json=data
        )
        return response.json()["choices"][0]["message"]["content"]

    def _call_kimi(self, system_prompt, query):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": "moonshot-v1-8k",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 0.1
        }
        response = requests.post(
            "https://api.moonshot.cn/v1/chat/completions",
            headers=headers,
            json=data
        )
        return response.json()["choices"][0]["message"]["content"]

    def _call_doubao(self, system_prompt, query):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": "skylark2-pro-4k",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 0.1
        }
        response = requests.post(
            "https://open.byteflowapi.com/api/v1/chat/completions",
            headers=headers,
            json=data
        )
        return response.json()["choices"][0]["message"]["content"]

    def _call_zhipu(self, system_prompt, query):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": "chatglm-pro",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 0.1
        }
        response = requests.post(
            "https://open.bigmodel.cn/api/paas/v3/chat/completions",
            headers=headers,
            json=data
        )
        return response.json()["choices"][0]["message"]["content"]


def main():
    st.title("📊 智能数据分析助手")

    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 配置")

        # API类型选择
        api_type = st.selectbox(
            "选择AI模型",
            ["DeepSeek", "Qwen", "Kimi", "豆包", "智谱"],
            index=0,
            help="选择需要使用的AI模型"
        )

        # 根据选择的API类型显示对应的输入框
        api_key = st.text_input(
            f"输入{api_type} API Key",
            type="password",
            help=f"输入{api_type}的API Key以启用AI分析功能"
        )

        uploaded_file = st.file_uploader(
            "上传数据文件",
            type=['csv', 'xlsx'],
            help="支持CSV和Excel文件"
        )

    # 初始化会话状态
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # 设置API key和加载数据
    if api_key and uploaded_file:
        if st.session_state.analyzer is None or \
                st.session_state.analyzer.api_type != api_type or \
                st.session_state.analyzer.api_key != api_key:
            st.session_state.analyzer = DataAnalyzer(api_key, api_type)

        if st.session_state.analyzer.df is None:
            if st.session_state.analyzer.load_data(uploaded_file):
                st.success("数据加载成功！")
                # 显示数据预览
                st.subheader("数据预览")
                st.dataframe(st.session_state.analyzer.df.head())

    # 聊天界面
    st.subheader("💬 与AI对话")

    # 用户输入
    user_input = st.chat_input("输入你的数据分析需求...")

    if user_input:
        if st.session_state.analyzer is None:
            st.error("请先配置API Key和上传数据文件！")
            return

        # 添加用户消息到历史
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # 获取AI响应
        with st.spinner("AI正在思考..."):
            response = st.session_state.analyzer.process_query(user_input)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

    # 显示聊天历史
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # 如果消息中包含代码块，则尝试执行
            if message["role"] == "assistant" and "```python" in message["content"]:
                code = message["content"].split("```python")[1].split("```")[0]
                try:
                    with st.expander("查看执行结果"):
                        exec(code)
                except Exception as e:
                    st.error(f"代码执行错误: {str(e)}")


if __name__ == "__main__":
    main()