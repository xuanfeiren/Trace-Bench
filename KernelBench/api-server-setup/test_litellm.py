#!/usr/bin/env python3
"""
测试 LiteLLM 调用 Claude 的脚本
使用方法：python test_litellm.py
"""

import os
from litellm import completion

# 如果使用 LiteLLM Proxy Server，设置 API base
# os.environ["LITELLM_API_BASE"] = "http://localhost:4000"
# os.environ["LITELLM_API_KEY"] = "sk-1234"

def test_direct_bedrock():
    """直接通过 LiteLLM 调用 Bedrock"""
    print("🧪 测试 1: 直接通过 LiteLLM 调用 Bedrock Claude")
    print("-" * 50)
    
    try:
        response = completion(
            model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            messages=[
                {"role": "user", "content": "用一句话介绍量子计算"}
            ],
            aws_region_name="us-east-1",
            max_tokens=200
        )
        
        print("✅ 调用成功！")
        print(f"回复: {response.choices[0].message.content}")
        print()
        
    except Exception as e:
        print(f"❌ 调用失败: {e}")
        print()


def test_litellm_proxy():
    """通过 LiteLLM Proxy Server 调用"""
    print("🧪 测试 2: 通过 LiteLLM Proxy Server 调用")
    print("-" * 50)
    print("⚠️  需要先启动 LiteLLM Proxy Server:")
    print("   litellm --config litellm_config.yaml --port 4000")
    print()
    
    try:
        # 设置 API base 指向 LiteLLM proxy
        response = completion(
            model="claude-3.7-sonnet",  # 使用配置文件中定义的模型名
            messages=[
                {"role": "user", "content": "Hello, who are you?"}
            ],
            api_base="http://localhost:4000",
            api_key="sk-1234",  # 对应 litellm_config.yaml 中的 master_key
            max_tokens=200
        )
        
        print("✅ 调用成功！")
        print(f"回复: {response.choices[0].message.content}")
        print()
        
    except Exception as e:
        print(f"❌ 调用失败: {e}")
        print("💡 提示: 确保 LiteLLM Proxy Server 正在运行")
        print()


def test_openai_compatible():
    """使用 OpenAI SDK 调用 LiteLLM Proxy"""
    print("🧪 测试 3: 使用 OpenAI SDK 调用 LiteLLM Proxy")
    print("-" * 50)
    
    try:
        from openai import OpenAI
        
        client = OpenAI(
            api_key="sk-1234",  # LiteLLM master_key
            base_url="http://localhost:4000"  # LiteLLM proxy URL
        )
        
        response = client.chat.completions.create(
            model="claude-3.7-sonnet",
            messages=[
                {"role": "user", "content": "用中文说你好"}
            ],
            max_tokens=100
        )
        
        print("✅ 调用成功！")
        print(f"回复: {response.choices[0].message.content}")
        print()
        
    except ImportError:
        print("⚠️  需要安装 openai: pip install openai")
        print()
    except Exception as e:
        print(f"❌ 调用失败: {e}")
        print()


if __name__ == "__main__":
    print("=" * 50)
    print("LiteLLM + Claude (Bedrock) 测试")
    print("=" * 50)
    print()
    
    # 测试 1: 直接调用
    test_direct_bedrock()
    
    # 测试 2 和 3: 通过 Proxy 调用（需要先启动 proxy server）
    print("💡 如果要测试 LiteLLM Proxy，请先在另一个终端运行:")
    print("   litellm --config api-server-setup/litellm_config.yaml --port 4000")
    print()
    
    user_input = input("是否测试 LiteLLM Proxy? (y/n): ")
    if user_input.lower() == 'y':
        test_litellm_proxy()
        test_openai_compatible()
    
    print("=" * 50)
    print("测试完成！")
    print("=" * 50)
