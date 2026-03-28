#!/bin/bash
# 快速启动本地 SSH 隧道
# 使用方法: ./start_tunnel.sh

echo "🚀 正在建立 SSH 隧道到 EC2..."

# 先检查端口是否被占用
if lsof -Pi :30000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  端口 30000 已被占用，正在关闭..."
    kill $(lsof -ti:30000)
    sleep 1
fi

# 建立 SSH 隧道
ssh ubuntu@16.79.116.249 \
    -L 30000:127.0.0.1:30000 \
    -i api-server-setup/allen-2025-aws-personal.pem \
    -fN

if [ $? -eq 0 ]; then
    echo "✅ SSH 隧道已建立！"
    echo "📡 现在可以通过 http://127.0.0.1:30000 访问 LLM API"
    echo ""
    echo "测试命令："
    echo "curl http://127.0.0.1:30000/v1/chat/completions \\"
    echo "  -H \"Content-Type: application/json\" \\"
    echo "  -d '{\"model\": \"anthropic.claude-3-5-sonnet-20240620-v1:0\", \"messages\": [{\"role\": \"user\", \"content\": \"你好\"}]}'"
else
    echo "❌ SSH 隧道建立失败，请检查网络连接和 EC2 状态"
fi
