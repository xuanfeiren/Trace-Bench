#!/bin/bash
# 断开 LiteLLM 连接
# 使用方法: ./disconnect_litellm.sh

echo "🛑 正在断开 LiteLLM 连接..."

if lsof -Pi :4000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    kill $(lsof -ti:4000)
    if [ $? -eq 0 ]; then
        echo "✅ 连接已断开"
    else
        echo "❌ 断开失败，请手动执行: kill \$(lsof -ti:4000)"
    fi
else
    echo "⚠️  没有检测到活动的连接"
fi
