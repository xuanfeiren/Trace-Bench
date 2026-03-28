#!/bin/bash
# LiteLLM 快速连接脚本
# 使用方法: ./connect_litellm.sh

echo "🚀 LiteLLM 快速连接工具"
echo "=" * 50

# 检查是否已有隧道在运行
if lsof -Pi :4000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  端口 4000 已被占用"
    echo "可能已经有 SSH 隧道在运行，或者有其他程序占用了该端口"
    echo ""
    read -p "是否关闭现有进程并重新连接？(y/n): " choice
    if [ "$choice" = "y" ]; then
        echo "正在关闭现有进程..."
        kill $(lsof -ti:4000) 2>/dev/null
        sleep 1
    else
        echo "取消操作"
        exit 0
    fi
fi

# 建立 SSH 隧道
echo "🔌 正在建立 SSH 隧道到 EC2..."
ssh ubuntu@16.79.116.249 \
    -L 4000:127.0.0.1:4000 \
    -i api-server-setup/allen-2025-aws-personal.pem \
    -fN

if [ $? -eq 0 ]; then
    echo "✅ SSH 隧道已建立！"
    echo ""
    echo "📡 API 访问信息："
    echo "   Base URL: http://localhost:4000"
    echo "   API Key:  sk-1234"
    echo ""
    echo "🤖 可用模型："
    echo "   - claude-3.7-sonnet (Claude 3.7 Sonnet - 2025年2月最新版本)"
    echo "   - claude-3-5-sonnet (Claude 3.5 Sonnet - 2024年6月版本)"
    echo "   - claude-3-sonnet (Claude 3 Sonnet - 2024年2月版本)"
    echo "   - claude-3-haiku (Claude 3 Haiku - 更快更便宜)"
    echo ""
    echo "💡 使用示例："
    echo "   python api-server-setup/litellm_credentials.py"
    echo ""
    echo "🛑 关闭连接："
    echo "   ./disconnect_litellm.sh"
    echo "   或者: kill \$(lsof -ti:4000)"
else
    echo "❌ SSH 隧道建立失败"
    echo ""
    echo "可能的原因："
    echo "1. 网络连接问题"
    echo "2. EC2 服务器未运行"
    echo "3. SSH 密钥路径不正确"
    echo ""
    echo "请检查后重试"
fi
