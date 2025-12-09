#!/usr/bin/env python3
"""
Lean Feedback Server 测试脚本

使用方法：
    1. 先启动服务器：
       uv run python webserver/lean_feedback_server.py --port 8000
       
    2. 运行测试：
       uv run python webserver/test_server.py
"""

import requests
import sys

SERVER_URL = "http://localhost:8000"


def test_health():
    """测试健康检查端点"""
    print("=" * 50)
    print("测试: 健康检查")
    print("=" * 50)
    
    response = requests.get(f"{SERVER_URL}/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert data["status"] == "ok"
    print(f"✅ 健康检查通过: {data}")
    print()


def test_valid_lean_code():
    """测试正确的 Lean 代码"""
    print("=" * 50)
    print("测试: 正确的 Lean 代码")
    print("=" * 50)
    
    lean_code = "def hello := 42"
    print(f"输入代码: {lean_code}")
    
    response = requests.post(
        f"{SERVER_URL}/feedback",
        json={"lean_code": lean_code}
    )
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert data["score"] == 1.0, f"Expected score 1.0, got {data['score']}"
    assert data["valid"] == True, f"Expected valid True, got {data['valid']}"
    assert data["num_errors"] == 0, f"Expected 0 errors, got {data['num_errors']}"
    
    print(f"✅ Score: {data['score']}")
    print(f"✅ Valid: {data['valid']}")
    print(f"✅ Feedback: {data['feedback']}")
    print()


def test_invalid_lean_code():
    """测试错误的 Lean 代码"""
    print("=" * 50)
    print("测试: 错误的 Lean 代码")
    print("=" * 50)
    
    lean_code = "def hello := undefined_var"
    print(f"输入代码: {lean_code}")
    
    response = requests.post(
        f"{SERVER_URL}/feedback",
        json={"lean_code": lean_code}
    )
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert data["score"] == 0.0, f"Expected score 0.0, got {data['score']}"
    assert data["valid"] == False, f"Expected valid False, got {data['valid']}"
    assert data["num_errors"] > 0, f"Expected errors > 0, got {data['num_errors']}"
    
    print(f"✅ Score: {data['score']}")
    print(f"✅ Valid: {data['valid']}")
    print(f"✅ Num errors: {data['num_errors']}")
    print(f"✅ Error messages: {data['error_messages']}")
    print()


def test_complex_lean_code():
    """测试复杂的 Lean 代码"""
    print("=" * 50)
    print("测试: 复杂的 Lean 代码（带定理）")
    print("=" * 50)
    
    lean_code = """
theorem add_comm (a b : Nat) : a + b = b + a := by
  induction a with
  | zero => simp
  | succ n ih => simp [Nat.succ_add, ih]
"""
    print(f"输入代码:\n{lean_code}")
    
    response = requests.post(
        f"{SERVER_URL}/feedback",
        json={"lean_code": lean_code}
    )
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    print(f"Score: {data['score']}")
    print(f"Valid: {data['valid']}")
    print(f"Num errors: {data['num_errors']}")
    if data['valid']:
        print("✅ 定理证明成功!")
    else:
        print(f"❌ 错误信息: {data['error_messages']}")
    print()


def test_empty_code():
    """测试空代码"""
    print("=" * 50)
    print("测试: 空代码")
    print("=" * 50)
    
    response = requests.post(
        f"{SERVER_URL}/feedback",
        json={"lean_code": ""}
    )
    
    # 空代码应该返回 400 错误
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    print(f"✅ 空代码正确返回 400 错误")
    print()


def main():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("Lean Feedback Server 测试")
    print("=" * 50 + "\n")
    
    try:
        # 首先检查服务器是否在运行
        try:
            requests.get(f"{SERVER_URL}/health", timeout=2)
        except requests.exceptions.ConnectionError:
            print("❌ 错误: 服务器未运行!")
            print(f"请先启动服务器: uv run python webserver/lean_feedback_server.py --port 8000")
            sys.exit(1)
        
        # 运行所有测试
        test_health()
        test_valid_lean_code()
        test_invalid_lean_code()
        test_complex_lean_code()
        test_empty_code()
        
        print("=" * 50)
        print("✅ 所有测试通过!")
        print("=" * 50)
        
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 测试出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

