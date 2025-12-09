"""
Lean Feedback Client

提供便捷的 Python 客户端，用于调用 Lean Feedback Server。

使用示例:
    from webserver.client import LeanFeedbackClient
    
    client = LeanFeedbackClient("http://localhost:8000")
    result = client.get_feedback("def hello := 42")
    print(result.score, result.valid)
"""

import requests
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FeedbackResult:
    """反馈结果数据类"""
    score: float
    feedback: str
    valid: bool
    num_errors: int
    error_messages: List[str]
    error_details: List[str]
    
    @property
    def is_correct(self) -> bool:
        """代码是否正确"""
        return self.valid and self.score == 1.0


class LeanFeedbackClient:
    """Lean Feedback Server 客户端"""
    
    def __init__(self, server_url: str = "http://localhost:8000", timeout: int = 60):
        """
        初始化客户端。
        
        Args:
            server_url: 服务器地址
            timeout: 请求超时时间（秒）
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
    
    def health_check(self) -> bool:
        """
        检查服务器是否健康。
        
        Returns:
            服务器是否正常运行
        """
        try:
            response = requests.get(
                f"{self.server_url}/health",
                timeout=self.timeout
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def get_feedback(
        self,
        lean_code: str,
        remove_import_errors: bool = True
    ) -> FeedbackResult:
        """
        获取 Lean 代码的反馈。
        
        Args:
            lean_code: Lean 4 代码
            remove_import_errors: 是否自动移除导入错误
            
        Returns:
            FeedbackResult 对象，包含评估结果
            
        Raises:
            requests.exceptions.RequestException: 请求失败时抛出
            ValueError: 服务器返回错误时抛出
        """
        response = requests.post(
            f"{self.server_url}/feedback",
            json={
                "lean_code": lean_code,
                "remove_import_errors": remove_import_errors
            },
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            error_detail = response.json().get("detail", "Unknown error")
            raise ValueError(f"Server error ({response.status_code}): {error_detail}")
        
        data = response.json()
        return FeedbackResult(
            score=data["score"],
            feedback=data["feedback"],
            valid=data["valid"],
            num_errors=data["num_errors"],
            error_messages=data["error_messages"],
            error_details=data["error_details"]
        )
    
    def get_score(self, lean_code: str) -> float:
        """
        获取 Lean 代码的分数（简化版）。
        
        Args:
            lean_code: Lean 4 代码
            
        Returns:
            分数（1.0 = 正确，0.0 = 错误）
        """
        result = self.get_feedback(lean_code)
        return result.score
    
    def is_valid(self, lean_code: str) -> bool:
        """
        检查 Lean 代码是否有效（简化版）。
        
        Args:
            lean_code: Lean 4 代码
            
        Returns:
            代码是否编译成功
        """
        result = self.get_feedback(lean_code)
        return result.valid


# 便捷函数
def get_lean_feedback(
    lean_code: str,
    server_url: str = "http://localhost:8000"
) -> FeedbackResult:
    """
    获取 Lean 代码的反馈（便捷函数）。
    
    Args:
        lean_code: Lean 4 代码
        server_url: 服务器地址
        
    Returns:
        FeedbackResult 对象
    """
    client = LeanFeedbackClient(server_url)
    return client.get_feedback(lean_code)


# 使用示例
if __name__ == "__main__":
    # 创建客户端
    client = LeanFeedbackClient("http://localhost:8000")
    
    # 检查服务器状态
    if not client.health_check():
        print("❌ 服务器未运行!")
        print("请先启动: uv run python webserver/lean_feedback_server.py --port 8000")
        exit(1)
    
    print("✅ 服务器正常运行\n")
    
    # 测试正确的代码
    print("测试正确的代码:")
    result = client.get_feedback("def hello := 42")
    print(f"  Score: {result.score}")
    print(f"  Valid: {result.valid}")
    print(f"  Is Correct: {result.is_correct}")
    print()
    
    # 测试错误的代码
    print("测试错误的代码:")
    result = client.get_feedback("def hello := undefined_var")
    print(f"  Score: {result.score}")
    print(f"  Valid: {result.valid}")
    print(f"  Errors: {result.error_messages}")
    print()
    
    # 使用便捷函数
    print("使用便捷函数:")
    result = get_lean_feedback("def x := 1 + 1")
    print(f"  Result: {result}")

