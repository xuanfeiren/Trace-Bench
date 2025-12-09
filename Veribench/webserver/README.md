# Lean Feedback Server

一个用于评估 Lean 4 代码的 Web 服务器，接收 Lean 代码并返回编译结果和反馈。

## 安装依赖

确保已运行项目的安装脚本，或者手动同步依赖：

```bash
cd /path/to/Veribench
uv sync --extra lean4
```

## 启动服务器

```bash
cd /path/to/Veribench
uv run python webserver/lean_feedback_server.py --port 8000
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 绑定的主机地址 |
| `--port` | `8000` | 绑定的端口 |
| `--reload` | `false` | 开发模式，自动重载 |

## API 端点

### `GET /` 和 `GET /health`

健康检查端点。

### `POST /feedback`

评估 Lean 代码并返回反馈。

**请求体：**

```json
{
    "lean_code": "def hello := 42",
    "remove_import_errors": true
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `lean_code` | string | 是 | 要评估的 Lean 4 代码 |
| `remove_import_errors` | bool | 否 | 是否自动移除导入错误（默认 true） |

**响应体：**

```json
{
    "score": 1.0,
    "feedback": "The answer is correct! No need to change anything.",
    "valid": true,
    "num_errors": 0,
    "error_messages": [],
    "error_details": []
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `score` | float | 分数（1.0 = 正确，0.0 = 错误） |
| `feedback` | string | 人类可读的反馈信息 |
| `valid` | bool | 代码是否编译成功 |
| `num_errors` | int | 错误数量 |
| `error_messages` | list[str] | 原始错误消息列表 |
| `error_details` | list[str] | 带上下文的详细错误信息 |

## 使用示例

### 使用 curl

```bash
# 测试正确的代码
curl -X POST "http://localhost:8000/feedback" \
     -H "Content-Type: application/json" \
     -d '{"lean_code": "def hello := 42"}'

# 测试错误的代码
curl -X POST "http://localhost:8000/feedback" \
     -H "Content-Type: application/json" \
     -d '{"lean_code": "/-!
# Binary Search Implementation in Lean 4

Implements binary search over a sorted list of integers.
Returns some index if found, none if not found.
-/

namespace BinarySearch

/-- Function to check if a list is sorted in ascending order -/
def isSorted (xs : List Int) : Bool :=
  match xs with
  | [] => true
  | [_] => true
  | x::y::rest => x ≤ y && isSorted (y::rest)

/-- Main binary search implementation -/
def binarySearch (arr : List Int) (target : Int) : Option Nat :=
  if arr.isEmpty then
    none
  else
    let rec search (left right : Nat) : Option Nat :=
      if left > right then
        none
      else
        let mid := (left + right) / 2
        match arr[mid]? with
        | none => none
        | some midVal =>
          if midVal = target then
            some mid
          else if midVal < target then
            search (mid + 1) right
          else
            search left (if mid = 0 then 0 else mid - 1)
    search 0 (arr.length - 1)

/-! Basic test cases -/
def testArray := [1, 2, 3, 4, 5]

#eval binarySearch testArray 3  -- expected: some 2
#eval binarySearch testArray 6  -- expected: none
#eval binarySearch ([] : List Int) 1  -- expected: none

/-! Edge cases -/
#eval binarySearch [5] 5  -- expected: some 0
#eval binarySearch [5] 3  -- expected: none

/-! Larger arrays -/
def largerArray := [10, 20, 30, 40, 50, 60]
#eval binarySearch largerArray 30  -- expected: some 2
#eval binarySearch largerArray 45  -- expected: none

/-! Test with duplicates -/
def duplicateArray := [1, 2, 3, 3, 3, 4, 5]
#eval binarySearch duplicateArray 3  -- expected: some index where value is 3

/-! Theorems -/

/-- Empty list always returns none -/
theorem empty_list_returns_none (target : Int) :
  binarySearch ([] : List Int) target = none := by native_decide

/-- Found element returns some index -/
theorem found_element_returns_index :
  binarySearch testArray 3 = some 2 := by native_decide

end BinarySearch"}'
```

### 在 Python 中调用

```python
import requests

def get_lean_feedback(lean_code: str, server_url: str = "http://localhost:8000") -> dict:
    """
    调用 Lean Feedback Server 获取代码评估结果。
    
    Args:
        lean_code: Lean 4 代码
        server_url: 服务器地址
        
    Returns:
        包含 score, feedback, valid, num_errors 等字段的字典
    """
    response = requests.post(
        f"{server_url}/feedback",
        json={"lean_code": lean_code, "remove_import_errors": True}
    )
    response.raise_for_status()
    return response.json()

# 使用示例
result = get_lean_feedback("def hello := 42")
print(f"Score: {result['score']}")
print(f"Valid: {result['valid']}")
print(f"Feedback: {result['feedback']}")
```

### 异步调用（使用 aiohttp）

```python
import aiohttp
import asyncio

async def get_lean_feedback_async(lean_code: str, server_url: str = "http://localhost:8000") -> dict:
    """异步调用 Lean Feedback Server。"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{server_url}/feedback",
            json={"lean_code": lean_code, "remove_import_errors": True}
        ) as response:
            return await response.json()

# 使用示例
async def main():
    result = await get_lean_feedback_async("def hello := 42")
    print(result)

asyncio.run(main())
```

## API 文档

启动服务器后，访问以下地址查看交互式 API 文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 运行测试

```bash
cd /path/to/Veribench
uv run python webserver/test_server.py
```

