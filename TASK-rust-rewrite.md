# 任务：claw-router Rust 重写

## 目标
把 Python FastAPI 的 claw-router v3.0 重写为 Rust，从 8 个参考项目中抄最好的设计。

## 当前 Python 版问题
- GIL 限制并发，httpx 连接池管理差
- 熔断器/重试逻辑手写 bug 多
- 无连接复用，无 health check 周期
- 智谱 /v4/ 路径需要 hack patch

## 参考项目（已 clone 到 D:/claude-workspace/）

### Rust（重点抄）
| 项目 | 路径 | 抄什么 |
|------|------|--------|
| **TensorZero** | `tensorzero/` | 网关核心: axum+tokio, 多Client连接池(http.rs), fallback链(model.rs:496-690), 重试(retries.rs), cancel-proof中间件 |
| **IronClaw** | `ironclaw/` | OpenClaw Rust 实现, WASM 沙箱, 隐私优先 |
| **OpenCrust** | `opencrust/` | OpenClaw 完全 Rust 重写 |
| **RustyClaw** | `RustyClaw/` | 轻量级 OS 级 AI 运行时 |
| **Debot** | `debot/` | 内置 Rust 智能路由器，按任务复杂度选模型 |

### 非 Rust（逻辑参考）
| 项目 | 路径 | 抄什么 |
|------|------|--------|
| **ClawRouter** | `ClawRouter/` | 44+ 模型路由逻辑, <1ms 路由 |
| **iblai-router** | `iblai-openclaw-router/` | 成本优化路由，节省 70%+ |
| **Bifrost** | `bifrost/` | Go 高性能参考, <100µs @ 5K RPS |

## 当前 Python 版配置（必须兼容）
- `D:/projects/claw-router/config/hubs.local.yaml` — 11 个免费模型 hub
- `D:/projects/claw-router/config/local.env` — API keys（ARK/GLM5/MiniMax）
- `D:/projects/claw-router/config/routes.yaml` — 路由规则

## TensorZero 架构分析要点（已完成）

### 最小 MVP 模块（~16K 行）
1. **HTTP Server** — axum 0.8 + tokio + mimalloc (~200行)
2. **Config** — TOML/YAML 解析, ModelConfig + routing (~500行)
3. **Provider trait** — InferenceProvider 4 方法 (~200行)
4. **OpenAI provider** — 覆盖 90% LLM API (~6000行)
5. **Fallback + Retry** — 顺序遍历 + backon 指数退避 (~400行)
6. **HTTP Client** — 多 reqwest::Client 绕 H2 100 并发限制 (~1000行)
7. **路由层** — /v1/chat/completions OpenAI 兼容 (~800行)
8. **Error + Types** — 错误和类型系统 (~1500行)

### 关键依赖
axum 0.8, reqwest 0.12, tokio 1.48, serde, backon, secrecy, tracing, mimalloc, moka

### 精华设计（必须抄）
1. **http.rs 多 Client 连接池** — 1024 个 reqwest::Client, 原子计数, RAII ticket
2. **Cancel-proof 中间件** — POST spawn 独立 task 防客户端断连
3. **三层超时** — Provider 级 → Model 级 → 全局
4. **重试包裹 fallback** — 每次重试走完整 fallback 链

## 功能需求
1. OpenAI 兼容 API (`/v1/chat/completions`, `/v1/models`)
2. YAML 配置 hub 定义（兼容现有 hubs.yaml 格式）
3. 环境变量 ${VAR} 解析
4. 顺序 fallback 链 + 指数退避重试
5. 熔断器（连续失败 N 次后暂停该 hub）
6. /v4/ 等非标准路径支持（智谱）
7. 管理 API: 添加/删除 hub, 热重载配置
8. Dashboard HTML（当前 Python 版有）
9. 健康检查（定期 ping upstream）
10. Prometheus metrics

## 部署目标
- 本地: localhost:3457（ShinkaEvolve 用）
- SUPER: :3456（2核2G，单 binary < 5MB）
- 新腾讯云: :3456

## 执行步骤
1. 读完 8 个参考项目的路由核心代码
2. 设计 claw-router-rs 的 crate 结构
3. 实现 MVP: config 加载 → HTTP server → OpenAI proxy → fallback
4. 兼容现有 hubs.yaml 配置
5. 测试: 用 batch_llm_extract.py 验证
6. 交叉编译部署到 SUPER (x86_64-unknown-linux-gnu)
