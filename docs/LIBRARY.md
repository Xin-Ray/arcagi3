# LIBRARY — 函数索引(API Reference)

最近一次更新:2026-04-27(Phase 1 起步 + observation + llm + LLMAgent:5 个模块入库,34 个 pytest 全过)

所有可复用代码都放在 `arc_agent/` 包里,**每个函数都必须在本文件登记一条**。
本文件是"先调用、再创建"工作流的目录:写新代码前先在这里搜一遍,有就直接 `from arc_agent.X import Y` 调用。

---

## 库优先开发流程(强制)

每次要写一段逻辑前,按顺序执行:

1. **搜本文件**:有没有现成函数能用?有 → 直接 import 调用,跳过 2–5。
2. **决定归属**:这段逻辑该进哪个子模块?(见下面"模块划分")。一次性的脚本逻辑可以留在脚本里;**任何会被第二次调用的逻辑必须进库**。
3. **实现**:在对应的 `arc_agent/<module>.py` 写函数。保持类型注解、docstring 一行、纯函数优先。
4. **测试**:同步在 `tests/test_<module>.py` 写测试,至少覆盖正常输入 + 一个 edge case。可以 `pytest tests/test_<module>.py::test_xxx -q` 单跑。
5. **登记**:在本文件对应模块下追加一条函数条目(模板见下)。
6. **更新 `docs/CODE_MAP.md`**:如果是新模块,在 CODE_MAP 加一行;旧模块只更新本文件即可。

新增/修改/废弃函数都要更新本文件的"最近一次更新"日期。**废弃** 不删,改 `Status` 为 `deprecated` 并写替代品名字。

---

## 函数条目模板

```markdown
### `module.function_name`
- **状态**:stable / experimental / deprecated(若 deprecated 写 → 替代品)
- **用途**:一句话说做什么
- **签名**:`def function_name(x: T1, y: T2) -> R`
- **输入**:
  - `x` (T1): 约束 / 取值范围 / 形状
  - `y` (T2): 约束
- **输出**:`R`,含义和形状
- **依赖**:库内调用了 `module_a.func_a`;外部库用了 `numpy.where`
- **测试**:`tests/test_module.py::test_function_name`
- **添加 / 最后修改**:2026-04-27 / 2026-04-27
- **备注**(可选):性能、坑、TODO
```

---

## 模块划分(`arc_agent/` 计划)

每个子模块对应 `ARCHITECTURE.md` 里的一个职责。完整文件清单见 `CODE_MAP.md`,本表只列模块级粒度。

| 子模块 | 职责 | 何时建立 |
|--------|------|----------|
| `arc_agent/observation.py` | 观测帧序列化、diff、特征提取 | Phase 1 |
| `arc_agent/agents/` | Agent 实现:`random.py`、`llm.py`(开发期)、`hybrid.py`(主力) | Phase 1–2 |
| `arc_agent/data/` | 数据加载 + 增广:`human_traces.py`、`silver_rules.py` | Phase 1 末 |
| `arc_agent/vlm_backbone.py` | Qwen2.5-VL-3B-Instruct 加载 + LoRA + generate() | Phase 2 |
| `arc_agent/agents/vlm.py` | VLMAgent:图像+文本双模态 HEI agent | Phase 2 |
| `arc_agent/train_bc.py` | Stage A QLoRA BC 训练(Qwen2.5-VL-3B) | Phase 2 |
| `arc_agent/train_rl.py` | Stage B 离线 RL 微调(PPO + KL) | Phase 3 |
| `arc_agent/intrinsic_reward.py` | dense intrinsic reward 计算 | Phase 3 |
| `arc_agent/online_adapt.py` | Stage C 在线 LoRA 测试时微调 | Phase 3 |
| `arc_agent/search.py` | UNDO + 浅 BFS / MCTS(可选) | Phase 3 |
| `arc_agent/llm.py` | API LLM 调用封装(开发期用,prompt cache / 重试 / token 计数) | Phase 1 |
| `arc_agent/memory.py` | per-episode / cross-level 三元组缓存 | Phase 1–2 |
| `arc_agent/eval.py` | 批量跑 episode、scorecard 聚合、结果落盘 | Phase 0 末 |
| `arc_agent/utils.py` | 通用小工具(日志、计时、jsonl 读写) | 按需 |

---

## 函数清单

下面按子模块组织。

### `arc_agent/runner.py`

#### `arc_agent.runner.Agent`(Protocol)
- **状态**:stable
- **用途**:Agent 接口 — 实现一个 `choose(latest, history) -> GameAction` 方法即可
- **签名**:`class Agent(Protocol): def choose(self, latest: FrameDataRaw, history: list[FrameDataRaw]) -> GameAction: ...`
- **依赖**:`arcengine.FrameDataRaw`, `arcengine.GameAction`
- **测试**:`tests/test_runner.py::test_counting_agent_satisfies_protocol`,`tests/test_agents_random.py::test_random_satisfies_agent_protocol`
- **添加 / 最后修改**:2026-04-27 / 2026-04-27

#### `arc_agent.runner.play_one`
- **状态**:stable
- **用途**:跑单局,返回 `{actions, final_state, levels_completed, total_levels, guid}` 或 `{error, ...}`
- **签名**:`def play_one(arc: Arcade, agent: Agent, game_id: str, card_id: str, max_actions: int = 80, history_limit: int = 8) -> dict`
- **输入**:
  - `arc`:已构造的 `arc_agi.Arcade`(库不读 env / API key)
  - `agent`:满足 `Agent` Protocol 的实例
  - `game_id`:**完整** ID(如 `"ls20-9607627b"`),来自 `arc.get_environments()`
  - `card_id`:已 open 的 scorecard ID;**库不负责 open/close**
  - `max_actions`:单局上限(默认 80)
  - `history_limit`:传给 agent 的历史帧数量上限(默认 8)
- **输出**:dict;成功路径含 `actions / final_state / levels_completed / total_levels / guid`;失败含 `error`
- **依赖**:`arc_agi.Arcade`, `arcengine.{FrameDataRaw, GameAction, GameState}`
- **测试**:`tests/test_runner.py::{test_play_one_returns_error_on_make_none, test_play_one_stops_at_max_actions, test_play_one_stops_on_win}`
- **添加 / 最后修改**:2026-04-27 / 2026-04-27
- **备注**:库代码无 file/env I/O — 调用方负责 scorecard 生命周期和日志落盘

### `arc_agent/agents/random.py`

#### `arc_agent.agents.random.RandomAgent`
- **状态**:stable(基线,绝对下界)
- **用途**:NOT_PLAYED/GAME_OVER 时返回 RESET,否则均匀随机其它动作;ACTION6 配随机 (x,y)
- **签名**:`class RandomAgent: def __init__(self, seed: int | None = None); def choose(self, latest, history) -> GameAction`
- **依赖**:`random.Random`, `arcengine.{GameAction, GameState}`
- **测试**:`tests/test_agents_random.py`(5 个测试,覆盖 RESET 触发、不在 NOT_FINISHED 选 RESET、ACTION6 坐标合法、Protocol 满足)
- **添加 / 最后修改**:2026-04-27 / 2026-04-27
- **备注**:可再现:`RandomAgent(seed=42)`

### `arc_agent/observation.py`

5 个纯函数把 `FrameDataRaw` 转成 LLM-readable 文本。**核心 16 色 → hex char 0..F 编码**。

#### `arc_agent.observation.latest_grid`
- **状态**:stable
- **签名**:`def latest_grid(frame: FrameDataRaw) -> np.ndarray`
- **输入/输出**:取动画的最后一帧 (H, W) 数组,即 step 后稳定状态
- **测试**:`tests/test_observation.py::{test_latest_grid_returns_last_animation_frame, test_latest_grid_raises_on_empty_animation}`

#### `arc_agent.observation.grid_to_text`
- **状态**:stable
- **签名**:`def grid_to_text(grid: np.ndarray) -> str`
- **输入/输出**:(H, W) 数组 → H 行 × W 字符 hex 字符串。值 clip 到 [0, 15]
- **测试**:`test_grid_to_text_{small, clips_out_of_range, rejects_non_2d}`

#### `arc_agent.observation.grid_diff`
- **状态**:stable
- **签名**:`def grid_diff(prev: np.ndarray, curr: np.ndarray) -> list[tuple[int, int, int, int]]`
- **输入/输出**:对比两帧,返回变化的 cells `[(row, col, old, new), ...]`
- **测试**:`test_grid_diff_{lists_changed_cells, empty_when_identical, rejects_shape_mismatch}`

#### `arc_agent.observation.available_action_names`
- **状态**:stable
- **签名**:`def available_action_names(frame: FrameDataRaw) -> list[str]`
- **输入/输出**:把 `frame.available_actions` 的 int IDs 转成 `["ACTION1", "ACTION2", ...]`
- **测试**:`test_available_action_names_maps_ids`

#### `arc_agent.observation.summarize_frame`
- **状态**:stable
- **签名**:`def summarize_frame(frame: FrameDataRaw, *, include_grid: bool = True, diff_with: np.ndarray | None = None) -> str`
- **输入/输出**:一站式生成 prompt 用文本(state / levels / available_actions / 可选 grid / 可选 diff)
- **测试**:`test_summarize_frame_{includes_state_and_grid, can_omit_grid, with_diff}`
- **依赖**:`grid_to_text`, `grid_diff`, `latest_grid`, `available_action_names`

#### `arc_agent.observation.grid_to_image`
- **状态**:experimental(Phase 2 起用,Phase 1 不需要)
- **用途**:把 grid 渲染成 PIL Image 供 Qwen2.5-VL-3B vision encoder 使用
- **签名**:`def grid_to_image(grid: np.ndarray, scale: int = 8) -> PIL.Image.Image`
- **输入**:`grid` (H,W) int 数组值域 0–15;`scale` 每 cell 像素边长,默认 8 → 64×64 → 512×512
- **输出**:RGB PIL Image,使用 ARC 标准 16 色调色板
- **依赖**:Pillow(`PIL.Image`),numpy
- **测试**:待添加 `tests/test_observation.py::test_grid_to_image_shape_and_colors`
- **备注**:ARC 色 0=黑,1=蓝,2=红,3=绿,4=黄,5=灰,6=品红,7=橙,8=浅蓝,9=深红;10–15 灰阶占位

**添加 / 最后修改**:2026-04-27 / 2026-04-28

### `arc_agent/llm.py`

#### `arc_agent.llm.LLMClient`
- **状态**:stable
- **用途**:Anthropic SDK 封装,顶层 `cache_control={"type":"ephemeral"}` 自动缓存 system prompt
- **签名**:`class LLMClient: def __init__(self, *, model="claude-opus-4-7", max_tokens=4096, client=None); def complete(self, *, system: str, user: str) -> LLMResponse`
- **依赖**:`anthropic.Anthropic`
- **测试**:`tests/test_llm.py`(5 个测试)
- **添加 / 最后修改**:2026-04-27 / 2026-04-27
- **备注**:默认 backbone `claude-opus-4-7`(per claude-api skill);开发期可 `LLMClient(model="claude-haiku-4-5")` 省钱

#### `arc_agent.llm.LLMResponse` (dataclass)
- **状态**:stable
- **字段**:`text / input_tokens / output_tokens / cache_read_input_tokens / cache_creation_input_tokens`,property `cached_fraction`

### `arc_agent/agents/llm.py`

#### `arc_agent.agents.llm.LLMAgent`
- **状态**:stable(Phase 1 第一版)
- **用途**:用 Claude 选动作,prompt-level HEI(Hypothesis/Execute/Iterate 三段式 + `ACTION:` 行)
- **签名**:`class LLMAgent: def __init__(self, llm=None, *, seed=None); def choose(self, latest, history) -> GameAction; def reset(self) -> None`
- **回退**:LLM 失败 / 解析失败 / 非法动作 → random over `available_actions`(`_state.parse_failures` 计数)
- **依赖**:`arc_agent.llm.LLMClient`, `arc_agent.observation.summarize_frame`, `arcengine.GameAction`
- **测试**:`tests/test_agents_llm.py`(8 个测试)
- **添加 / 最后修改**:2026-04-27 / 2026-04-27

### `arc_agent/memory.py`

(空)

### `arc_agent/world_model.py`

(空)

### `arc_agent/search.py`

(空)

### `arc_agent/llm.py`

(空)

### `arc_agent/eval.py`

(空)

### `arc_agent/utils.py`

(空)

---

## 命名与设计约定

- **纯函数优先**:除非必须维护状态,否则不要写类。状态用显式参数传。
- **类型注解必须**:输入输出都标注;`np.ndarray` 标 shape 在 docstring。
- **错误处理**:库函数遇到非法输入直接抛 `ValueError` / `TypeError`,不静默兜底。兜底由 caller 决定。
- **不在库里读 env / 文件**:I/O 集中在 `eval.py` 和入口脚本,库函数保持可测。
- **避免"全能"函数**:一个函数一件事;参数超过 5 个就考虑拆。
- **废弃流程**:`@deprecated` 装饰器(自己写一个简版)+ 本文件改 Status,**至少保留两周**再删除。
