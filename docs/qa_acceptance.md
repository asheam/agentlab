# QA 验收记录（v0.2 候选）

验收日期：2026-05-09  
目标：验证 `real search + critic auto/rule` 在不同主题下的稳定性与可解释性。

## 1. 验收命令

规则评审链路（已执行）：

```bash
uv run python examples/04_deep_research.py "研究 LangGraph、AutoGen、CrewAI 的区别" --search-mode real --critic-mode rule --output-dir outputs/qa/topic1
uv run python examples/04_deep_research.py "比较 LangGraph 与 CrewAI 在企业流程自动化中的适用性" --search-mode real --critic-mode rule --output-dir outputs/qa/topic2
uv run python examples/04_deep_research.py "多 Agent 框架在工具调用与状态管理上的关键差异" --search-mode real --critic-mode rule --output-dir outputs/qa/topic3
```

LLM 评审链路（已执行）：

```bash
uv run python examples/04_deep_research.py "研究 LangGraph、AutoGen、CrewAI 的区别" --search-mode real --critic-mode auto --use-openai --output-dir outputs/qa_llm/topic1
uv run python examples/04_deep_research.py "比较 LangGraph 与 CrewAI 在企业流程自动化中的适用性" --search-mode real --critic-mode auto --use-openai --output-dir outputs/qa_llm/topic2
uv run python examples/04_deep_research.py "多 Agent 框架在工具调用与状态管理上的关键差异" --search-mode real --critic-mode auto --use-openai --output-dir outputs/qa_llm/topic3
```

## 2. 规则评审结果（`outputs/qa`）

| 主题 | real/mock/fallback | assessment_mode | verdict | overall | dimension_coverage |
| --- | --- | --- | --- | ---: | ---: |
| topic1 | 5/0/0 | rule | acceptable | 78.3 | 100.0 |
| topic2 | 5/0/0 | rule | need_improvement | 55.3 | 100.0 |
| topic3 | 4/0/0 | rule | need_improvement | 21.4 | 20.0 |

判定：
- 基线样例：`outputs/qa/topic1`
- 失败样例：`outputs/qa/topic3`（维度覆盖低，`dimensions_missing` 明确）

## 3. LLM 评审结果（`outputs/qa_llm`）

| 主题 | real/mock/fallback | assessment_mode | verdict | overall |
| --- | --- | --- | --- | ---: |
| topic1 | 4/0/0 | llm | need_improvement | 55.0 |
| topic2 | 5/0/0 | llm | need_improvement | 35.0 |
| topic3 | 5/0/0 | llm | need_improvement | 55.0 |

判定：
- `--use-openai --critic-mode auto` 已走到 `assessment_mode=llm`
- 三个主题均生成完整 `report.md / trace.json / workspace.json`

## 4. 关键输出路径

规则评审基线：
- `outputs/qa/topic1/report.md`
- `outputs/qa/topic1/trace.json`
- `outputs/qa/topic1/workspace.json`

规则评审失败样例：
- `outputs/qa/topic3/report.md`
- `outputs/qa/topic3/trace.json`
- `outputs/qa/topic3/workspace.json`

LLM 评审样例：
- `outputs/qa_llm/topic1/report.md`
- `outputs/qa_llm/topic2/report.md`
- `outputs/qa_llm/topic3/report.md`

## 5. 验收结论

- 真实检索链路稳定：三组主题均成功完成执行，且无 mock fallback。
- 维度化输出有效：规则评审可区分“覆盖完整”与“覆盖不足”主题。
- LLM 评审链路有效：`auto` 模式可触发 LLM Critic 并产出结果。
