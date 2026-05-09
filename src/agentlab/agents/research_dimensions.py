from __future__ import annotations

DIMENSIONS = (
    "core_paradigm",
    "coordination_style",
    "state_memory",
    "best_fit",
    "trade_off",
)

QUESTION_DIMENSION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "core_paradigm": ("核心设计", "理念", "目标场景", "core", "paradigm"),
    "coordination_style": ("任务编排", "图", "对话", "角色分工", "coordination", "workflow"),
    "state_memory": ("工具调用", "状态管理", "可观测", "state", "memory", "observability"),
    "trade_off": ("优缺点", "学习成本", "扩展性", "工程落地", "trade-off", "cost", "risk"),
    "best_fit": ("技术选型", "决策", "团队", "best fit", "selection", "scenario"),
}

CONTENT_DIMENSION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "core_paradigm": (
        "core paradigm",
        "design principle",
        "理念",
        "核心",
        "目标场景",
        "架构",
    ),
    "coordination_style": (
        "coordination",
        "orchestration",
        "workflow",
        "conversation",
        "role",
        "任务编排",
        "对话",
        "角色分工",
    ),
    "state_memory": (
        "state",
        "memory",
        "observability",
        "tool",
        "checkpoint",
        "状态",
        "记忆",
        "可观测",
        "工具调用",
    ),
    "best_fit": (
        "best fit",
        "use case",
        "scenario",
        "selection",
        "适用",
        "场景",
        "选型",
        "团队",
        "决策",
    ),
    "trade_off": (
        "trade-off",
        "pros",
        "cons",
        "cost",
        "risk",
        "limitations",
        "优缺点",
        "成本",
        "风险",
        "限制",
        "工程落地",
    ),
}
