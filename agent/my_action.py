import json

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction


def _parse_param(raw) -> dict:
    """pipeline 里的 custom_action_param 是 JSON 字符串，也兼容已经是 dict 的写法。"""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _walk(obj):
    if isinstance(obj, dict):
        yield obj
        for item in obj.values():
            yield from _walk(item)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk(item)


def _reco_values(reco_detail) -> dict:
    """从前序识别详情里捞 value / text / threshold，用来填消息占位符。"""
    if reco_detail is None:
        return {}

    raw = getattr(reco_detail, "raw_detail", None) or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = {}
    if isinstance(raw, dict) and isinstance(raw.get("detail"), dict):
        raw = raw["detail"]

    found = {}
    for node in _walk(raw):
        for key in ("value", "text", "threshold"):
            if key in node and key not in found:
                if isinstance(node[key], (str, int, float, bool)):
                    found[key] = node[key]
    return found


def _format_message(template: str, values: dict, node_name: str, x: int, y: int) -> str:
    mapping = {
        "node": node_name,
        "x": x,
        "y": y,
        "value": values.get("value"),
        "text": values.get("text"),
        "threshold": values.get("threshold"),
    }
    for key, value in mapping.items():
        if value is None:
            text = "?"
        elif isinstance(value, float) and value.is_integer():
            text = str(int(value))
        else:
            text = str(value)
        template = template.replace("{" + key + "}", text)
    return template


@AgentServer.custom_action("click_target")
class ClickTargetAction(CustomAction):
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:

        print("ClickTargetAction is running!")

        param = json.loads(argv.custom_action_param or "{}") or {}
        target = param.get("target")
        if not isinstance(target, list) or len(target) != 2:
            return False

        # 执行点击
        click_job = context.tasker.controller.post_click(target[0], target[1])
        # 等待点击完成，这一步必须有！
        click_job.wait()

        return True


@AgentServer.custom_action("click_and_notify")
class ClickAndNotifyAction(CustomAction):
    """按绝对坐标点一下，同时输出一条消息（走 MAA 日志）。

    pipeline 写法：
        "action": {
            "type": "Custom",
            "param": {
                "custom_action": "click_and_notify",
                "custom_action_param": {
                    "target": [210, 211],
                    "message": "任务一：数值 {value} 小于 {threshold}，已点击刷新"
                }
            }
        }

    参数：
        target   [x, y] 绝对坐标；也接受 [x, y, w, h]，点区域中心
        message  消息模板，支持 {value} {text} {threshold} {node} {x} {y}
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:

        param = _parse_param(argv.custom_action_param)

        target = param.get("target")
        if not (isinstance(target, (list, tuple)) and len(target) in (2, 4)):
            print(
                f"[click_and_notify] {argv.node_name}: target 要写成 [x, y] 或 [x, y, w, h]，"
                f"现在拿到的是 {target!r}"
            )
            return False

        if len(target) == 2:
            x, y = int(target[0]), int(target[1])
        else:
            # 给的是区域，点中心
            x = int(target[0]) + int(target[2]) // 2
            y = int(target[1]) + int(target[3]) // 2

        values = _reco_values(argv.reco_detail)
        template = str(param.get("message") or "{node}: 条件命中，点击 ({x}, {y})")
        message = _format_message(template, values, argv.node_name, x, y)

        print(f"[click_and_notify] {message}")

        click_job = context.tasker.controller.post_click(x, y)
        click_job.wait()

        return True
