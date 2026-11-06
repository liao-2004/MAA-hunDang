import json
import re

from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context
from maa.pipeline import JOCR, JRecognitionType

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_param(raw) -> dict:
    """pipeline 里的 custom_recognition_param 是 JSON 字符串，也兼容已经是 dict 的写法。"""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _parse_number(text):
    """从 OCR 文本里抠出第一个数字，千分位逗号先吃掉。"""
    if not text:
        return None
    cleaned = str(text).replace(",", "").replace("，", "").replace(" ", "")
    match = _NUMBER_RE.search(cleaned)
    if not match:
        return None
    number = float(match.group())
    return int(number) if number.is_integer() else number


@AgentServer.custom_recognition("find_smallest_number")
class SmallestNumberRecognition(CustomRecognition):
    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:

        param = json.loads(argv.custom_recognition_param or "{}") or {}
        rois = param.get("candidate_rois")
        if not isinstance(rois, list) or not rois:
            return CustomRecognition.AnalyzeResult(
                box=None,
                detail={"error": "candidate_rois 不能为空"},
            )

        res = []

        for roi in rois:
            # 找到 pipeline 中名为 "RecognizeNumber" 的节点，并执行它的识别逻辑
            reco_detail = context.run_recognition(
                "RecognizeNumber",
                argv.image,  # 使用当前获取到的图片
                pipeline_override={  # 临时覆盖节点参数
                    "RecognizeNumber": {  # 要覆盖的节点的名称
                        "roi": roi,  # 覆盖识别区域
                        "only_rec": True,  # 不进行文本检测，直接进行识别
                    }
                },
            )

            if reco_detail is None or not reco_detail.hit:
                print(f"无法读取到内容 {roi}")
                res.append(None)
                continue

            text = str(reco_detail.best_result.text)
            try:
                number = float(text)
            except ValueError:
                print(f"{text} 不是数字!")
                res.append(None)
                continue

            res.append(number)

        if not res or None in res:
            return CustomRecognition.AnalyzeResult(
                box=None,
                detail={"error": "存在无法读取到内容"},
            )

        min_index = res.index(min(res))
        return CustomRecognition.AnalyzeResult(
            box=rois[min_index],
            detail={"smallest_number": res[min_index]},
        )


@AgentServer.custom_recognition("value_less_than")
class ValueLessThanRecognition(CustomRecognition):
    """OCR 指定区域，抠出数字，小于阈值才算命中。

    pipeline 写法：
        "recognition": {
            "type": "Custom",
            "param": {
                "custom_recognition": "value_less_than",
                "custom_recognition_param": {
                    "roi": [334, 354, 65, 36],
                    "threshold": 100
                }
            }
        }

    参数：
        roi        [x, y, w, h]，省略时退化成节点自己的 roi
        threshold  阈值，默认 100
        only_rec   纯数字区域传 true，跳过文本检测直接识别，默认 true
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:

        param = _parse_param(argv.custom_recognition_param)

        roi = param.get("roi")
        if not (isinstance(roi, (list, tuple)) and len(roi) == 4):
            roi = (argv.roi.x, argv.roi.y, argv.roi.w, argv.roi.h)
        roi = [int(v) for v in roi]

        try:
            threshold = float(param.get("threshold", 100))
        except (TypeError, ValueError):
            threshold = 100.0

        only_rec = bool(param.get("only_rec", True))

        # 直接对当前截图做 OCR，不依赖 pipeline 里另外的节点
        reco = context.run_recognition_direct(
            JRecognitionType.OCR,
            JOCR(roi=tuple(roi), only_rec=only_rec),
            argv.image,
        )

        if reco is None or not reco.hit or reco.best_result is None:
            print(f"[value_less_than] {argv.node_name}: roi {roi} 没 OCR 到内容")
            return CustomRecognition.AnalyzeResult(
                box=None,
                detail={"hit": False, "reason": "ocr miss", "roi": roi, "threshold": threshold},
            )

        text = str(getattr(reco.best_result, "text", "") or "")
        value = _parse_number(text)

        detail = {
            "hit": False,
            "text": text,
            "value": value,
            "threshold": threshold,
            "roi": roi,
        }

        if value is None:
            print(f"[value_less_than] {argv.node_name}: 文本「{text}」里没有数字")
            return CustomRecognition.AnalyzeResult(box=None, detail=detail)

        if value >= threshold:
            print(f"[value_less_than] {argv.node_name}: {value} 不小于 {threshold:g}，不命中")
            return CustomRecognition.AnalyzeResult(box=None, detail=detail)

        print(f"[value_less_than] {argv.node_name}: {value} 小于 {threshold:g}，命中")
        detail["hit"] = True
        return CustomRecognition.AnalyzeResult(
            box=tuple(reco.box) if reco.box else tuple(roi),
            detail=detail,
        )
