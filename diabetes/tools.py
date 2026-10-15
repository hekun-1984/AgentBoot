"""糖尿病助手的原子能力。规则判断不交给模型。"""

from __future__ import annotations

import json

from agentboot import tool
from diabetes.store import STORE

GUIDELINES = [
    {
        "id": "ada-hba1c",
        "title": "HbA1c 控制目标",
        "keywords": ["hba1c", "糖化", "目标"],
        "excerpt": "多数非妊娠成人的 HbA1c 目标通常低于 7.0%，需要结合年龄、并发症和低血糖风险个体化，不能自行改目标。",
    },
    {
        "id": "metformin-safety",
        "title": "二甲双胍用药安全",
        "keywords": ["二甲双胍", "副作用", "metformin"],
        "excerpt": "二甲双胍常见胃肠道反应。严重肾功能下降、缺氧或酗酒时乳酸酸中毒风险增加，是否停药要由医生判断。",
    },
    {
        "id": "nutrition",
        "title": "医学营养治疗",
        "keywords": ["饮食", "碳水", "营养", "空腹"],
        "excerpt": "饮食以总能量稳定和碳水化合物分配为基础，减少含糖饮料和精制碳水，不建议靠极端节食降糖。",
    },
    {
        "id": "exercise",
        "title": "运动与低血糖",
        "keywords": ["运动", "散步", "低血糖"],
        "excerpt": "规律有氧运动有助于血糖控制。血糖过低或已有危急症状时不要开始运动。",
    },
    {
        "id": "hypoglycemia",
        "title": "低血糖急症处理",
        "keywords": ["低血糖", "手抖", "头晕", "昏迷", "2.9"],
        "excerpt": "血糖低于 3.9 mmol/L 按低血糖处理；低于 3.0 mmol/L，或伴随意识改变、抽搐，属于急症，应立即升糖并求助。",
    },
    {
        "id": "followup",
        "title": "复查与随访",
        "keywords": ["复查", "随访", "用药"],
        "excerpt": "血糖高于个体化目标时缩短复查间隔。任何药物加减都必须由接诊医生决定。",
    },
]


@tool(name="get_patient_profile", description="按患者 ID 查询基本信息、病史和过敏史")
def get_patient_profile(patient_id: str) -> dict:
    profile = STORE.patients.get(patient_id)
    if profile is None:
        return {"found": False, "patient_id": patient_id}
    return {"found": True, **profile}


@tool(name="get_glucose_records", description="按患者 ID 和时间范围查询血糖记录")
def get_glucose_records(patient_id: str, start_date: str = "", end_date: str = "") -> dict:
    records = []
    for item in STORE.glucose.get(patient_id, []):
        if start_date and item["date"] < start_date:
            continue
        if end_date and item["date"] > end_date:
            continue
        records.append(dict(item))
    return {"patient_id": patient_id, "records": records}


@tool(name="get_medication_plan", description="按患者 ID 查询当前用药清单")
def get_medication_plan(patient_id: str) -> dict:
    return {"patient_id": patient_id, "medications": [dict(item) for item in STORE.medications.get(patient_id, [])]}


@tool(name="query_guideline_rag", description="检索糖尿病指南片段，并返回来源")
def query_guideline_rag(question: str, top_k: int = 3) -> dict:
    text = (question or "").lower()
    scored = []
    for chunk in GUIDELINES:
        score = sum(1 for word in chunk["keywords"] if word.lower() in text)
        if score:
            scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    chunks = [
        {"id": chunk["id"], "title": chunk["title"], "excerpt": chunk["excerpt"]}
        for _, chunk in scored[: max(top_k, 1)]
    ]
    return {"question": question, "chunks": chunks}


@tool(name="check_critical_value", description="按固定阈值判断血糖是否危急，不依赖模型自由发挥")
def check_critical_value(glucose_value: float, symptoms: str = "", fasting: bool = False) -> dict:
    text = symptoms or ""
    if any(word in text for word in ("昏迷", "意识不清", "胸痛", "抽搐", "叫不醒")):
        return {
            "critical": True,
            "level": "critical",
            "category": "symptom",
            "message": "出现昏迷、胸痛、抽搐或意识改变等危急症状",
        }
    if glucose_value < 3.0:
        return {"critical": True, "level": "critical", "category": "hypoglycemia", "message": "血糖低于 3.0 mmol/L，属于严重低血糖"}
    if glucose_value < 3.9:
        return {"critical": False, "level": "high", "category": "hypoglycemia", "message": "血糖低于 3.9 mmol/L，属于低血糖"}
    if glucose_value >= 22.2:
        return {
            "critical": True,
            "level": "critical",
            "category": "hyperglycemia",
            "message": "血糖达到 22.2 mmol/L 及以上，需要排除高渗状态或酮症",
        }
    if glucose_value >= 16.7:
        return {"critical": False, "level": "high", "category": "hyperglycemia", "message": "血糖达到 16.7 mmol/L 及以上"}
    if fasting and glucose_value >= 7.0:
        return {
            "critical": False,
            "level": "medium",
            "category": "elevated_fasting",
            "message": "空腹血糖高于 7.0 mmol/L 的常用控制范围",
        }
    if glucose_value >= 11.1:
        return {"critical": False, "level": "medium", "category": "hyperglycemia", "message": "随机血糖偏高"}
    return {"critical": False, "level": "low", "category": "in_range", "message": "这次读数没有落到危急值规则里"}


@tool(name="generate_diet_suggestion", description="根据已给定的指标和偏好生成饮食建议文本")
def generate_diet_suggestion(
    risk_level: str = "low",
    preference: str = "",
    glucose_text: str = "",
    fasting: bool = False,
) -> dict:
    if risk_level in {"high", "critical"}:
        return {"text": "当前存在高危情况，先处理急症，不提供饮食调整。"}
    preference_text = f"结合{preference}的饮食偏好，" if preference else ""
    prefix = ""
    if fasting and glucose_text:
        prefix = f"空腹血糖 {glucose_text} mmol/L 高于常用目标。"
    text = (
        f"{prefix}{preference_text}定时定量，减少含糖饮料和精制碳水，提高蔬菜和优质蛋白比例。"
        "不要靠短期节食压血糖，也不要自行更改药物。"
    )
    return {"text": text}


@tool(name="generate_exercise_suggestion", description="根据风险和血糖生成运动建议文本")
def generate_exercise_suggestion(risk_level: str = "low", preference: str = "", glucose_text: str = "") -> dict:
    if risk_level in {"high", "critical"}:
        return {"text": "暂停运动。低血糖或危急症状没有解除前不要开始活动。"}
    try:
        value = float(glucose_text) if glucose_text else None
    except ValueError:
        value = None
    if value is not None and value < 3.9:
        return {"text": "血糖偏低，现在不要运动。"}
    activity = preference or "散步"
    return {"text": f"餐后进行 20 到 30 分钟{activity}，避免空腹剧烈运动，运动前后按原监测计划看血糖。"}


@tool(name="send_followup_reminder", description="登记一条复查提醒")
def send_followup_reminder(patient_id: str, remind_at: str, message: str) -> dict:
    record = {"patient_id": patient_id, "remind_at": remind_at, "message": message, "sent": True, "channel": "in_app"}
    with STORE._lock:
        STORE.reminders.append(record)
    return dict(record)


@tool(name="log_risk_event", description="记录一条高危事件并返回记录号")
def log_risk_event(patient_id: str, risk_type: str, description: str) -> dict:
    with STORE._lock:
        record_id = f"risk-{len(STORE.events) + 1:04d}"
        record = {
            "record_id": record_id,
            "patient_id": patient_id,
            "risk_type": risk_type,
            "description": description,
        }
        STORE.events.append(record)
    return {"record_id": record_id}


@tool(name="check_guideline_compliance", description="校验指南回答是否在能力范围内且带来源")
def check_guideline_compliance(answer: str = "", in_scope: bool = True, sources_json: str = "[]") -> dict:
    try:
        sources = json.loads(sources_json or "[]")
    except json.JSONDecodeError:
        sources = []
    if not isinstance(sources, list):
        sources = []
    if not in_scope:
        return {"passed": False, "answer": "不在本能力范围", "sources": []}
    if not sources:
        return {"passed": False, "answer": "回答缺少指南来源，已拦截", "sources": []}
    return {"passed": True, "answer": answer, "sources": sources}
