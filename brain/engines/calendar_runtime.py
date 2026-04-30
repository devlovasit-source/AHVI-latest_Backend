from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from models.calendar_models import (
    BufferPlan,
    CalendarEventInput,
    CalendarRuntimeResult,
    ChecklistBundle,
    ChecklistSection,
    ClassifiedEvent,
    OutfitPrompt,
    PredictiveOutput,
    Reminder,
)


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: str | None) -> datetime | None:
    text = _safe_text(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def _infer_group_subtype_priority(title: str) -> Tuple[str, str, float, List[str], str]:
    t = (title or "").lower()
    matched = []

    def hit(word: str) -> bool:
        ok = word in t
        if ok:
            matched.append(word)
        return ok

    group = "miscellaneous"
    subtype = "general"
    priority = "light"
    conf = 0.55

    if any(hit(w) for w in ["flight", "airport", "train", "bus", "trip", "travel", "boarding"]):
        group, subtype, priority, conf = "travel", "travel_prep", "important", 0.78
    elif any(hit(w) for w in ["interview", "presentation", "meeting", "client", "office", "work"]):
        group, subtype, priority, conf = "work", "work_event", "important", 0.72
        if "interview" in t:
            subtype, conf = "interview", 0.8
        elif "presentation" in t:
            subtype, conf = "presentation", 0.78
    elif any(hit(w) for w in ["doctor", "appointment", "clinic", "hospital", "dentist", "lab", "checkup"]):
        group, subtype, priority, conf = "health", "doctor_appointment", "important", 0.74
        if "lab" in t or "test" in t:
            subtype, conf = "lab_test", 0.76
    elif any(hit(w) for w in ["gym", "workout", "yoga", "run", "training", "pilates"]):
        group, subtype, priority, conf = "fitness", "gym_class", "light", 0.7
    elif any(hit(w) for w in ["wedding", "party", "birthday", "dinner", "date", "event"]):
        group, subtype, priority, conf = "social", "social_event", "important", 0.7
        if "wedding" in t:
            subtype, conf = "wedding", 0.82
        elif "birthday" in t:
            subtype, conf = "birthday_party", 0.78
        elif "date" in t:
            subtype, conf = "date", 0.74
        elif "dinner" in t:
            subtype, conf = "dinner", 0.74
    elif any(hit(w) for w in ["payment", "bill", "emi", "rent", "tax", "invoice", "due"]):
        group, subtype, priority, conf = "finance", "payment", "important", 0.7

    return group, subtype, float(conf), matched, priority


def _prep_tasks(group: str, subtype: str, dress_code: str | None = None) -> List[str]:
    tasks = set()
    if group == "travel":
        tasks.update(["check documents", "pack essentials", "set alarm", "leave with buffer"])
    elif group == "social":
        tasks.update(["decide outfit", "confirm venue/time"])
        if subtype in {"wedding", "birthday_party"}:
            tasks.add("check shoes and bag")
    elif group in {"health"}:
        tasks.update(["keep reports ready", "leave with buffer"])
        if subtype == "lab_test":
            tasks.add("check fasting instructions")
    elif group == "work":
        if subtype in {"presentation", "interview"}:
            tasks.update(["review materials", "charge laptop", "set outfit aside"])
        else:
            tasks.add("review agenda")
    elif group == "finance":
        tasks.update(["keep payment method ready", "pay before due window"])
    elif group == "fitness":
        tasks.update(["prep water bottle", "pack towel"])
    else:
        tasks.add("quick prep check")

    if _safe_text(dress_code):
        tasks.add(f"dress code: {_safe_text(dress_code)}")

    return sorted(tasks)


def _packing_list(subtype: str) -> List[str]:
    mapping = {
        "travel_prep": ["id", "phone", "wallet", "charger", "tickets"],
        "doctor_appointment": ["reports", "id"],
        "lab_test": ["reports", "id"],
        "presentation": ["laptop", "charger", "deck"],
        "interview": ["resume", "id"],
        "gym_class": ["water", "towel"],
    }
    return list(mapping.get(subtype, []))


def _outfit_prompt(subtype: str) -> Optional[OutfitPrompt]:
    rules = {
        "presentation": (["structured", "clean", "confident"], ["smart footwear"], ["minimal accessory"]),
        "interview": (["sharp", "clean", "confident"], ["smart footwear"], ["minimal accessory"]),
        "wedding": (["occasionwear", "event-ready"], ["comfortable dress shoes"], ["small bag / watch"]),
        "birthday_party": (["elevated casual", "confident"], ["clean sneakers"], ["statement accent"]),
        "gym_class": (["activewear", "breathable"], ["training shoes"], ["none"]),
        "doctor_appointment": (["comfortable", "neat"], ["easy slip-ons"], ["none"]),
        "travel_prep": (["comfortable", "layerable"], ["sneakers"], ["travel pouch"]),
    }
    row = rules.get(subtype)
    if not row:
        return None
    outfit, footwear, accessories = row
    return OutfitPrompt(
        styleMode="auto",
        outfitKeywords=outfit,
        footwearKeywords=footwear,
        accessoryKeywords=accessories,
    )


def _buffer_plan(start_at: datetime | None, group: str) -> BufferPlan | None:
    if start_at is None:
        return None
    leave_minutes = 30
    if group == "travel":
        leave_minutes = 120
    leave_by = start_at - timedelta(minutes=leave_minutes)
    get_ready = start_at - timedelta(minutes=max(20, leave_minutes - 10))
    return BufferPlan(
        prepNightBefore=group in {"travel", "work"},
        startGettingReadyAtISO=get_ready.isoformat(),
        leaveByISO=leave_by.isoformat(),
        bufferReason="auto buffer based on event category",
    )


def _stress_score(group: str, priority: str, subtype: str) -> int:
    score = 20
    if priority == "critical":
        score += 25
    if group == "travel":
        score += 20
    if subtype in {"wedding", "presentation", "interview"}:
        score += 15
    return int(min(score, 100))


def _followups(subtype: str) -> List[str]:
    out = []
    if "travel" in subtype or "flight" in subtype:
        out.append("check hotel or stay details")
    if subtype == "interview":
        out.append("send follow-up email")
    return out


def _default_reminders(start_at: datetime | None, priority: str, tone_profile: str = "gentle") -> List[Reminder]:
    if start_at is None:
        return []

    if priority == "critical":
        offsets = [180, 60, 15]
    elif priority == "important":
        offsets = [120, 30]
    else:
        offsets = [60, 15]

    reminders = []
    for idx, minutes in enumerate(offsets, start=1):
        send_at = start_at - timedelta(minutes=minutes)
        reminders.append(
            Reminder(
                id=f"rem_{idx}",
                offsetMinutes=int(minutes),
                message=f"Reminder: {minutes} min until your event.",
                priority="important" if priority in {"important", "critical"} else "light",
                toneProfile=tone_profile,
                sendAtISO=send_at.isoformat(),
            )
        )
    return reminders


def _checklists(prep_tasks: List[str], packing_list: List[str]) -> ChecklistBundle:
    carry = ChecklistSection(title="Carry", items=packing_list[:12]) if packing_list else None
    prep = ChecklistSection(title="Prep Tonight", items=prep_tasks[:12]) if prep_tasks else None
    return ChecklistBundle(carry=carry, prepTonight=prep)


def run_calendar_runtime(event: CalendarEventInput, *, user_id: str | None = None) -> CalendarRuntimeResult:
    title = _safe_text(event.title)
    group, subtype, confidence, matched, priority = _infer_group_subtype_priority(title)

    start_at = _parse_iso(event.startAtISO)

    classified = ClassifiedEvent(
        **event.model_dump(),
        group=group,
        subtype=subtype,
        confidenceScore=float(confidence),
        matchedSignals=matched,
        missingFields=[],
        needsUserConfirmation=False,
        priority=priority,
    )

    prep_tasks = _prep_tasks(group, subtype, dress_code=event.dressCode)
    packing = _packing_list(subtype)
    outfit_prompt = _outfit_prompt(subtype)
    buffer = _buffer_plan(start_at, group)
    stress = _stress_score(group, priority, subtype)
    followups = _followups(subtype)

    predictive = PredictiveOutput(
        prepTasks=prep_tasks,
        packingList=packing,
        linkedErrands=[],
        stressLoadScore=stress,
        outfitPrompt=outfit_prompt,
        bufferPlan=buffer,
        followupCandidates=followups,
    )

    reminders = _default_reminders(start_at, priority, tone_profile="gentle")
    checklists = _checklists(prep_tasks, packing)

    return CalendarRuntimeResult(
        classifiedEvent=classified,
        predictiveOutput=predictive,
        checklistBundle=checklists,
        reminders=reminders,
        dayBriefingHint=[f"{group.title()} event: {title}"],
    )

