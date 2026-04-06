from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from fpdf import FPDF


STYLE = {
    "company_name": "GlobeTrek AI",
    "font": "Helvetica",
    "title_size": 18,
    "section_heading_size": 13,
    "sub_heading_size": 11,
    "body_size": 10.5,
    "small_size": 9,
    "line_height": 5.6,
    "margin_left": 16,
    "margin_right": 16,
    "margin_top": 18,
    "margin_bottom": 18,
    "section_gap": 4.5,
    "sub_gap": 2.4,
    "footer_height": 10,
}


DEFAULT_PACKING = {
    "Documents": [
        "Passport / ID",
        "Bookings and confirmations",
        "Travel insurance details",
    ],
    "Clothing": [
        "Weather-appropriate outfits",
        "Comfortable walking shoes",
        "Evening wear if needed",
    ],
    "Essentials": [
        "Phone charger and adapter",
        "Personal medication",
        "Reusable water bottle",
    ],
}


@dataclass
class TimeSlotBlock:
    name: str
    place: str = ""
    activity: str = ""
    duration: str = ""
    transport: str = ""
    food_tip: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class DayPlan:
    day_number: int
    title: str
    date_label: str = ""
    slots: list[TimeSlotBlock] = field(default_factory=list)
    raw_lines: list[str] = field(default_factory=list)
    daily_budget: float = 0.0
    budget_note: str = ""


@dataclass
class ParsedItinerary:
    trip_summary: dict[str, str]
    overview: str
    days: list[DayPlan]
    important_notes: dict[str, list[str]]
    packing_checklist: dict[str, list[str]]
    total_budget: dict[str, float]


class TravelItineraryPDF(FPDF):
    def header(self):
        if self.page_no() <= 1:
            return
        self.set_xy(STYLE["margin_left"], STYLE["margin_top"] - 7)
        self.set_font(STYLE["font"], "", STYLE["small_size"])
        self.set_text_color(105, 105, 105)
        self.cell(0, 4, _safe_text(STYLE["company_name"] + " - Travel Itinerary"), align="L")
        self.set_draw_color(188, 188, 188)
        y = STYLE["margin_top"] - 1.5
        self.line(STYLE["margin_left"], y, self.w - STYLE["margin_right"], y)

    def footer(self):
        self.set_y(-STYLE["footer_height"])
        self.set_font(STYLE["font"], "", STYLE["small_size"])
        self.set_text_color(120, 120, 120)
        self.cell(
            0,
            4,
            _safe_text(f"{STYLE['company_name']} | Page {self.page_no()}/{{nb}}"),
            align="C",
        )


def _safe_text(value: Any) -> str:
    return str(value).encode("latin-1", "replace").decode("latin-1")


def _clean_markdown_line(line: str) -> str:
    text = str(line or "").strip()
    if not text:
        return ""

    text = text.replace("```", "")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = text.replace("__", "")
    text = text.replace("`", "")
    text = re.sub(r"^\s{0,3}>+\s*", "", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text)
    text = re.sub(r"^\s*(\d+)[\)\.]\s+", "", text)
    text = re.sub(r"^\s*AI\s*Plan\s*-\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text).strip()

    if text in {"-", "--", "---", "----"}:
        return ""
    if re.search(r"\b(not specified|n/?a|null|none|unknown)\b", text, flags=re.IGNORECASE):
        return ""

    return text


def _parse_start_date(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    text = text.replace("/", "-")
    for fmt in ["%Y-%m-%d", "%d-%m-%Y", "%Y-%m-%d %H:%M:%S"]:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _extract_day_heading(line: str) -> tuple[int, str] | None:
    match = re.match(r"^day\s*(\d+)\s*[:\-]?\s*(.*)$", line, flags=re.IGNORECASE)
    if not match:
        return None
    number = int(match.group(1))
    title = match.group(2).strip()
    return number, title


def _extract_timeslot_heading(line: str) -> tuple[str, str, str] | None:
    match = re.match(
        r"^(Morning|Afternoon|Evening|Night)\s*(\((.*?)\))?\s*[:\-]?\s*(.*)$",
        line,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    name = match.group(1).title()
    time_hint = (match.group(3) or "").strip()
    remainder = (match.group(4) or "").strip()
    return name, time_hint, remainder


def _extract_named_field(line: str) -> tuple[str, str] | None:
    match = re.match(
        r"^(Place|Location|Activity|Activity Type|Duration|Estimated Time|Transport|Local Transport|Food Tip|Meal Tip|Meal|Budget|Cost|Spend|Note|Notes|Tip)\s*[:\-]\s*(.+)$",
        line,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    raw_key = match.group(1).strip().lower()
    value = match.group(2).strip()

    key_map = {
        "place": "place",
        "location": "place",
        "activity": "activity",
        "activity type": "activity",
        "duration": "duration",
        "estimated time": "duration",
        "transport": "transport",
        "local transport": "transport",
        "food tip": "food_tip",
        "meal tip": "food_tip",
        "meal": "food_tip",
        "budget": "budget",
        "cost": "budget",
        "spend": "budget",
        "note": "note",
        "notes": "note",
        "tip": "note",
    }

    return key_map.get(raw_key, "note"), value


def _default_daily_budget(budget_level: str) -> float:
    level = str(budget_level or "").strip().lower()
    if level == "low":
        return 90.0
    if level == "mid-range":
        return 170.0
    if level == "luxury":
        return 340.0
    return 160.0


def _extract_amount_from_text(text: str) -> float | None:
    match = re.search(r"(?:\$|usd\s*)?(\d+(?:\.\d+)?)", str(text or ""), flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _is_missing_value(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {
        "",
        "not specified",
        "n/a",
        "na",
        "null",
        "none",
        "unknown",
        "unspecified",
        "-",
        "--",
        "---",
    }


def _clean_value(value: Any) -> str:
    text = str(value or "").strip()
    if _is_missing_value(text):
        return ""
    return text


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        cleaned = _clean_value(value)
        if not cleaned:
            continue
        key = re.sub(r"\s+", " ", cleaned).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


def _ensure_day_slot(day: DayPlan) -> TimeSlotBlock:
    if day.slots:
        return day.slots[-1]
    slot = TimeSlotBlock(name="Morning")
    day.slots.append(slot)
    return slot


def _clean_slot(slot: TimeSlotBlock):
    slot_name = _clean_value(slot.name).title()
    if not slot_name or slot_name.lower() == "general":
        slot_name = "Morning"
    slot.name = slot_name

    slot.place = _clean_value(slot.place)
    slot.activity = _clean_value(slot.activity)
    slot.duration = _clean_value(slot.duration)
    slot.transport = _clean_value(slot.transport)
    slot.food_tip = _clean_value(slot.food_tip)

    filtered_notes = []
    for note in slot.notes:
        cleaned = _clean_value(note)
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if any(k in lowered for k in ["not specified", "n/a", "null", "none", "unknown"]):
            continue
        if any(k in lowered for k in ["budget:", "cost:", "spend:", "safety", "safe", "emergency"]):
            continue
        filtered_notes.append(cleaned)
    slot.notes = _dedupe_preserve_order(filtered_notes)[:2]


def _slot_has_content(slot: TimeSlotBlock) -> bool:
    return any(
        [
            _clean_value(slot.place),
            _clean_value(slot.activity),
            _clean_value(slot.duration),
            _clean_value(slot.transport),
            _clean_value(slot.food_tip),
            bool(slot.notes),
        ]
    )


def _normalize_overview(overview_lines: list[str], trip_details: dict[str, Any]) -> str:
    compact = [_clean_value(line) for line in overview_lines if _clean_value(line)]
    if compact:
        joined = " ".join(compact)
        return joined[:950]

    destination = _clean_value(trip_details.get("country", "")) or "the destination"
    duration = trip_details.get("days", "multi")
    traveler = _clean_value(trip_details.get("age_group", "")) or "traveler"
    interests = _clean_value(trip_details.get("interests", "")) or "culture, cuisine, and signature local experiences"
    return (
        f"This itinerary is designed for a {traveler} journey to {destination} over {duration} day(s), "
        f"with focus on {interests}. The schedule balances logistics, activities, meals, and rest "
        "while keeping daily pacing practical and easy to follow."
    )


def _categorize_notes(raw_notes: list[str]) -> dict[str, list[str]]:
    safety = []
    local = []
    weather = []

    for line in _dedupe_preserve_order(raw_notes):
        if _is_missing_value(line):
            continue
        lowered = line.lower()
        if any(k in lowered for k in ["safe", "safety", "scam", "emergency", "secure"]):
            safety.append(line)
        elif any(k in lowered for k in ["weather", "rain", "heat", "storm", "alternative", "backup"]):
            weather.append(line)
        else:
            local.append(line)

    if not safety:
        safety = [
            "Keep digital and physical copies of key documents.",
            "Use trusted transport at night and avoid isolated routes.",
        ]
    if not local:
        local = [
            "Carry small cash for local vendors and tips.",
            "Respect local customs and dress guidelines at cultural sites.",
        ]
    if not weather:
        weather = [
            "If weather changes, switch outdoor visits with indoor museum or culinary experiences.",
            "Keep one flexible slot daily for weather-driven adjustments.",
        ]

    return {
        "Safety": _dedupe_preserve_order(safety)[:5],
        "Local Tips": _dedupe_preserve_order(local)[:6],
        "Weather Alternatives": _dedupe_preserve_order(weather)[:5],
    }


def _categorize_packing(lines: list[str]) -> dict[str, list[str]]:
    if not lines:
        return DEFAULT_PACKING

    categorized = {
        "Documents": [],
        "Clothing": [],
        "Essentials": [],
    }

    for line in lines:
        lowered = line.lower()
        if any(k in lowered for k in ["passport", "visa", "ticket", "booking", "insurance", "id"]):
            categorized["Documents"].append(line)
        elif any(k in lowered for k in ["shirt", "jacket", "coat", "shoe", "clothes", "swim", "outfit"]):
            categorized["Clothing"].append(line)
        else:
            categorized["Essentials"].append(line)

    for category, defaults in DEFAULT_PACKING.items():
        if not categorized[category]:
            categorized[category] = defaults

    return {k: v[:8] for k, v in categorized.items()}


def _calculate_total_budget(days: list[DayPlan]) -> dict[str, float]:
    subtotal = round(sum(day.daily_budget for day in days), 2)
    service_fee = round(subtotal * 0.08, 2)
    contingency = round(subtotal * 0.10, 2)
    grand_total = round(subtotal + service_fee + contingency, 2)

    return {
        "daily_subtotal": subtotal,
        "service_fee": service_fee,
        "contingency": contingency,
        "grand_total": grand_total,
    }


def _parse_itinerary(ai_itinerary_text: str, trip_details: dict[str, Any]) -> ParsedItinerary:
    cleaned_lines = []
    for raw in str(ai_itinerary_text or "").splitlines():
        cleaned = _clean_markdown_line(raw)
        if cleaned:
            cleaned_lines.append(cleaned)

    days: list[DayPlan] = []
    overview_lines: list[str] = []
    global_notes: list[str] = []
    packing_lines: list[str] = []

    current_day: DayPlan | None = None
    current_slot: TimeSlotBlock | None = None
    in_packing = False
    in_notes = False

    for line in cleaned_lines:
        lowered = line.lower()

        if lowered in {
            "trip summary",
            "overview",
            "day-wise itinerary",
            "daily itinerary",
            "budget summary",
            "daily budget summary",
        }:
            continue

        day_heading = _extract_day_heading(line)
        if day_heading:
            day_number, heading_text = day_heading
            current_day = DayPlan(day_number=day_number, title=heading_text or f"Day {day_number}")
            days.append(current_day)
            current_slot = None
            in_packing = False
            in_notes = False
            continue

        if lowered.startswith("packing checklist") or lowered.startswith("packing list"):
            in_packing = True
            in_notes = False
            continue

        if lowered.startswith("important notes") or lowered == "notes":
            in_notes = True
            in_packing = False
            continue

        if in_packing:
            packing_lines.append(line)
            continue

        if in_notes:
            global_notes.append(line)
            continue

        if current_day is None:
            if re.match(
                r"^(destination|start date|duration|budget|traveler type)\s*[:\-]\s*",
                line,
                flags=re.IGNORECASE,
            ):
                continue
            overview_lines.append(line)
            continue

        date_match = re.match(r"^date\s*[:\-]\s*(.+)$", line, flags=re.IGNORECASE)
        if date_match:
            date_value = _clean_value(date_match.group(1))
            if date_value:
                current_day.date_label = "Date: " + date_value
            continue

        current_day.raw_lines.append(line)

        slot_heading = _extract_timeslot_heading(line)
        if slot_heading:
            slot_name, time_hint, remainder = slot_heading
            current_slot = TimeSlotBlock(name=slot_name)
            if time_hint:
                current_slot.duration = time_hint
            if remainder:
                named_field = _extract_named_field(remainder)
                if named_field:
                    key, value = named_field
                    if key == "place":
                        current_slot.place = value
                    elif key == "activity":
                        current_slot.activity = value
                    elif key == "duration":
                        current_slot.duration = value
                    elif key == "transport":
                        current_slot.transport = value
                    elif key == "food_tip":
                        current_slot.food_tip = value
                    else:
                        current_slot.notes.append(value)
                else:
                    current_slot.activity = remainder
            current_day.slots.append(current_slot)
            continue

        named_field = _extract_named_field(line)
        if named_field:
            key, value = named_field
            slot = _ensure_day_slot(current_day)
            current_slot = slot

            if key == "place":
                slot.place = value
            elif key == "activity":
                slot.activity = value
            elif key == "duration":
                slot.duration = value
            elif key == "transport":
                slot.transport = value
            elif key == "food_tip":
                slot.food_tip = value
            elif key == "budget":
                current_day.raw_lines.append(f"Budget: {value}")
            else:
                slot.notes.append(value)
            continue

        slot = _ensure_day_slot(current_day)
        current_slot = slot
        if not slot.activity:
            slot.activity = line
        else:
            slot.notes.append(line)

    if not days:
        day_count = int(trip_details.get("days", 1) or 1)
        if day_count < 1:
            day_count = 1
        fallback_day = DayPlan(day_number=1, title="Day 1")
        if cleaned_lines:
            fallback_day.raw_lines = cleaned_lines[:]
            first_line = cleaned_lines[0]
            fallback_slot = TimeSlotBlock(name="Morning", activity=first_line)
            for extra in cleaned_lines[1:]:
                fallback_slot.notes.append(extra)
            fallback_day.slots.append(fallback_slot)
        else:
            fallback_day.slots.append(
                TimeSlotBlock(name="Morning", activity="Arrival and orientation in the destination area.")
            )
        days = [fallback_day]

    budget_level = str(trip_details.get("budget", "")).strip()
    default_daily_budget = _default_daily_budget(budget_level)

    for day in days:
        if not day.slots:
            if day.raw_lines:
                day.slots.append(TimeSlotBlock(name="Morning", activity=day.raw_lines[0]))
            else:
                day.slots.append(
                    TimeSlotBlock(name="Morning", activity="Arrival, check-in, and destination orientation.")
                )

        slot_order = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
        cleaned_slots: list[TimeSlotBlock] = []
        seen_slot_signatures = set()
        for slot in day.slots:
            _clean_slot(slot)
            if not _slot_has_content(slot):
                continue
            signature = (
                slot.name.lower(),
                slot.place.lower(),
                slot.activity.lower(),
                slot.duration.lower(),
                slot.transport.lower(),
                slot.food_tip.lower(),
            )
            if signature in seen_slot_signatures:
                continue
            seen_slot_signatures.add(signature)
            cleaned_slots.append(slot)

        cleaned_slots.sort(key=lambda s: slot_order.get(s.name, 9))
        day.slots = cleaned_slots[:4]
        if not day.slots:
            day.slots = [
                TimeSlotBlock(name="Morning", activity="Arrival, local orientation, and curated neighborhood walk.")
            ]

        day_budget_amount = None
        day_budget_note = ""
        for raw_line in day.raw_lines:
            if any(k in raw_line.lower() for k in ["budget", "cost", "spend", "estimate"]):
                possible = _extract_amount_from_text(raw_line)
                if possible is not None:
                    day_budget_amount = possible
                    day_budget_note = raw_line
                    break

        if day_budget_amount is None:
            day.daily_budget = default_daily_budget
            day.budget_note = f"Estimated from budget level ({budget_level or 'standard'})"
        else:
            day.daily_budget = round(day_budget_amount, 2)
            day.budget_note = _clean_value(day_budget_note) or "Estimated daily spend"

    start_dt = _parse_start_date(trip_details.get("start_date"))
    for idx, day in enumerate(days):
        if _clean_value(day.date_label):
            continue
        if start_dt is None:
            day.date_label = ""
        else:
            current_date = start_dt + timedelta(days=idx)
            day.date_label = "Date: " + current_date.strftime("%d %b %Y (%a)")

    trip_summary = {}
    destination_value = _clean_value(trip_details.get("country", ""))
    start_date_value = _clean_value(trip_details.get("start_date", ""))
    duration_value = trip_details.get("days", len(days))
    budget_value = _clean_value(trip_details.get("budget", ""))
    traveler_value = _clean_value(trip_details.get("age_group", ""))

    if destination_value:
        trip_summary["Destination"] = destination_value
    if start_date_value:
        trip_summary["Start Date"] = start_date_value
    trip_summary["Duration"] = f"{duration_value} day(s)"
    if budget_value:
        trip_summary["Budget"] = budget_value
    if traveler_value:
        trip_summary["Traveler Type"] = traveler_value

    overview = _normalize_overview(overview_lines, trip_details)

    for day in days:
        for slot in day.slots:
            for note in slot.notes:
                global_notes.append(note)

    important_notes = _categorize_notes(global_notes)
    packing_checklist = _categorize_packing(packing_lines)
    total_budget = _calculate_total_budget(days)

    return ParsedItinerary(
        trip_summary=trip_summary,
        overview=overview,
        days=days,
        important_notes=important_notes,
        packing_checklist=packing_checklist,
        total_budget=total_budget,
    )


def _ensure_space(pdf: TravelItineraryPDF, required_height: float):
    max_y = pdf.h - STYLE["margin_bottom"]
    if pdf.get_y() + required_height > max_y:
        pdf.add_page()


def _draw_separator(pdf: TravelItineraryPDF):
    y = pdf.get_y() + 1.2
    pdf.set_draw_color(170, 170, 170)
    pdf.line(STYLE["margin_left"], y, pdf.w - STYLE["margin_right"], y)
    pdf.set_y(y + 2.8)


def render_header(pdf: TravelItineraryPDF, title: str):
    pdf.set_font(STYLE["font"], "B", STYLE["title_size"])
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 9, _safe_text(title), align="C", ln=True)

    pdf.set_font(STYLE["font"], "", STYLE["small_size"])
    pdf.set_text_color(95, 95, 95)
    pdf.cell(0, 5, _safe_text("Prepared by GlobeTrek AI"), align="C", ln=True)

    pdf.ln(2)
    _draw_separator(pdf)


def render_section_heading(pdf: TravelItineraryPDF, heading: str):
    _ensure_space(pdf, 10)
    pdf.set_font(STYLE["font"], "B", STYLE["section_heading_size"])
    pdf.set_text_color(25, 25, 25)
    pdf.cell(0, 7, _safe_text(heading), ln=True)
    pdf.ln(0.6)


def _render_labeled_row(pdf: TravelItineraryPDF, label: str, value: str):
    value = _clean_value(value)
    if not value:
        return

    _ensure_space(pdf, STYLE["line_height"] + 2)
    x = STYLE["margin_left"]
    key_w = 38

    pdf.set_x(x)
    pdf.set_font(STYLE["font"], "B", STYLE["body_size"])
    pdf.set_text_color(35, 35, 35)
    pdf.cell(key_w, STYLE["line_height"], _safe_text(label + ":"), border=0)

    pdf.set_font(STYLE["font"], "", STYLE["body_size"])
    pdf.set_text_color(35, 35, 35)
    pdf.multi_cell(0, STYLE["line_height"], _safe_text(value))


def render_trip_summary(pdf: TravelItineraryPDF, summary: dict[str, str]):
    render_section_heading(pdf, "Trip Summary")
    rendered_rows = 0
    for key in ["Destination", "Start Date", "Duration", "Budget", "Traveler Type"]:
        value = _clean_value(summary.get(key, ""))
        if not value:
            continue
        _render_labeled_row(pdf, key, value)
        rendered_rows += 1

    if rendered_rows == 0:
        _render_labeled_row(pdf, "Duration", "Multi-day itinerary")

    pdf.ln(STYLE["section_gap"] - 1.2)
    _draw_separator(pdf)


def render_overview(pdf: TravelItineraryPDF, overview: str):
    render_section_heading(pdf, "Overview")
    pdf.set_font(STYLE["font"], "", STYLE["body_size"])
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, STYLE["line_height"], _safe_text(overview))
    pdf.ln(STYLE["section_gap"] - 1.2)
    _draw_separator(pdf)


def _render_slot_block(pdf: TravelItineraryPDF, slot: TimeSlotBlock):
    _clean_slot(slot)
    if not _slot_has_content(slot):
        return

    _ensure_space(pdf, 30)

    pdf.set_x(STYLE["margin_left"] + 2)
    pdf.set_font(STYLE["font"], "B", STYLE["sub_heading_size"])
    pdf.set_text_color(35, 35, 35)
    pdf.cell(0, 6.2, _safe_text(f"- {slot.name}"), ln=True)

    field_rows = [
        ("Place", slot.place),
        ("Activity", slot.activity),
        ("Duration", slot.duration),
        ("Transport", slot.transport),
        ("Food Tip", slot.food_tip),
    ]

    for label, value in field_rows:
        value = _clean_value(value)
        if not value:
            continue
        _ensure_space(pdf, 7)
        pdf.set_x(STYLE["margin_left"] + 10)
        pdf.set_font(STYLE["font"], "B", STYLE["body_size"])
        pdf.set_text_color(35, 35, 35)
        pdf.cell(24, STYLE["line_height"], _safe_text(label + ":"), border=0)

        pdf.set_font(STYLE["font"], "", STYLE["body_size"])
        pdf.set_text_color(35, 35, 35)
        pdf.multi_cell(0, STYLE["line_height"], _safe_text(value))

    if slot.notes:
        for note in _dedupe_preserve_order(slot.notes)[:2]:
            if not note:
                continue
            _ensure_space(pdf, 7)
            pdf.set_x(STYLE["margin_left"] + 10)
            pdf.set_font(STYLE["font"], "", STYLE["body_size"])
            pdf.set_text_color(60, 60, 60)
            pdf.multi_cell(0, STYLE["line_height"], _safe_text(f"Note: {note}"))

    pdf.ln(1.0)


def render_day_section(pdf: TravelItineraryPDF, day: DayPlan):
    _ensure_space(pdf, 18)
    day_title = f"Day {day.day_number}"
    if day.title and day.title.lower() != day_title.lower():
        day_title = f"{day_title} - {day.title}"

    pdf.set_font(STYLE["font"], "B", STYLE["section_heading_size"])
    pdf.set_text_color(22, 22, 22)
    pdf.cell(0, 7, _safe_text(day_title), ln=True)

    if _clean_value(day.date_label):
        pdf.set_font(STYLE["font"], "", STYLE["small_size"])
        pdf.set_text_color(85, 85, 85)
        pdf.cell(0, 5, _safe_text(day.date_label), ln=True)
        pdf.ln(0.8)

    for slot in day.slots:
        _render_slot_block(pdf, slot)

    pdf.ln(1.2)


def render_daily_itinerary(pdf: TravelItineraryPDF, days: list[DayPlan]):
    render_section_heading(pdf, "Daily Itinerary")
    for day in days:
        render_day_section(pdf, day)
    _draw_separator(pdf)


def render_daily_budget_summary(pdf: TravelItineraryPDF, days: list[DayPlan]):
    render_section_heading(pdf, "Daily Budget Summary")

    col_day = 24
    col_budget = 34
    col_notes = pdf.w - STYLE["margin_left"] - STYLE["margin_right"] - col_day - col_budget

    _ensure_space(pdf, 12)
    pdf.set_fill_color(238, 238, 238)
    pdf.set_draw_color(170, 170, 170)
    pdf.set_font(STYLE["font"], "B", STYLE["body_size"])
    pdf.set_text_color(30, 30, 30)

    pdf.set_x(STYLE["margin_left"])
    pdf.cell(col_day, 7, "Day", border=1, fill=True)
    pdf.cell(col_budget, 7, "Budget (USD)", border=1, fill=True)
    pdf.cell(col_notes, 7, "Notes", border=1, ln=True, fill=True)

    pdf.set_font(STYLE["font"], "", STYLE["body_size"])
    for day in days:
        _ensure_space(pdf, 10)
        note_text = day.budget_note if day.budget_note else "Estimated daily spend"
        note_text = note_text[:96]

        pdf.set_x(STYLE["margin_left"])
        pdf.cell(col_day, 7, _safe_text(str(day.day_number)), border=1)
        pdf.cell(col_budget, 7, _safe_text(f"{day.daily_budget:.2f}"), border=1)
        pdf.cell(col_notes, 7, _safe_text(note_text), border=1, ln=True)

    pdf.ln(STYLE["section_gap"] - 1.5)
    _draw_separator(pdf)


def render_important_notes(pdf: TravelItineraryPDF, notes: dict[str, list[str]]):
    render_section_heading(pdf, "Important Notes")

    for category in ["Safety", "Local Tips", "Weather Alternatives"]:
        entries = notes.get(category, [])
        _ensure_space(pdf, 9)
        pdf.set_font(STYLE["font"], "B", STYLE["sub_heading_size"])
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 6, _safe_text(category), ln=True)

        pdf.set_font(STYLE["font"], "", STYLE["body_size"])
        pdf.set_text_color(40, 40, 40)
        for line in entries:
            _ensure_space(pdf, 7)
            pdf.set_x(STYLE["margin_left"] + 4)
            pdf.multi_cell(0, STYLE["line_height"], _safe_text(f"- {line}"))

        pdf.ln(0.8)

    _draw_separator(pdf)


def render_packing_checklist(pdf: TravelItineraryPDF, checklist: dict[str, list[str]]):
    render_section_heading(pdf, "Packing Checklist")

    for category in ["Documents", "Clothing", "Essentials"]:
        items = checklist.get(category, [])
        _ensure_space(pdf, 9)

        pdf.set_font(STYLE["font"], "B", STYLE["sub_heading_size"])
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 6, _safe_text(category), ln=True)

        pdf.set_font(STYLE["font"], "", STYLE["body_size"])
        pdf.set_text_color(40, 40, 40)
        for item in items:
            _ensure_space(pdf, 7)
            pdf.set_x(STYLE["margin_left"] + 4)
            pdf.multi_cell(0, STYLE["line_height"], _safe_text(f"- {item}"))

        pdf.ln(0.6)

    _draw_separator(pdf)


def render_total_budget_summary(pdf: TravelItineraryPDF, total_budget: dict[str, float]):
    render_section_heading(pdf, "Total Budget Summary")

    rows = [
        ("Daily Subtotal", total_budget.get("daily_subtotal", 0.0)),
        ("Service Fee", total_budget.get("service_fee", 0.0)),
        ("Contingency", total_budget.get("contingency", 0.0)),
        ("Estimated Grand Total", total_budget.get("grand_total", 0.0)),
    ]

    for idx, (label, amount) in enumerate(rows):
        _ensure_space(pdf, 8)
        pdf.set_x(STYLE["margin_left"])

        font_style = "B" if idx == len(rows) - 1 else ""
        pdf.set_font(STYLE["font"], font_style, STYLE["body_size"])
        pdf.set_text_color(25, 25, 25)
        pdf.cell(58, STYLE["line_height"], _safe_text(label + ":"), border=0)

        pdf.set_font(STYLE["font"], font_style, STYLE["body_size"])
        pdf.cell(0, STYLE["line_height"], _safe_text(f"USD {amount:.2f}"), border=0, ln=True)

    pdf.ln(2)


def generate_pdf(itinerary, start_date, days):
    pdf = TravelItineraryPDF(format="A4")
    pdf.alias_nb_pages()
    pdf.set_margins(STYLE["margin_left"], STYLE["margin_top"], STYLE["margin_right"])
    pdf.set_auto_page_break(auto=True, margin=STYLE["margin_bottom"])
    pdf.add_page()

    render_header(pdf, "GlobeTrek AI - Travel Itinerary")

    summary = {
        "Destination": "Mixed itinerary",
        "Start Date": str(start_date),
        "Duration": f"{days} day(s)",
    }
    render_trip_summary(pdf, summary)

    render_section_heading(pdf, "Daily Itinerary")
    pdf.set_font(STYLE["font"], "", STYLE["body_size"])
    for row in itinerary:
        date_text = str(row.get("date", "")).strip() if isinstance(row, dict) else ""
        destination_text = str(row.get("destination", "")).strip() if isinstance(row, dict) else str(row)
        line = f"{date_text} - {destination_text}" if date_text else destination_text
        if line:
            pdf.set_x(STYLE["margin_left"] + 4)
            pdf.multi_cell(0, STYLE["line_height"], _safe_text(f"- {line}"))

    file_path = "travel_itinerary.pdf"
    pdf.output(file_path)
    return file_path


def generate_ai_itinerary_pdf(ai_itinerary_text, trip_details=None, file_path="travel_itinerary_ai.pdf"):
    trip_details = trip_details or {}
    parsed = _parse_itinerary(str(ai_itinerary_text or ""), trip_details)

    pdf = TravelItineraryPDF(format="A4")
    pdf.alias_nb_pages()
    pdf.set_margins(STYLE["margin_left"], STYLE["margin_top"], STYLE["margin_right"])
    pdf.set_auto_page_break(auto=True, margin=STYLE["margin_bottom"])
    pdf.add_page()

    render_header(pdf, "GlobeTrek AI - Travel Itinerary")
    render_trip_summary(pdf, parsed.trip_summary)
    render_overview(pdf, parsed.overview)
    render_daily_itinerary(pdf, parsed.days)
    render_daily_budget_summary(pdf, parsed.days)
    render_important_notes(pdf, parsed.important_notes)
    render_packing_checklist(pdf, parsed.packing_checklist)
    render_total_budget_summary(pdf, parsed.total_budget)

    pdf.output(file_path)
    return file_path
