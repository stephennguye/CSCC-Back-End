"""PhoBERT NLU adapter — placeholder with keyword-based mock.

TODO: Replace with ONNX JointIDSF + PhoBERT model inference
when trained model checkpoint is available.
"""

from __future__ import annotations

import re

from src.domain.entities.dialogue_state import NLUResult, SlotValue


# Vietnamese city name mapping
CITY_PATTERNS: dict[str, str] = {
    r"hà\s*nội|ha\s*noi|hn": "Hà Nội",
    r"hồ\s*chí\s*minh|ho\s*chi\s*minh|sài\s*gòn|sai\s*gon|hcm|sgn|tp\.?\s*hcm": "Hồ Chí Minh",
    r"đà\s*nẵng|da\s*nang|dng": "Đà Nẵng",
    r"hải\s*phòng|hai\s*phong|hpg": "Hải Phòng",
    r"nha\s*trang": "Nha Trang",
    r"huế|hue": "Huế",
    r"cần\s*thơ|can\s*tho": "Cần Thơ",
    r"phú\s*quốc|phu\s*quoc": "Phú Quốc",
    r"đà\s*lạt|da\s*lat": "Đà Lạt",
    r"quy\s*nhơn|quy\s*nhon": "Quy Nhơn",
}

AIRLINE_PATTERNS: dict[str, str] = {
    r"vietnam\s*airlines?|vna|vn\d": "Vietnam Airlines",
    r"vietjet|vj\d": "Vietjet Air",
    r"bamboo|qh\d": "Bamboo Airways",
    r"pacific|bl\d": "Pacific Airlines",
}

# Intent keyword patterns
INTENT_PATTERNS: list[tuple[str, list[str], float]] = [
    ("atis_flight", ["chuyến bay", "bay", "flight", "đặt vé", "dat ve", "tìm chuyến", "book"], 0.85),
    ("atis_airfare", ["giá vé", "gia ve", "bao nhiêu", "fare", "chi phí", "chi phi", "giá"], 0.85),
    ("atis_airline", ["hãng", "hang", "airline", "hàng không", "hang khong"], 0.85),
    (
        "atis_ground_service",
        ["dịch vụ", "dich vu", "sân bay", "san bay", "ground", "đón", "don"],
        0.80,
    ),
    (
        "atis_abbreviation",
        ["viết tắt", "viet tat", "nghĩa là", "nghia la", "là gì", "la gi"],
        0.80,
    ),
    (
        "affirm",
        ["đúng", "dung", "vâng", "vang", "ừ", "đúng rồi", "dung roi", "ok", "yes", "xác nhận", "xac nhan"],
        0.90,
    ),
    ("deny", ["không", "khong", "sai", "không đúng", "khong dung", "sai rồi", "no"], 0.90),
    ("greet", ["xin chào", "xin chao", "chào", "chao", "hello", "hi"], 0.95),
    (
        "farewell",
        ["tạm biệt", "tam biet", "bye", "goodbye", "cảm ơn", "cam on", "hẹn gặp"],
        0.90,
    ),
]


class PhoBERTNLUAdapter:
    """Placeholder NLU adapter using keyword matching.

    Implements the NLUPort protocol. Will be replaced with
    JointIDSF + PhoBERT ONNX inference when model is trained.
    """

    async def understand(self, text: str) -> NLUResult:
        """Parse Vietnamese text into intent and slots."""
        lower = text.lower().strip()
        slots = self._extract_slots(lower, text)
        intent, confidence = self._classify_intent(lower)

        # Boost confidence if slots were extracted for flight intent
        if intent == "atis_flight" and slots:
            confidence = min(confidence + 0.1, 0.99)

        return NLUResult(
            intent=intent,
            intent_confidence=confidence,
            slots=slots,
            raw_text=text,
        )

    def _classify_intent(self, text: str) -> tuple[str, float]:
        """Classify intent using keyword patterns."""
        best_intent = "atis_flight"
        best_confidence = 0.5

        for intent, keywords, base_conf in INTENT_PATTERNS:
            for kw in keywords:
                if kw in text:
                    if base_conf > best_confidence:
                        best_intent = intent
                        best_confidence = base_conf
                    break

        return best_intent, best_confidence

    def _extract_slots(self, lower: str, original: str) -> list[SlotValue]:
        """Extract slot values from text using pattern matching."""
        slots: list[SlotValue] = []
        found_cities: list[str] = []

        # Extract cities
        for pattern, city_name in CITY_PATTERNS.items():
            if re.search(pattern, lower):
                found_cities.append(city_name)

        # Assign cities to fromloc/toloc based on context
        if found_cities:
            from_markers = ["từ", "tu", "from", "ở", "o", "tại", "tai"]
            to_markers = ["đến", "den", "tới", "toi", "to", "về", "ve", "đi", "di"]

            if len(found_cities) >= 2:
                slots.append(SlotValue(
                    name="fromloc.city_name",
                    value=found_cities[0],
                    confidence=0.85,
                ))
                slots.append(SlotValue(
                    name="toloc.city_name",
                    value=found_cities[1],
                    confidence=0.85,
                ))
            elif len(found_cities) == 1:
                # Check context for from/to
                city = found_cities[0]
                is_from = any(m in lower for m in from_markers)
                is_to = any(m in lower for m in to_markers)

                if is_from and not is_to:
                    slots.append(SlotValue(
                        name="fromloc.city_name", value=city, confidence=0.80,
                    ))
                elif is_to and not is_from:
                    slots.append(SlotValue(
                        name="toloc.city_name", value=city, confidence=0.80,
                    ))
                else:
                    # Default: assume destination
                    slots.append(SlotValue(
                        name="toloc.city_name", value=city, confidence=0.60,
                    ))

        # Extract airline
        for pattern, airline_name in AIRLINE_PATTERNS.items():
            if re.search(pattern, lower):
                slots.append(SlotValue(
                    name="airline_name", value=airline_name, confidence=0.90,
                ))
                break

        # Extract class type
        if any(w in lower for w in ["thương gia", "thuong gia", "business"]):
            slots.append(SlotValue(
                name="class_type", value="thương gia", confidence=0.90,
            ))
        elif any(w in lower for w in ["phổ thông", "pho thong", "economy"]):
            slots.append(SlotValue(
                name="class_type", value="phổ thông", confidence=0.90,
            ))
        elif any(w in lower for w in ["hạng nhất", "hang nhat", "first"]):
            slots.append(SlotValue(
                name="class_type", value="hạng nhất", confidence=0.90,
            ))

        # Extract round trip
        if any(w in lower for w in ["khứ hồi", "khu hoi", "round trip", "hai chiều"]):
            slots.append(SlotValue(
                name="round_trip", value="khứ hồi", confidence=0.85,
            ))
        elif any(w in lower for w in ["một chiều", "mot chieu", "one way"]):
            slots.append(SlotValue(
                name="round_trip", value="một chiều", confidence=0.85,
            ))

        # Extract day names
        day_patterns: dict[str, str] = {
            r"thứ\s*hai|thu\s*hai|monday": "thứ hai",
            r"thứ\s*ba|thu\s*ba|tuesday": "thứ ba",
            r"thứ\s*tư|thu\s*tu|wednesday": "thứ tư",
            r"thứ\s*năm|thu\s*nam|thursday": "thứ năm",
            r"thứ\s*sáu|thu\s*sau|friday": "thứ sáu",
            r"thứ\s*bảy|thu\s*bay|saturday": "thứ bảy",
            r"chủ\s*nhật|chu\s*nhat|sunday": "chủ nhật",
        }
        for pattern, day_name in day_patterns.items():
            if re.search(pattern, lower):
                slots.append(SlotValue(
                    name="depart_date.day_name", value=day_name, confidence=0.85,
                ))
                break

        # Extract "ngày mai" (tomorrow), "hôm nay" (today)
        if any(w in lower for w in ["ngày mai", "ngay mai", "tomorrow"]):
            slots.append(SlotValue(
                name="depart_date.day_name", value="ngày mai", confidence=0.90,
            ))
        elif any(w in lower for w in ["hôm nay", "hom nay", "today"]):
            slots.append(SlotValue(
                name="depart_date.day_name", value="hôm nay", confidence=0.90,
            ))

        return slots
