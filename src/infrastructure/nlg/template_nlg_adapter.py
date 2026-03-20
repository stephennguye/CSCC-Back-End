"""Template-based Natural Language Generation adapter.

Uses Vietnamese response templates for each policy action type.
"""

from __future__ import annotations

import random
import string

from src.domain.entities.dialogue_state import (
    DialogueState,
    PolicyAction,
    PolicyDecision,
)

# Slot name -> Vietnamese prompt for requesting that slot
_SLOT_PROMPTS: dict[str, str] = {
    "fromloc.city_name": "Xin vui lòng cho biết bạn muốn bay từ thành phố nào?",
    "toloc.city_name": "Bạn muốn bay đến thành phố nào?",
    "depart_date": "Bạn muốn khởi hành vào ngày nào?",
    "depart_date.day_name": "Bạn muốn bay vào ngày nào trong tuần?",
    "depart_date.month_name": "Bạn muốn bay vào tháng nào?",
    "depart_date.day_number": "Bạn muốn bay vào ngày mấy?",
    "depart_time.time": "Bạn muốn bay vào lúc mấy giờ?",
    "airline_name": "Bạn muốn bay với hãng hàng không nào?",
    "class_type": "Bạn muốn đặt hạng ghế nào (phổ thông, thương gia, hạng nhất)?",
    "round_trip": "Bạn muốn đặt vé một chiều hay khứ hồi?",
    "return_date": "Bạn muốn bay về vào ngày nào?",
}


def _format_date(filled: dict[str, str], prefix: str) -> str | None:
    """Build a human-readable date string from date sub-slots."""
    if f"{prefix}.day_number" in filled:
        day = filled[f"{prefix}.day_number"]
        # day should already be cleaned by DST (just the number like "20")
        # but safeguard against double "ngày ngày" just in case
        if day.lower().startswith("ngày"):
            date_str = day
        else:
            date_str = f"ngày {day}"
        if f"{prefix}.month_name" in filled:
            month = filled[f"{prefix}.month_name"]
            date_str += f" {month}"
        return date_str
    if f"{prefix}.day_name" in filled:
        return filled[f"{prefix}.day_name"]
    if f"{prefix}.today_relative" in filled:
        return filled[f"{prefix}.today_relative"]
    return None


def _build_booking_summary(filled: dict[str, str]) -> list[str]:
    """Build list of Vietnamese description parts from filled slots."""
    parts: list[str] = []
    if "fromloc.city_name" in filled:
        parts.append(f"từ {filled['fromloc.city_name']}")
    if "toloc.city_name" in filled:
        parts.append(f"đến {filled['toloc.city_name']}")
    depart = _format_date(filled, "depart_date")
    if depart:
        parts.append(f"ngày đi {depart}")
    if "depart_time.time" in filled:
        parts.append(f"lúc {filled['depart_time.time']}")
    if "airline_name" in filled:
        parts.append(f"hãng {filled['airline_name']}")
    if "class_type" in filled:
        parts.append(f"hạng {filled['class_type']}")
    if "round_trip" in filled:
        parts.append(f"loại vé {filled['round_trip']}")
    ret = _format_date(filled, "return_date")
    if ret:
        parts.append(f"ngày về {ret}")
    return parts


def _estimate_price(filled: dict[str, str]) -> str:
    """Generate a realistic fake price based on class and round-trip type."""
    # Base prices in VND for domestic flights
    base_prices = {
        "phổ thông": 1_200_000,
        "thương gia": 3_500_000,
        "hạng nhất": 6_000_000,
    }
    class_type = (filled.get("class_type") or "phổ thông").lower()
    base = base_prices.get(class_type, 1_500_000)

    # Add variation based on airline
    airline = (filled.get("airline_name") or "").lower()
    if "vietjet" in airline:
        base = int(base * 0.85)
    elif "bamboo" in airline:
        base = int(base * 0.90)

    # Round-trip doubles the price
    round_trip = (filled.get("round_trip") or "").lower()
    if round_trip in ("khứ hồi", "khu hoi", "round trip", "round-trip"):
        base *= 2

    return f"{base:,.0f} VNĐ".replace(",", ".")


def _generate_ticket(filled: dict[str, str]) -> str:
    """Generate a realistic fake ticket confirmation message."""
    # Booking code: 6 uppercase alphanumeric chars
    booking_code = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))

    # Flight number based on airline
    airline = filled.get("airline_name", "")
    airline_lower = airline.lower()
    if "vietjet" in airline_lower:
        prefix = "VJ"
    elif "bamboo" in airline_lower:
        prefix = "QH"
    else:
        # Default to Vietnam Airlines
        prefix = "VN"
    flight_number = f"{prefix}-{random.randint(100, 9999)}"

    # Build ticket lines
    lines = [
        "Đặt vé thành công! Thông tin chuyến bay của bạn:",
        f"- Mã đặt chỗ: {booking_code}",
        f"- Chuyến bay: {flight_number}",
    ]

    from_city = filled.get("fromloc.city_name")
    to_city = filled.get("toloc.city_name")
    if from_city and to_city:
        lines.append(f"- Hành trình: {from_city} → {to_city}")

    depart = _format_date(filled, "depart_date")
    if depart:
        lines.append(f"- Ngày đi: {depart}")

    depart_time = filled.get("depart_time.time")
    if depart_time:
        lines.append(f"- Giờ khởi hành: {depart_time}")

    if airline:
        lines.append(f"- Hãng: {airline}")

    class_type = filled.get("class_type")
    if class_type:
        lines.append(f"- Hạng ghế: {class_type.capitalize()}")

    round_trip = filled.get("round_trip")
    if round_trip:
        lines.append(f"- Loại vé: {round_trip.capitalize()}")

    ret = _format_date(filled, "return_date")
    if ret:
        lines.append(f"- Ngày về: {ret}")

    # Price estimate based on route and class
    price = _estimate_price(filled)
    lines.append(f"- Giá vé: {price}")
    lines.append("- Thanh toán: Chuyển khoản ngân hàng, thẻ tín dụng/ghi nợ, hoặc thanh toán tại quầy.")
    lines.append("- Cổng sẽ được thông báo tại sân bay.")
    lines.append("Cảm ơn bạn đã sử dụng dịch vụ!")

    return "\n".join(lines)


class TemplateNLGAdapter:
    """Template-based NLG for Vietnamese airline booking dialogue.

    Implements the NLGPort protocol.
    """

    def generate(
        self,
        decision: PolicyDecision,
        state: DialogueState,
    ) -> str:
        """Generate Vietnamese response text from policy decision and state."""
        action = decision.action

        if action == PolicyAction.GREET:
            return (
                "Xin chào! Tôi là trợ lý đặt vé máy bay. "
                "Tôi có thể giúp gì cho bạn hôm nay?"
            )

        if action == PolicyAction.FAREWELL:
            return "Cảm ơn bạn đã sử dụng dịch vụ. Chúc bạn một ngày tốt lành!"

        if action == PolicyAction.REQUEST_SLOT:
            slot = decision.target_slot or ""
            prompt = _SLOT_PROMPTS.get(slot, f"Xin vui lòng cung cấp thông tin: {slot}")
            return prompt

        if action in (PolicyAction.CONFIRM, PolicyAction.EXECUTE):
            filled = state.filled_slots()
            parts = _build_booking_summary(filled)
            summary = ", ".join(parts) if parts else "thông tin đã cung cấp"

            if action == PolicyAction.CONFIRM:
                price = _estimate_price(filled)
                return (
                    f"Bạn muốn đặt chuyến bay {summary}. "
                    f"Giá vé ước tính: {price}. "
                    "Thanh toán bằng chuyển khoản, thẻ tín dụng, hoặc tại quầy. "
                    "Xin xác nhận thông tin này đúng không?"
                )
            return _generate_ticket(filled)

        if action == PolicyAction.PROVIDE_INFO:
            return (
                "Yêu cầu đặt vé của bạn đã được xử lý. "
                "Bạn cần đặt thêm chuyến bay khác không?"
            )

        if action == PolicyAction.ESCALATE:
            return (
                "Xin lỗi, tôi không thể xử lý yêu cầu này. "
                "Tôi sẽ chuyển bạn đến nhân viên hỗ trợ."
            )

        if action == PolicyAction.CLARIFY:
            filled = state.filled_slots()
            if filled:
                return (
                    "Bạn muốn thay đổi thông tin nào? "
                    "Xin vui lòng cho biết thông tin cần sửa."
                )
            return "Xin lỗi, tôi chưa hiểu rõ. Bạn có thể nói lại được không?"

        return "Tôi có thể giúp gì thêm cho bạn?"
