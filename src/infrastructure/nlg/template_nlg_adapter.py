"""Template-based Natural Language Generation adapter.

Uses Vietnamese response templates for each policy action type.
"""

from __future__ import annotations

from src.domain.entities.dialogue_state import (
    DialogueState,
    PolicyAction,
    PolicyDecision,
)

# Slot name -> Vietnamese prompt for requesting that slot
_SLOT_PROMPTS: dict[str, str] = {
    "fromloc.city_name": "Xin vui lòng cho biết bạn muốn bay từ thành phố nào?",
    "toloc.city_name": "Bạn muốn bay đến thành phố nào?",
    "depart_date.day_name": "Bạn muốn bay vào ngày nào?",
    "depart_date.month_name": "Bạn muốn bay vào tháng nào?",
    "depart_date.day_number": "Bạn muốn bay vào ngày mấy?",
    "depart_time.time": "Bạn muốn bay vào lúc mấy giờ?",
    "airline_name": "Bạn muốn bay với hãng hàng không nào?",
    "flight_number": "Bạn có biết số hiệu chuyến bay không?",
    "class_type": "Bạn muốn đặt hạng ghế nào (phổ thông, thương gia, hạng nhất)?",
    "round_trip": "Bạn muốn đặt vé một chiều hay khứ hồi?",
}


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

        if action == PolicyAction.CONFIRM:
            filled = state.filled_slots()
            parts = []
            if "fromloc.city_name" in filled:
                parts.append(f"từ {filled['fromloc.city_name']}")
            if "toloc.city_name" in filled:
                parts.append(f"đến {filled['toloc.city_name']}")
            if "depart_date.day_number" in filled:
                date_str = filled["depart_date.day_number"]
                if "depart_date.month_name" in filled:
                    date_str += f" {filled['depart_date.month_name']}"
                parts.append(f"ngày {date_str}")
            elif "depart_date.day_name" in filled:
                parts.append(f"vào {filled['depart_date.day_name']}")
            elif "depart_date.today_relative" in filled:
                parts.append(f"vào {filled['depart_date.today_relative']}")
            if "depart_time.time" in filled:
                parts.append(f"lúc {filled['depart_time.time']}")
            if "airline_name" in filled:
                parts.append(f"hãng {filled['airline_name']}")
            if "class_type" in filled:
                parts.append(f"hạng {filled['class_type']}")
            if "round_trip" in filled:
                parts.append(f"loại vé {filled['round_trip']}")

            summary = ", ".join(parts) if parts else "thông tin đã cung cấp"
            return (
                f"Bạn muốn đặt chuyến bay {summary}. "
                "Xin xác nhận thông tin này đúng không?"
            )

        if action == PolicyAction.EXECUTE:
            filled = state.filled_slots()
            origin = filled.get("fromloc.city_name", "")
            dest = filled.get("toloc.city_name", "")
            return (
                f"Tôi đã tìm thấy các chuyến bay từ {origin} đến {dest}. "
                "Hệ thống đang xử lý yêu cầu đặt vé của bạn."
            )

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
            return "Xin lỗi, tôi chưa hiểu rõ. Bạn có thể nói lại được không?"

        return "Tôi có thể giúp gì thêm cho bạn?"
