"""Tests for src.infrastructure.nlu.phobert_nlu_adapter."""

from __future__ import annotations

import pytest

from src.infrastructure.nlu.phobert_nlu_adapter import PhoBERTNLUAdapter


@pytest.fixture()
def nlu() -> PhoBERTNLUAdapter:
    return PhoBERTNLUAdapter()


class TestIntentClassification:
    async def test_flight_intent(self, nlu: PhoBERTNLUAdapter) -> None:
        result = await nlu.understand("Tôi muốn đặt vé bay đi Đà Nẵng")
        assert result.intent == "atis_flight"
        assert result.intent_confidence >= 0.85

    async def test_airfare_intent(self, nlu: PhoBERTNLUAdapter) -> None:
        result = await nlu.understand("giá vé đắt lắm")
        assert result.intent == "atis_airfare"

    async def test_airline_intent(self, nlu: PhoBERTNLUAdapter) -> None:
        result = await nlu.understand("tôi muốn biết về hãng airline")
        assert result.intent == "atis_airline"

    async def test_greet_intent(self, nlu: PhoBERTNLUAdapter) -> None:
        result = await nlu.understand("xin chào")
        assert result.intent == "greet"
        assert result.intent_confidence >= 0.90

    async def test_farewell_intent(self, nlu: PhoBERTNLUAdapter) -> None:
        result = await nlu.understand("tạm biệt")
        assert result.intent == "farewell"

    async def test_affirm_intent(self, nlu: PhoBERTNLUAdapter) -> None:
        result = await nlu.understand("đúng rồi")
        assert result.intent == "affirm"
        assert result.intent_confidence >= 0.90

    async def test_deny_intent(self, nlu: PhoBERTNLUAdapter) -> None:
        result = await nlu.understand("sai rồi")
        assert result.intent == "deny"

    async def test_default_intent_for_ambiguous(self, nlu: PhoBERTNLUAdapter) -> None:
        result = await nlu.understand("ngày mai")
        # Default fallback is atis_flight
        assert result.intent == "atis_flight"


class TestSlotExtraction:
    async def test_two_cities(self, nlu: PhoBERTNLUAdapter) -> None:
        result = await nlu.understand("bay từ Hà Nội đến Đà Nẵng")
        slot_dict = {s.name: s.value for s in result.slots}
        assert slot_dict["fromloc.city_name"] == "Hà Nội"
        assert slot_dict["toloc.city_name"] == "Đà Nẵng"

    async def test_single_city_with_from_marker(self, nlu: PhoBERTNLUAdapter) -> None:
        result = await nlu.understand("từ Hà Nội")
        slot_dict = {s.name: s.value for s in result.slots}
        assert slot_dict.get("fromloc.city_name") == "Hà Nội"
        assert "toloc.city_name" not in slot_dict

    async def test_single_city_with_to_marker(self, nlu: PhoBERTNLUAdapter) -> None:
        result = await nlu.understand("đi Đà Nẵng")
        slot_dict = {s.name: s.value for s in result.slots}
        assert slot_dict.get("toloc.city_name") == "Đà Nẵng"

    async def test_single_city_defaults_to_destination(self, nlu: PhoBERTNLUAdapter) -> None:
        result = await nlu.understand("Hồ Chí Minh")
        slot_dict = {s.name: s.value for s in result.slots}
        assert "toloc.city_name" in slot_dict

    async def test_airline_extraction(self, nlu: PhoBERTNLUAdapter) -> None:
        result = await nlu.understand("bay Vietnam Airlines")
        slot_dict = {s.name: s.value for s in result.slots}
        assert slot_dict.get("airline_name") == "Vietnam Airlines"

    async def test_class_type_business(self, nlu: PhoBERTNLUAdapter) -> None:
        result = await nlu.understand("hạng thương gia")
        slot_dict = {s.name: s.value for s in result.slots}
        assert slot_dict.get("class_type") == "thương gia"

    async def test_class_type_economy(self, nlu: PhoBERTNLUAdapter) -> None:
        result = await nlu.understand("hạng phổ thông")
        slot_dict = {s.name: s.value for s in result.slots}
        assert slot_dict.get("class_type") == "phổ thông"

    async def test_round_trip(self, nlu: PhoBERTNLUAdapter) -> None:
        result = await nlu.understand("vé khứ hồi")
        slot_dict = {s.name: s.value for s in result.slots}
        assert slot_dict.get("round_trip") == "khứ hồi"

    async def test_one_way(self, nlu: PhoBERTNLUAdapter) -> None:
        result = await nlu.understand("vé một chiều")
        slot_dict = {s.name: s.value for s in result.slots}
        assert slot_dict.get("round_trip") == "một chiều"

    async def test_day_name_extraction(self, nlu: PhoBERTNLUAdapter) -> None:
        result = await nlu.understand("bay ngày thứ sáu")
        slot_dict = {s.name: s.value for s in result.slots}
        assert slot_dict.get("depart_date.day_name") == "thứ sáu"

    async def test_tomorrow_extraction(self, nlu: PhoBERTNLUAdapter) -> None:
        result = await nlu.understand("bay ngày mai")
        slot_dict = {s.name: s.value for s in result.slots}
        assert slot_dict.get("depart_date.day_name") == "ngày mai"

    async def test_confidence_boost_with_slots(self, nlu: PhoBERTNLUAdapter) -> None:
        result = await nlu.understand("đặt vé đi Đà Nẵng")
        # Flight intent + slots should boost confidence
        assert result.intent_confidence > 0.85

    async def test_no_slots_for_greeting(self, nlu: PhoBERTNLUAdapter) -> None:
        result = await nlu.understand("xin chào")
        assert result.slots == []

    async def test_hcm_aliases(self, nlu: PhoBERTNLUAdapter) -> None:
        for alias in ["sài gòn", "tp hcm", "hcm"]:
            result = await nlu.understand(f"đi {alias}")
            slot_dict = {s.name: s.value for s in result.slots}
            assert "Hồ Chí Minh" in slot_dict.values(), f"Failed for alias: {alias}"
