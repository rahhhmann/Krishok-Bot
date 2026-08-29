"""Unit tests for agent/tools.py.

Focus: pure text-normalization helpers (deterministic, no I/O) and
get_weather() with HTTP entirely mocked -- no real OpenWeatherMap calls.
DAM price-scraping internals (_get_division_pdf_price etc.) are not
covered here: they need a real/representative DAM HTML+PDF fixture to
test meaningfully rather than mock into meaninglessness, and are better
covered by an integration-style test against a saved sample page/PDF if
one is added later.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from agent import config as agent_config
from agent import tools


# --- text/digit normalization ------------------------------------------


def test_normalize_digits_converts_bangla_digits():
    assert tools.normalize_digits("১২৩") == "123"


def test_normalize_digits_leaves_latin_digits_alone():
    assert tools.normalize_digits("123") == "123"


def test_normalize_text_collapses_whitespace_and_dashes():
    # normalize_text also converts Bangla digits to Latin (via normalize_digits)
    assert tools.normalize_text("আলু   –  ৫০ টাকা") == "আলু - 50 টাকা"


def test_normalize_text_empty_input():
    assert tools.normalize_text("") == ""
    assert tools.normalize_text(None) == ""


# --- crop / division normalization --------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("আলু", "potato"),
        ("potato", "potato"),
        ("টমেটো", "tomato"),
        ("রসুন", "garlic"),
        ("unknown_crop_xyz", "unknown_crop_xyz"),  # unmapped -> passthrough, not guessed
    ],
)
def test_normalize_crop_name(raw, expected):
    assert tools.normalize_crop_name(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("dhaka", "ঢাকা"),
        ("Barisal", "বরিশাল"),
        ("বরিশাল", "বরিশাল"),
        (None, ""),
    ],
)
def test_normalize_division(raw, expected):
    assert tools.normalize_division(raw) == expected


def test_canonical_division_defaults_to_config_default():
    assert tools._canonical_division(None) == agent_config.DAM_DEFAULT_DIVISION


# --- unit normalization ---------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("কেজি", "প্রতি কেজি"),
        ("লিটার", "প্রতি লিটার"),
        ("হালি", "প্রতি হালি"),
    ],
)
def test_normalize_unit_known_variants(raw, expected):
    assert tools.normalize_unit(raw) == expected


def test_normalize_unit_unknown_passthrough():
    assert tools.normalize_unit("অজানা একক") == "অজানা একক"


# --- get_weather: mocked HTTP --------------------------------------------


def test_get_weather_no_location_returns_unavailable():
    result = tools.get_weather("")
    assert result.available is False
    assert "No location" in result.error


def test_get_weather_missing_api_key(monkeypatch):
    monkeypatch.setattr(agent_config, "OPENWEATHER_API_KEY", "")
    result = tools.get_weather("Barishal")
    assert result.available is False
    assert "not configured" in result.error


def test_get_weather_known_location_success(monkeypatch):
    monkeypatch.setattr(agent_config, "OPENWEATHER_API_KEY", "dummy-key")
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "name": "Barishal",
        "main": {"temp": 31.5, "humidity": 78},
        "weather": [{"description": "broken clouds"}],
        "rain": {},
    }
    with patch.object(tools, "_weather_get", return_value=fake_response) as mocked_get:
        result = tools.get_weather("Barishal")
    assert result.available is True
    assert result.location == "Barishal"
    assert result.temperature_c == 31.5
    assert result.humidity_percent == 78
    # Barishal is in BD_LOCATIONS -> geocoding endpoint must NOT be hit,
    # only the weather endpoint itself.
    assert mocked_get.call_count == 1


def test_get_weather_unknown_location_uses_geocoding(monkeypatch):
    monkeypatch.setattr(agent_config, "OPENWEATHER_API_KEY", "dummy-key")
    geocode_response = MagicMock()
    geocode_response.json.return_value = [{"lat": 23.0, "lon": 91.0, "country": "BD"}]
    weather_response = MagicMock()
    weather_response.json.return_value = {
        "name": "Comilla",
        "main": {"temp": 29.0, "humidity": 70},
        "weather": [{"description": "clear sky"}],
    }
    with patch.object(
        tools, "_weather_get", side_effect=[geocode_response, weather_response]
    ) as mocked_get:
        result = tools.get_weather("Comilla")
    assert result.available is True
    assert result.location == "Comilla"
    assert mocked_get.call_count == 2  # geocode then weather


def test_get_weather_location_not_found(monkeypatch):
    monkeypatch.setattr(agent_config, "OPENWEATHER_API_KEY", "dummy-key")
    geocode_response = MagicMock()
    geocode_response.json.return_value = []  # nothing found
    with patch.object(tools, "_weather_get", return_value=geocode_response):
        result = tools.get_weather("Nonexistent Place")
    assert result.available is False
    assert "was not found" in result.error


def test_get_weather_timeout_returns_unavailable(monkeypatch):
    monkeypatch.setattr(agent_config, "OPENWEATHER_API_KEY", "dummy-key")
    with patch.object(tools, "_weather_get", side_effect=requests.Timeout()):
        result = tools.get_weather("Barishal")
    assert result.available is False
    assert "timed out" in result.error


def test_get_weather_http_401_reports_invalid_key(monkeypatch):
    monkeypatch.setattr(agent_config, "OPENWEATHER_API_KEY", "dummy-key")
    fake_resp = MagicMock(status_code=401)
    err = requests.HTTPError(response=fake_resp)
    with patch.object(tools, "_weather_get", side_effect=err):
        result = tools.get_weather("Barishal")
    assert result.available is False
    assert "invalid" in result.error.lower()
