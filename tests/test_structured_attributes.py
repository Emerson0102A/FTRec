from ftrec.attributes.structured import build_prompt, parse_attributes


def test_parse_attributes_accepts_json_fence_and_deduplicates():
    result = parse_attributes(
        '```json\n{"attributes": ["Waterproof", "waterproof", "Light weight", "Outdoor"]}\n```',
        3,
    )
    assert result == ["Waterproof", "Light weight", "Outdoor"]


def test_parse_attributes_pads_malformed_generation():
    assert parse_attributes("red; compact", 3) == ["red", "compact", ""]


def test_controlled_prompt_only_contains_selected_fields():
    row = {
        "title": "Trail Shoe",
        "features": ["waterproof"],
        "details": {"Color": "Blue"},
    }
    prompt = build_prompt(
        row,
        attribute_count=2,
        input_fields=("title",),
        max_field_characters=100,
    )
    assert "Trail Shoe" in prompt
    assert "waterproof" not in prompt
    assert "Blue" not in prompt
    assert '"attribute 2"' in prompt
