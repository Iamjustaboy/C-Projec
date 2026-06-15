from src.rag.prompts import SYSTEM_PROMPT


def test_prompt_requires_grounded_unknown_answer():
    assert "不知道" in SYSTEM_PROMPT
    assert "不要编造" in SYSTEM_PROMPT
    assert "检索到的企业资料" in SYSTEM_PROMPT or "企业资料" in SYSTEM_PROMPT

