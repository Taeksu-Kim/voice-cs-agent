"""APO 후보 관리 (scripts/apo_candidate.py)와 지시문 덮어쓰기 (protocol.py CSAGENT_PROTOCOL_OVERRIDE)."""
import yaml
import pytest

from csagent.agent.protocol import OVERRIDE_ENV, PROTOCOL_DIR, load_protocol
from scripts.apo_candidate import CandidateError, add, new, write_doc
from tests.test_triage_save import REPORT

ORIGINAL = (PROTOCOL_DIR / "exchange_return.yaml").read_text(encoding="utf-8")


def _exp(tmp_path):
    case = tmp_path / "case1"
    case.mkdir()
    (case / "case.json").write_text("{}")
    (case / "triage.md").write_text(REPORT, encoding="utf-8")
    return case, new(case, 0, tmp_path)


def _edit(fn):
    raw = yaml.safe_load(ORIGINAL)
    fn(raw)
    return yaml.safe_dump(raw, allow_unicode=True, sort_keys=False)


def test_new_takes_only_prompt_layer_issues(tmp_path):
    case, exp = _exp(tmp_path)
    assert (exp / "source.json").exists() and exp.parent.name == "apo"
    with pytest.raises(CandidateError, match="layer=code"):
        new(case, 1, tmp_path)                       # 둘째 문제는 code 층


def test_add_saves_copy_and_diff_without_touching_repo(tmp_path):
    _, exp = _exp(tmp_path)
    anchor = "  - 고객의 용건이 해결되면 intent를 done으로 한다.\n"
    text = ORIGINAL.replace(anchor, "  - 고객이 이번에 한 말에 대한 답이 되게 한다.\n" + anchor)   # 실제 후보처럼 원문을 고친다
    d = add(exp, "c1", "exchange_return", text)
    assert "+  - 고객이 이번에 한 말" in (d / "diff.patch").read_text(encoding="utf-8")
    assert (PROTOCOL_DIR / "exchange_return.yaml").read_text(encoding="utf-8") == ORIGINAL
    assert write_doc(exp, "c1/rationale.md", "원칙 점검").exists()
    with pytest.raises(CandidateError, match="후보가 없다"):
        write_doc(exp, "c9/rationale.md", "x")


@pytest.mark.parametrize("fn, msg", [
    (lambda r: r["steps"][0].update(tools=["send_app_link"]), "tools가 바뀌었다"),
    (lambda r: r["steps"].reverse(), "id·순서"),
    (lambda r: r.update(always_tools=["end_call"]), "always_tools"),
    (lambda r: r.update(extra="x"), "없던 키"),
])
def test_add_rejects_structure_changes(tmp_path, fn, msg):
    _, exp = _exp(tmp_path)
    with pytest.raises(CandidateError, match=msg):
        add(exp, "c1", "exchange_return", _edit(fn))


def test_override_env_loads_candidate(tmp_path, monkeypatch):
    _, exp = _exp(tmp_path)
    d = add(exp, "c1", "exchange_return", _edit(lambda r: r.update(common="후보 지시\n")))
    monkeypatch.setenv(OVERRIDE_ENV, str(d))
    assert load_protocol("exchange_return").common == "후보 지시\n"
    assert load_protocol("general").name == "general"     # 후보 폴더에 없는 프로토콜은 원본
    monkeypatch.delenv(OVERRIDE_ENV)
    assert load_protocol("exchange_return").common != "후보 지시\n"
