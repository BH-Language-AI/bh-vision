import json


def check_row(r):
    assert isinstance(r["messages"], list) and len(r["messages"]) >= 2
    assert r["messages"][-1]["role"] == "assistant"
    assert isinstance(r["images"], list) and r["images"]
    for m in r["messages"]:
        assert m["role"] in ("user", "assistant", "system")
        assert isinstance(m["content"], list) and m["content"]


def test_contract_sample():
    r = {"messages": [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Šta piše?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Račun čćžšđ"}]}],
        "images": ["data/img/001.jpg"]}
    check_row(r)
    json.dumps(r, ensure_ascii=False)
