"""CPU-only checks for the matched independent validation."""

from trustrag.longbench_supplement import chunk_text, official_f1, selected_pair


class CharacterTokenizer:
    def __call__(self, text, **_):
        return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}

    def apply_chat_template(self, messages, **_):
        return list(messages[0]["content"])


def test_chunking_preserves_every_original_character():
    text = "abcdefghij"
    chunks = chunk_text(CharacterTokenizer(), text, maximum=3)
    assert "".join(chunk["text"] for chunk in chunks) == text
    assert [len(chunk["text"]) for chunk in chunks] == [3, 3, 3, 1]


def test_dos_changes_only_render_order():
    tokenizer = CharacterTokenizer()
    chunks = [
        {"chunk_id": f"c{index}", "index": index, "text": text}
        for index, text in enumerate(("first", "second", "third"))
    ]
    pair = selected_pair(tokenizer, "Which?", chunks, [2, 0, 1], 1000)
    vanilla = [chunk["chunk_id"] for chunk in pair["vanilla_rag"][0]]
    dos = [chunk["chunk_id"] for chunk in pair["dos_rag"][0]]
    assert set(vanilla) == set(dos)
    assert vanilla == ["c2", "c0", "c1"]
    assert dos == ["c0", "c1", "c2"]


def test_official_english_qa_f1_matches_longbench_normalization():
    assert official_f1("The South West Ultras fan club.", ["South West Ultras fan club."]) == 1
    assert official_f1("unrelated", ["South West Ultras fan club."]) == 0
