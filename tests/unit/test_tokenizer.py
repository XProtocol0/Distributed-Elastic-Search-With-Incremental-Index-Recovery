from dsearch.index.tokenizer import query_terms, tokenize


def test_lowercases_and_splits_on_non_alphanumerics() -> None:
    assert tokenize("Hello, World! foo_bar 42x") == ["hello", "world", "foo", "bar", "42x"]


def test_keeps_duplicates_and_unicode_letters() -> None:
    assert tokenize("Café café — naïve") == ["café", "café", "naïve"]


def test_empty_and_punctuation_only() -> None:
    assert tokenize("") == []
    assert tokenize("... --- !!!") == []


def test_query_terms_unique_and_sorted() -> None:
    assert query_terms("zeta Alpha alpha beta") == ["alpha", "beta", "zeta"]
