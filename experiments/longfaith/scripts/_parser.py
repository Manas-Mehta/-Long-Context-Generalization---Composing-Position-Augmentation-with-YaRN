"""Answer-letter parser for LongBench v2 evaluations.

Factored out of eval_longbench_v2.py so rescore.py can re-parse existing
predictions without importing torch.

Three-stage parser:
  1. PRIMARY:  "The answer is X" where X ∈ {A,B,C,D}
  2. FALLBACK: last standalone A/B/C/D in trailing ~300 chars
  3. RECOVERY: narrative answers that name the choice content but never emit a
              letter. Maps the model's prose to a letter by frequency-scoring
              each choice's *distinctive* tokens (tokens present in choice X
              but not in the other three) inside the last ~800 chars of output.
              Critical for evals on models that prefer narrative form — without
              this, parser misses undercount narrative-form correct answers and
              bias YaRN-vs-noYaRN comparisons. See
              notes/longfaith_diagnosis_2026-05-15.md for the diagnostic story.
"""

import re

_ANS_PRIMARY  = re.compile(r"[Tt]he answer is\s*[:\-]?\s*\(?\s*([ABCD])\b")
_ANS_FALLBACK = re.compile(r"\b([ABCD])\b")
_ANS_EXPLICIT = re.compile(r"\b(?:answer|choice|option)\b\D{0,10}\b([ABCD])\b", re.IGNORECASE)

_STOP = frozenset((
    "the a an of in to for and or with on by from at is are was were as that "
    "this these those it its be have has had will would can could may might "
    "should do does did not but if then so than which who whom whose what when "
    "where why how"
).split())


def _tokens(s: str) -> list[str]:
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    return [t for t in s.split() if len(t) > 2 and t not in _STOP]


def _recover_from_choices(raw_output: str, choices: dict | None) -> str | None:
    if not choices:
        return None
    m = _ANS_EXPLICIT.search(raw_output[-400:])
    if m:
        return m.group(1).upper()
    tail = raw_output[-800:].lower()
    scores: dict = {}
    for letter, text in choices.items():
        toks = _tokens(text)
        if not toks:
            continue
        other_toks = set()
        for L, T in choices.items():
            if L != letter:
                other_toks.update(_tokens(T))
        distinctive = [t for t in toks if t not in other_toks] or toks
        scores[letter] = sum(tail.count(t) for t in distinctive)
    if not scores:
        return None
    sv = sorted(scores.values(), reverse=True)
    if sv[0] >= 2 and sv[0] > sv[1]:
        return max(scores, key=scores.get)
    return None


def parse_answer(raw_output: str, choices: dict | None = None) -> tuple:
    """Return (parsed_letter, parser_stage).

    parser_stage ∈ {"primary", "fallback", "recovery", "miss"}.
    parsed_letter is None iff parser_stage == "miss".
    """
    m = _ANS_PRIMARY.search(raw_output)
    if m:
        return m.group(1), "primary"
    tail = raw_output[-300:]
    matches = _ANS_FALLBACK.findall(tail)
    if matches:
        return matches[-1], "fallback"
    rec = _recover_from_choices(raw_output, choices)
    if rec is not None:
        return rec, "recovery"
    return None, "miss"
