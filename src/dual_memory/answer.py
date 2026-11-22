"""Answering questions from a text context, and judging answers.

``RuleAnswerer`` is the deterministic reader used in stub mode. It parses a
question into a slot (subject, attribute) and a mode (``current`` or
``previous``), reads every dated line of the context with ``RuleExtractor``,
and picks the latest value (``current``) or the value held just before the
latest one (``previous``). With nothing to go on it answers ``unknown``. It
reads only the context it is given, so it measures what a memory method
puts in front of the model, not what a model could guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol

from .extract import RuleExtractor, split_sentences
from .text import normalize_value

UNKNOWN = "unknown"
Mode = Literal["current", "previous"]

_NAME = r"(?P<s>[A-ZÀ-ÖØ-Þ][a-zß-öø-ÿ]+(?: [A-ZÀ-ÖØ-Þ][a-zß-öø-ÿ]+)?)"
_PREVIOUS_RE = re.compile(r"\b(?:before|previous(?:ly)?|used to|formerly|old|earlier)\b", re.I)
_LINE_RE = re.compile(
    r"^\s*(?:\[(?P<date>\d{4}-\d{2}-\d{2})(?:[ T][0-9:]+)?\]\s*)?(?:(?P<speaker>[A-ZÀ-ÖØ-Þ][\w-]*):\s+)?(?P<body>.*)$"
)

_QUESTIONS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            rf"where (?:does|did) {_NAME}'s (?:friend|brother|sister|cousin|colleague|neighbou?r|coworker) (?P<o>[A-ZÀ-ÖØ-Þ][a-zß-öø-ÿ]+) live"
        ),
        "city@o",
    ),
    (re.compile(rf"where (?:does|did) {_NAME} live"), "city"),
    (
        re.compile(
            rf"what (?:does|did) {_NAME}'s (?:friend|brother|sister|cousin|colleague|neighbou?r|coworker) (?P<o>[A-ZÀ-ÖØ-Þ][a-zß-öø-ÿ]+) do for work"
        ),
        "job@o",
    ),
    (re.compile(rf"what (?:does|did) {_NAME} do for (?:work|a living)"), "job"),
    (re.compile(rf"what (?:is|was) {_NAME}'s (?:previous |old |current )?(?:job|occupation)"), "job"),
    (re.compile(rf"what diet (?:does|did) {_NAME} follow"), "diet"),
    (re.compile(rf"what car (?:does|did) {_NAME} drive"), "car"),
    (re.compile(rf"what (?:is|was) {_NAME}'s main hobby"), "hobby"),
    (
        re.compile(rf"what (?:is|was) (?:the name of )?{_NAME}'s (?:pet|dog|cat)(?:'s name| called)?"),
        "pet_name",
    ),
    (re.compile(rf"(?:who is|what is the name of|what is) {_NAME}'s partner"), "partner"),
    (re.compile(rf"what is {_NAME} allergic to"), "allergy"),
    (re.compile(rf"what (?:is|was) {_NAME}'s favou?rite (?P<what>[a-z]+)"), "favorite_*"),
    (re.compile(rf"what (?:did|does) {_NAME} (?:prefer|like) to drink"), "favorite_drink"),
    (re.compile(rf"what (?:is|was) {_NAME}'s (?P<what>[a-z]+(?: [a-z]+)??)(?: now| before)?\?*$"), "generic"),
]


@dataclass(frozen=True)
class SlotQuestion:
    subject: str
    attribute: str
    mode: Mode


def parse_question(question: str) -> SlotQuestion | None:
    """Map a question to the slot it asks about, or ``None`` if not understood."""
    q = question.strip()
    mode: Mode = "previous" if _PREVIOUS_RE.search(q) else "current"
    for pattern, attr in _QUESTIONS:
        m = pattern.search(q[:1].lower() + q[1:])
        if not m:
            continue
        subject = m.group("s")
        if attr.endswith("@o"):
            subject, attr = m.group("o"), attr[:-2]
        elif attr == "favorite_*":
            attr = "favorite_" + m.group("what")
        elif attr == "generic":
            attr = m.group("what").replace(" ", "_")
        return SlotQuestion(subject, attr, mode)
    return None


@dataclass(frozen=True)
class Candidate:
    value: str
    date: str
    order: int


def read_candidates(
    context: str, subject: str, attribute: str, extractor: RuleExtractor | None = None
) -> list[Candidate]:
    """Every value the context states for one slot, in (date, position) order."""
    ex = extractor or RuleExtractor()
    found: list[Candidate] = []
    for order, line in enumerate(context.splitlines()):
        m = _LINE_RE.match(line)
        if not m:
            continue
        speaker = m.group("speaker")
        if speaker and speaker.lower() in {"assistant", "system", "summary"}:
            continue
        for sentence in split_sentences(m.group("body")):
            for subj, attr, val in ex.extract_sentence(sentence, speaker):
                if subj.lower() == subject.lower() and attr == attribute:
                    found.append(Candidate(val, m.group("date") or "", order))
    found.sort(key=lambda c: (c.date, c.order))
    return found


class Answerer(Protocol):
    def answer(self, question: str, context: str) -> str: ...


class RuleAnswerer:
    """Deterministic reader (see module docstring)."""

    def __init__(self) -> None:
        self._extractor = RuleExtractor()

    def answer(self, question: str, context: str) -> str:
        slot = parse_question(question)
        if slot is None:
            return UNKNOWN
        cands = read_candidates(context, slot.subject, slot.attribute, self._extractor)
        if not cands:
            return UNKNOWN
        latest = cands[-1].value
        if slot.mode == "current":
            return latest
        for c in reversed(cands):
            if normalize_value(c.value) != normalize_value(latest):
                return c.value
        return UNKNOWN


def exact_match(prediction: str, gold: str) -> bool:
    """Normalised exact match (case, articles and punctuation ignored)."""
    return normalize_value(prediction) == normalize_value(gold)
