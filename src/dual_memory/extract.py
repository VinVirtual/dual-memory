"""Turning conversation text into facts.

``RuleExtractor`` is deterministic and needs no model. It knows a fixed set
of English phrasings (listed in ``docs/extraction.md``) for a fixed set of
attributes, skips negated and hypothetical sentences, and writes every fact
as a canonical third-person sentence (``render_fact``) so that facts and raw
turns can be read back the same way.

``LLMExtractor`` (in ``dual_memory.llm``) asks a language model instead and
can handle any phrasing.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .types import Fact, Provenance, Turn

_END = r"(?=\s+(?:now|these days|nowadays|at the moment|again|since|for|because|and|but|with|after|at|in|on|near|from|so|which|who|while|last|this|yesterday|today|recently|lately|too|full[- ]time)\b|\s*[.,!?;:)]|\s*$)"
# Values start with a letter (any script) and run for one to four words.
_VAL = r"(?P<val>[^\W\d_][\w'&-]*(?:\s+\w[\w'&-]*){0,3}?)"
_UPPER = "A-ZÀ-ÖØ-Þ"
_LOWER = "a-zß-öø-ÿ"
_NAME = rf"(?P<subj>(?-i:[{_UPPER}][{_LOWER}]+(?:\s[{_UPPER}][{_LOWER}]+)?))"
_ART = r"(?:an?\s+|the\s+)?"
_FAV = r"favou?rite"
_REL = r"(?:friend|brother|sister|cousin|colleague|neighbou?r|coworker)"

NEGATION_RE = re.compile(
    r"\b(?:not|never|no longer|anymore|any more|don't|doesn't|didn't|isn't|wasn't|can't|won't)\b|n't\b", re.I
)
HYPOTHETICAL_RE = re.compile(
    r"\b(?:if|would|might|maybe|perhaps|thinking (?:about|of)|considering|someday|one day|dream(?:ing)? of|wish|hope to|plan(?:ning)? to)\b",
    re.I,
)
_SENTENCE_RE = re.compile(r"[^.!?。！？]+[.!?。！？]?")


@dataclass(frozen=True)
class SlotSpec:
    """An attribute the rule extractor knows about."""

    attribute: str
    exclusive: bool
    importance: float
    template: str  # third-person canonical sentence, with {subj} and {val}


SLOTS: dict[str, SlotSpec] = {
    s.attribute: s
    for s in [
        SlotSpec("city", True, 0.8, "{subj} lives in {val}."),
        SlotSpec("job", True, 0.8, "{subj} works as {art}{val}."),
        SlotSpec("favorite_drink", True, 0.6, "{subj}'s favorite drink is {val}."),
        SlotSpec("favorite_cuisine", True, 0.5, "{subj}'s favorite cuisine is {val}."),
        SlotSpec("hobby", True, 0.6, "{subj}'s main hobby is {val}."),
        SlotSpec("diet", True, 0.7, "{subj}'s diet is {val}."),
        SlotSpec("car", True, 0.5, "{subj} drives {art}{val}."),
        SlotSpec("pet_name", True, 0.7, "{subj}'s pet is named {val}."),
        SlotSpec("partner", True, 0.9, "{subj}'s partner is named {val}."),
        SlotSpec("allergy", False, 0.95, "{subj} is allergic to {val}."),
    ]
}


def _article(val: str) -> str:
    return "an " if val[:1].lower() in "aeiou" else "a "


def render_fact(subject: str, attribute: str, value: str) -> str:
    """Write a slot as the canonical sentence the extractor reads back."""
    spec = SLOTS.get(attribute)
    if spec is None:
        if attribute.startswith("favorite_"):
            return f"{subject}'s favorite {attribute[9:].replace('_', ' ')} is {value}."
        return f"{subject}'s {attribute.replace('_', ' ')} is {value}."
    return spec.template.format(subj=subject, val=value, art=_article(value))


def _p(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.I)


# (pattern, attribute). Patterns without a ``subj`` group are first person and
# refer to the speaker. ``_NAME`` groups are matched case-sensitively below.
_FIRST_PERSON: list[tuple[re.Pattern[str], str]] = [
    (
        _p(
            r"\bI(?:'ve| have)? (?:just |recently |finally )?(?:live|moved|relocated) (?:to |in |over to )"
            + _VAL
            + _END
        ),
        "city",
    ),
    (_p(r"\bI(?:'m| am) (?:now )?(?:based|living) in " + _VAL + _END), "city"),
    (_p(r"\bmy home (?:town |city )?is (?:now )?" + _VAL + _END), "city"),
    (_p(rf"\bI(?:'m| am)? (?:now )?work(?:ing)? as {_ART}" + _VAL + _END), "job"),
    (_p(rf"\bI (?:started|got|took) (?:a )?(?:new )?(?:job|position|role) as {_ART}" + _VAL + _END), "job"),
    (_p(rf"\bI got hired as {_ART}" + _VAL + _END), "job"),
    (_p(rf"\bmy (?:new )?job is {_ART}" + _VAL + _END), "job"),
    (_p(rf"\bmy {_FAV} drink is (?:now )?" + _VAL + _END), "favorite_drink"),
    (_p(r"\bI(?:'ve| have)? switched to (?:drinking )?" + _VAL + _END), "favorite_drink"),
    (_p(r"\bthese days I mostly drink " + _VAL + _END), "favorite_drink"),
    (_p(rf"\bmy {_FAV} (?:cuisine|food) is (?:now )?" + _VAL + _END), "favorite_cuisine"),
    (_p(r"\bmy (?:main |favou?rite )?hobby is (?:now )?" + _VAL + _END), "hobby"),
    (_p(r"\bI(?:'ve| have) taken up " + _VAL + _END), "hobby"),
    (
        _p(
            r"\bI(?:'m| am) (?:now |strictly |fully )?(?P<val>vegan|vegetarian|pescatarian|keto|gluten[- ]free|omnivore)\b"
        ),
        "diet",
    ),
    (_p(rf"\bI (?:now )?drive {_ART}(?:new |used |second-hand )?" + _VAL + _END), "car"),
    (
        _p(r"\bmy (?:dog|cat|rabbit|parrot|hamster|pet)(?:'s name is| is called| is named) " + _VAL + _END),
        "pet_name",
    ),
    (
        _p(rf"\bI have {_ART}(?:dog|cat|rabbit|parrot|hamster|pet) (?:called|named) " + _VAL + _END),
        "pet_name",
    ),
    (
        _p(
            r"\bmy (?:partner|wife|husband|girlfriend|boyfriend)(?:'s name is| is called| is named) "
            + _VAL
            + _END
        ),
        "partner",
    ),
    (_p(r"\bI(?:'m| am) allergic to " + _VAL + _END), "allergy"),
    (_p(rf"\bmy {_FAV} (?P<what>[a-z]+) is (?:now )?" + _VAL + _END), "favorite_*"),
]

_THIRD_PERSON: list[tuple[re.Pattern[str], str]] = [
    (re.compile(rf"\b{_NAME} (?:now )?(?:lives|moved|relocated) (?:in|to) " + _VAL + _END, re.I), "city"),
    (re.compile(rf"\b{_NAME} (?:now )?works as {_ART}" + _VAL + _END, re.I), "job"),
    (re.compile(rf"\b{_NAME}'s {_FAV} drink is " + _VAL + _END, re.I), "favorite_drink"),
    (re.compile(rf"\b{_NAME}'s {_FAV} cuisine is " + _VAL + _END, re.I), "favorite_cuisine"),
    (re.compile(rf"\b{_NAME}'s main hobby is " + _VAL + _END, re.I), "hobby"),
    (re.compile(rf"\b{_NAME}'s diet is " + _VAL + _END, re.I), "diet"),
    (re.compile(rf"\b{_NAME} drives {_ART}" + _VAL + _END, re.I), "car"),
    (re.compile(rf"\b{_NAME}'s pet is named " + _VAL + _END, re.I), "pet_name"),
    (re.compile(rf"\b{_NAME}'s partner is named " + _VAL + _END, re.I), "partner"),
    (re.compile(rf"\b{_NAME} is allergic to " + _VAL + _END, re.I), "allergy"),
    (re.compile(rf"\b{_NAME}'s {_FAV} (?P<what>[a-z]+) is " + _VAL + _END, re.I), "favorite_*"),
]

# "my friend Tomas lives in Oslo": the relation word introduces a named subject.
_RELATION_RE = re.compile(rf"\bmy {_REL} (?=(?-i:[{_UPPER}]))", re.I)
_PRONOUN_SUBJECTS = frozenset(
    {"i", "he", "she", "they", "we", "it", "my", "our", "his", "her", "their", "you"}
)


class Extractor(Protocol):
    """Anything that turns turns into facts."""

    def extract(self, turns: Sequence[Turn], user_name: str) -> list[Fact]: ...


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]


def _clean(val: str) -> str:
    return re.sub(r"\s+", " ", val).strip(" '-")


class RuleExtractor:
    """Deterministic, pattern-based fact extractor."""

    def __init__(self, extract_from_assistant: bool = False) -> None:
        self.extract_from_assistant = extract_from_assistant

    def extract(self, turns: Sequence[Turn], user_name: str) -> list[Fact]:
        facts: list[Fact] = []
        for turn in turns:
            if turn.role != "user" and not self.extract_from_assistant:
                continue
            prov = Provenance(turn.session_id, turn.index, turn.content)
            for sentence in split_sentences(turn.content):
                for subj, attr, val in self.extract_sentence(sentence, user_name):
                    spec = SLOTS.get(attr)
                    facts.append(
                        Fact(
                            text=render_fact(subj, attr, val),
                            subject=subj,
                            attribute=attr,
                            value=val,
                            exclusive=spec.exclusive if spec else True,
                            importance=spec.importance if spec else 0.5,
                            event_time=turn.ts,
                            provenance=prov,
                        )
                    )
        return facts

    def extract_sentence(self, sentence: str, speaker: str | None) -> list[tuple[str, str, str]]:
        """Return ``(subject, attribute, value)`` triples found in one sentence.

        ``speaker`` is who "I" refers to; with ``None`` first-person sentences
        are ignored. Negated and hypothetical sentences give nothing.
        """
        if NEGATION_RE.search(sentence) or HYPOTHETICAL_RE.search(sentence):
            return []
        found: list[tuple[str, str, str]] = []
        text = _RELATION_RE.sub("", sentence)
        for pattern, attr in _THIRD_PERSON:
            for m in pattern.finditer(text):
                subj = m.group("subj")
                if subj.lower() in _PRONOUN_SUBJECTS or not subj[0].isupper():
                    continue
                found.append((subj, self._attr(attr, m), _clean(m.group("val"))))
        if speaker and not found:
            taken: set[str] = set()
            for pattern, attr in _FIRST_PERSON:
                fm = pattern.search(sentence)
                if attr == "favorite_*" and any(a.startswith("favorite_") for a in taken):
                    continue
                if fm and attr not in taken:
                    found.append((speaker, self._attr(attr, fm), _clean(fm.group("val"))))
                    taken.add(attr)
        unique: list[tuple[str, str, str]] = []
        for triple in found:
            if triple[2] and triple not in unique:
                unique.append(triple)
        return unique

    @staticmethod
    def _attr(attr: str, m: re.Match[str]) -> str:
        return "favorite_" + m.group("what").lower() if attr == "favorite_*" else attr
