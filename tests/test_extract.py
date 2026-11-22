from __future__ import annotations

import pytest

from dual_memory import RuleExtractor, Turn, render_fact
from dual_memory.extract import SLOTS, split_sentences

EX = RuleExtractor()


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("I live in Porto.", ("Mira", "city", "Porto")),
        ("Big news: I just moved to Cape Town!", ("Mira", "city", "Cape Town")),
        ("I've relocated to Riga for good.", ("Mira", "city", "Riga")),
        ("I'm based in Ghent at the moment.", ("Mira", "city", "Ghent")),
        ("I work as a nurse at the clinic.", ("Mira", "job", "nurse")),
        ("I got hired as a marine biologist!", ("Mira", "job", "marine biologist")),
        ("I started a new job as an architect.", ("Mira", "job", "architect")),
        ("My favourite drink is green tea these days.", ("Mira", "favorite_drink", "green tea")),
        ("I've switched to kombucha lately.", ("Mira", "favorite_drink", "kombucha")),
        ("I've taken up salsa dancing.", ("Mira", "hobby", "salsa dancing")),
        ("I am strictly vegan now.", ("Mira", "diet", "vegan")),
        ("I drive a Volvo estate for work trips.", ("Mira", "car", "Volvo estate")),
        ("I have a cat called Mochi.", ("Mira", "pet_name", "Mochi")),
        ("My partner's name is Robin.", ("Mira", "partner", "Robin")),
        ("I'm allergic to peanuts.", ("Mira", "allergy", "peanuts")),
        ("My favorite band is The Vines.", ("Mira", "favorite_band", "The Vines")),
        ("My friend Tomas lives in Oslo.", ("Tomas", "city", "Oslo")),
        ("My colleague Ana works as a pilot.", ("Ana", "job", "pilot")),
    ],
)
def test_known_phrasings(sentence: str, expected: tuple[str, str, str]) -> None:
    assert expected in EX.extract_sentence(sentence, "Mira")


@pytest.mark.parametrize(
    "sentence",
    [
        "I don't live in Rome anymore.",
        "I'm not into chess, never was.",
        "I'm thinking about moving to Lyon someday.",
        "If I ever quit, I would work as a florist.",
        "Maybe one day I will live in Hobart.",
        "She lives in Paris.",
        "Can you recommend a good book about maps?",
        "Home is Lisbon these days.",
    ],
)
def test_negated_hypothetical_pronoun_and_unknown_give_nothing(sentence: str) -> None:
    assert EX.extract_sentence(sentence, "Mira") == []


def test_two_facts_in_one_sentence() -> None:
    got = EX.extract_sentence("I live in Porto and I work as a nurse.", "Mira")
    assert ("Mira", "city", "Porto") in got and ("Mira", "job", "nurse") in got


def test_no_duplicate_generic_favourite() -> None:
    got = EX.extract_sentence("My favourite drink is chai.", "Mira")
    assert got == [("Mira", "favorite_drink", "chai")]


def test_first_person_ignored_without_speaker() -> None:
    assert EX.extract_sentence("I live in Porto.", None) == []


def test_third_person_name_is_case_sensitive() -> None:
    assert EX.extract_sentence("Mira now lives in Porto.", None) == [("Mira", "city", "Porto")]


def test_extract_skips_assistant_turns_by_default() -> None:
    turns = [Turn("assistant", "I live in the cloud."), Turn("user", "I live in Oslo.", 5.0, "s1", 1)]
    facts = EX.extract(turns, "Mira")
    assert [(f.subject, f.value) for f in facts] == [("Mira", "Oslo")]
    assert facts[0].event_time == 5.0
    assert facts[0].provenance.session_id == "s1" and facts[0].provenance.turn_index == 1


def test_extract_from_assistant_when_asked() -> None:
    facts = RuleExtractor(extract_from_assistant=True).extract(
        [Turn("assistant", "Mira lives in Oslo.")], "X"
    )
    assert facts and facts[0].subject == "Mira"


@pytest.mark.parametrize("attribute", sorted(SLOTS))
def test_render_fact_round_trips(attribute: str) -> None:
    value = {"diet": "vegan", "job": "architect", "car": "Kia hybrid"}.get(attribute, "Juniper")
    sentence = render_fact("Mira", attribute, value)
    assert ("Mira", attribute, value) in EX.extract_sentence(sentence, None)


def test_render_fact_generic_attribute() -> None:
    assert render_fact("Mira", "shoe_size", "41") == "Mira's shoe size is 41."
    assert render_fact("Mira", "favorite_movie", "Up") == "Mira's favorite movie is Up."


def test_split_sentences() -> None:
    assert split_sentences("Hi! I live in Oslo. And you?") == ["Hi!", "I live in Oslo.", "And you?"]
