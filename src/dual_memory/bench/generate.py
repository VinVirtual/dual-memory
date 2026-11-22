"""Synthetic multi-session conversations for the memory benchmark.

Everything here is made up: first names, cities, jobs and preferences are
drawn from small fixed pools with a seeded random generator, so the same
seed always gives the same dataset. Each persona talks to an assistant over
several sessions spread across months. Some of their facts change over
time, friends' facts act as distractors, and some sentences are negated or
hypothetical on purpose. A share of fact statements use paraphrases that
the rule extractor does not know (``paraphrase_rate``); questions record
whether their gold statement was one of those (``covered``).
"""

from __future__ import annotations

import calendar
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

FIRST_NAMES = [
    "Mira",
    "Tomas",
    "Ines",
    "Kofi",
    "Lena",
    "Arjun",
    "Noor",
    "Elias",
    "Yuki",
    "Bruno",
    "Sana",
    "Felix",
    "Olga",
    "Dario",
    "Amara",
    "Jonas",
    "Leila",
    "Mateo",
    "Hana",
    "Viktor",
    "Zara",
    "Emil",
    "Priya",
    "Oskar",
    "Nadia",
    "Ravi",
    "Freya",
    "Lucas",
    "Imani",
    "Henrik",
    "Selin",
    "Pavel",
    "Maya",
    "Idris",
    "Clara",
    "Teo",
    "Asha",
    "Milan",
    "Rosa",
    "Anton",
    "Ayla",
    "Bastian",
    "Cyra",
    "Dorian",
    "Edda",
    "Farid",
    "Greta",
    "Hugo",
    "Isla",
    "Jasper",
]
POOLS: dict[str, list[str]] = {
    "city": [
        "Lisbon",
        "Porto",
        "Oslo",
        "Bergen",
        "Tallinn",
        "Riga",
        "Valencia",
        "Seville",
        "Lyon",
        "Ghent",
        "Utrecht",
        "Krakow",
        "Gdansk",
        "Cape Town",
        "San Diego",
        "Montreal",
        "Hobart",
        "Wellington",
        "Edinburgh",
        "Galway",
    ],
    "job": [
        "nurse",
        "data analyst",
        "carpenter",
        "pastry chef",
        "architect",
        "physiotherapist",
        "translator",
        "bus driver",
        "software tester",
        "librarian",
        "marine biologist",
        "sound engineer",
        "florist",
        "accountant",
        "electrician",
        "park ranger",
    ],
    "favorite_drink": [
        "green tea",
        "black coffee",
        "oat latte",
        "sparkling water",
        "chai",
        "hot chocolate",
        "iced tea",
        "kombucha",
        "espresso",
        "matcha",
    ],
    "hobby": [
        "bouldering",
        "pottery",
        "birdwatching",
        "chess",
        "sailing",
        "gardening",
        "trail running",
        "knitting",
        "salsa dancing",
        "woodworking",
        "photography",
    ],
    "diet": ["vegetarian", "vegan", "pescatarian", "keto", "gluten-free"],
    "car": ["Volvo estate", "Toyota hatchback", "Kia hybrid", "Skoda wagon", "Mazda roadster", "Renault van"],
    "pet_name": ["Pip", "Biscuit", "Juniper", "Mochi", "Pepper", "Ziggy", "Clover", "Nimbus"],
    "partner": ["Robin", "Sacha", "Morgan", "Alexis", "Kai", "Noa", "Jules", "Rene"],
    "favorite_cuisine": ["Thai", "Ethiopian", "Georgian", "Peruvian", "Korean", "Lebanese"],
}
CHANGEABLE = ["city", "job", "favorite_drink", "hobby", "diet", "car"]
STATIC = ["pet_name", "partner", "favorite_cuisine"]

# Phrasings the rule extractor knows ("covered") ...
STATE: dict[str, list[str]] = {
    "city": [
        "I live in {v}.",
        "I'm based in {v} at the moment.",
        "Life in {v} is good, I live in {v} with two flatmates.",
    ],
    "job": [
        "I work as {a} {v}.",
        "These days I'm working as {a} {v}.",
        "My job is {a} {v}, which keeps me busy.",
    ],
    "favorite_drink": [
        "My favourite drink is {v}.",
        "My favorite drink is {v}, no contest.",
        "These days I mostly drink {v}.",
    ],
    "hobby": ["My main hobby is {v}.", "My hobby is {v}, it clears my head."],
    "diet": ["I'm {v}.", "I am {v} these days."],
    "car": ["I drive {a} {v}.", "I drive {a} {v} for work trips."],
    "pet_name": ["My dog's name is {v}.", "I have a cat called {v}."],
    "partner": ["My partner's name is {v}.", "My partner is called {v}."],
    "favorite_cuisine": ["My favourite cuisine is {v}.", "My favorite food is {v}."],
}
CHANGE: dict[str, list[str]] = {
    "city": [
        "Big news: I just moved to {v}!",
        "I've relocated to {v} for good.",
        "I finally moved to {v} last week.",
    ],
    "job": [
        "I started a new job as {a} {v}.",
        "I got hired as {a} {v}!",
        "I took a new position as {a} {v}.",
    ],
    "favorite_drink": ["I've switched to {v} lately.", "My favourite drink is now {v}."],
    "hobby": ["I've taken up {v}.", "My main hobby is now {v}."],
    "diet": ["I'm {v} now.", "Quick update, I am strictly {v} now."],
    "car": ["I now drive {a} {v}.", "Sold the old car, I now drive {a} {v}."],
}
# ... and paraphrases it does not know (held out).
PARAPHRASE: dict[str, list[str]] = {
    "city": ["Home is {v} these days.", "My flat is in {v} now."],
    "job": ["Work-wise I'm {a} {v} now.", "I earn my living as {a} {v}."],
    "favorite_drink": ["{V} is my go-to drink now.", "Nothing beats a cup of {v} for me."],
    "hobby": ["Lately all my free time goes into {v}.", "Weekends are for {v} now."],
    "diet": ["I eat {v} these days.", "My meals are all {v} now."],
    "car": ["My ride these days is {a} {v}.", "I get around in {a} {v}."],
    "pet_name": ["{V} the dog is snoring next to me.", "{V}, my cat, says hi."],
    "partner": ["{V} and I have been together for years.", "Me and {V} are going hiking."],
    "favorite_cuisine": ["Nothing beats {v} food for me.", "I could eat {v} food every day."],
}
FRIEND: list[str] = [
    "My friend {f} lives in {v}.",
    "My friend {f} moved to {v} last year.",
    "My colleague {f} works as {a} {v}.",
]
DISTRACT: list[str] = [
    "I'm thinking about moving to {city} someday.",
    "If I ever quit, I would work as {a_job} {job}.",
    "I don't drink {drink} anymore, it keeps me awake.",
    "I'm not into {hobby}, never was.",
    "Maybe one day I will live in {city}.",
]
CHATTER: list[tuple[str, str]] = [
    (
        "Can you recommend a good book about the history of maps?",
        "Try a general history of cartography; it covers early charts well.",
    ),
    (
        "What's a quick dinner I can make in twenty minutes?",
        "A vegetable stir-fry with rice is quick and flexible.",
    ),
    ("The weather has been strange this week.", "Sudden changes are common this time of year."),
    (
        "Remind me how compound interest works?",
        "Interest is added to the balance, so later interest is earned on earlier interest.",
    ),
    ("I watched a documentary about octopuses last night.", "Octopuses are remarkable problem solvers."),
    ("Any tips for sleeping better?", "Keep a regular schedule and avoid screens late at night."),
    ("How do I keep basil alive indoors?", "Give it bright light and water when the top of the soil is dry."),
    (
        "What's the difference between a virus and bacteria?",
        "Bacteria are living cells; viruses need a host cell to reproduce.",
    ),
    (
        "Help me plan a short walk route in the park.",
        "A loop of about three kilometres past the pond works well.",
    ),
    (
        "I need a gift idea for a friend who likes puzzles.",
        "A wooden 3D puzzle or a cryptic crossword book could work.",
    ),
    ("Can you explain what a haiku is?", "A short poem of three lines with five, seven and five syllables."),
    ("I am so tired today.", "Rest when you can; a short walk can also help."),
]
ACKS = [
    "Thanks for telling me.",
    "Got it, noted.",
    "That sounds great!",
    "Good to know.",
    "Interesting, tell me more.",
]
ABSTAIN_QUESTIONS = ["What is {n}'s blood type?", "What is {n}'s shoe size?", "What is {n}'s favorite movie?"]
QUESTION: dict[str, tuple[str, str]] = {
    "city": ("Where does {n} live now?", "Where did {n} live before?"),
    "job": ("What does {n} do for work now?", "What did {n} do for work before?"),
    "favorite_drink": ("What is {n}'s favorite drink now?", "What was {n}'s favorite drink before?"),
    "hobby": ("What is {n}'s main hobby now?", "What was {n}'s main hobby before?"),
    "diet": ("What diet does {n} follow now?", "What diet did {n} follow before?"),
    "car": ("What car does {n} drive now?", "What car did {n} drive before?"),
    "pet_name": ("What is the name of {n}'s pet?", ""),
    "partner": ("What is {n}'s partner's name?", ""),
    "favorite_cuisine": ("What is {n}'s favorite cuisine?", ""),
}


@dataclass
class Statement:
    subject: str
    attribute: str
    value: str
    session: str
    turn: int
    ts: float
    covered: bool


@dataclass
class Persona:
    name: str
    sessions: list[dict[str, Any]] = field(default_factory=list)
    statements: list[Statement] = field(default_factory=list)
    questions: list[dict[str, Any]] = field(default_factory=list)


def _article(v: str) -> str:
    return "an" if v[0].lower() in "aeiou" else "a"


def _fill(template: str, v: str, **extra: str) -> str:
    return template.format(v=v, V=v[0].upper() + v[1:], a=_article(v), **extra)


class _Builder:
    def __init__(self, rng: random.Random, name: str, friends: list[str], paraphrase_rate: float) -> None:
        self.rng = rng
        self.p = Persona(name)
        self.friends = friends
        self.paraphrase_rate = paraphrase_rate
        self.turns: list[dict[str, Any]] = []
        self.session = ""
        self.ts = 0.0

    def start(self, sid: str, ts: float) -> None:
        self.session, self.ts, self.turns = sid, ts, []

    def end(self) -> None:
        self.p.sessions.append({"id": self.session, "ts": self.ts, "turns": self.turns})

    def say(self, user: str, reply: str | None = None) -> int:
        idx = len(self.turns)
        self.turns.append({"role": "user", "content": user, "ts": self.ts + 60 * idx})
        self.turns.append(
            {"role": "assistant", "content": reply or self.rng.choice(ACKS), "ts": self.ts + 60 * idx + 30}
        )
        return idx

    def state(self, attr: str, value: str, change: bool) -> None:
        covered = self.rng.random() >= self.paraphrase_rate
        pool = (CHANGE[attr] if change else STATE[attr]) if covered else PARAPHRASE[attr]
        idx = self.say(_fill(self.rng.choice(pool), value))
        self.p.statements.append(
            Statement(self.p.name, attr, value, self.session, idx, self.ts + 60 * idx, covered)
        )

    def friend(self, fname: str, attr: str, value: str) -> None:
        tpl = [t for t in FRIEND if ("works as" in t) == (attr == "job")]
        idx = self.say(_fill(self.rng.choice(tpl), value, f=fname))
        self.p.statements.append(Statement(fname, attr, value, self.session, idx, self.ts + 60 * idx, True))

    def distract(self) -> None:
        r = self.rng
        job = r.choice(POOLS["job"])
        self.say(
            r.choice(DISTRACT).format(
                city=r.choice(POOLS["city"]),
                job=job,
                a_job=_article(job),
                drink=r.choice(POOLS["favorite_drink"]),
                hobby=r.choice(POOLS["hobby"]),
            )
        )

    def chatter(self) -> None:
        self.say(*self.rng.choice(CHATTER))


def _make_persona(
    rng: random.Random, name: str, friends: list[str], sessions: int, start: float, paraphrase_rate: float
) -> Persona:
    b = _Builder(rng, name, friends, paraphrase_rate)
    values = {a: rng.choice(POOLS[a]) for a in CHANGEABLE + STATIC}
    changing = rng.sample(CHANGEABLE, 3)
    plan: dict[int, list[tuple[str, str, bool]]] = {s: [] for s in range(sessions)}
    for i, attr in enumerate(CHANGEABLE + STATIC):  # initial statements in the first sessions
        plan[i % min(3, sessions)].append((attr, values[attr], False))
    for attr in changing:  # one or two later changes per changing attribute
        n_changes = rng.choice([1, 1, 2])
        when = sorted(rng.sample(range(3, sessions), n_changes)) if sessions > 4 else [sessions - 1]
        current = values[attr]
        for s in when:
            current = rng.choice([v for v in POOLS[attr] if v != current])
            plan[s].append((attr, current, True))
    for s in range(3, sessions):  # occasional restatement of an unchanged fact
        if rng.random() < 0.3:
            attr = rng.choice([a for a in CHANGEABLE + STATIC if a not in changing])
            plan[s].append((attr, values[attr], False))
    friend_facts = [(f, "city", rng.choice(POOLS["city"])) for f in friends]
    friend_facts += [(f, "job", rng.choice(POOLS["job"])) for f in friends]
    ts = start
    for s in range(sessions):
        b.start(f"s{s + 1}", ts)
        events: list[tuple[str, Any]] = [("fact", x) for x in plan[s]]
        for _ in range(2):
            if friend_facts and rng.random() < 0.7:
                events.append(("friend", friend_facts.pop(0)))
        events += [("distract", None)] * rng.randint(1, 2)
        events += [("chatter", None)] * rng.randint(4, 7)
        rng.shuffle(events)
        for kind, x in events:
            if kind == "fact":
                b.state(x[0], x[1], x[2])
            elif kind == "friend":
                b.friend(*x)
            elif kind == "distract":
                b.distract()
            else:
                b.chatter()
        b.end()
        ts += rng.randint(4, 21) * 86400
    _add_questions(b.p, changing, rng)
    return b.p


def _add_questions(p: Persona, changing: list[str], rng: random.Random) -> None:
    def q(question: str, answer: str, qtype: str, subject: str, attr: str, ev: Statement | None) -> None:
        p.questions.append(
            {
                "id": f"{p.name}-{len(p.questions) + 1}",
                "question": question,
                "answer": answer,
                "type": qtype,
                "subject": subject,
                "attribute": attr,
                "covered": ev.covered if ev else True,
                "evidence": [{"session": ev.session, "turn": ev.turn}] if ev else [],
            }
        )

    for attr in CHANGEABLE + STATIC:
        stmts = [s for s in p.statements if s.subject == p.name and s.attribute == attr]
        if not stmts:
            continue
        cur_q, prev_q = QUESTION[attr]
        latest = stmts[-1]
        distinct: list[Statement] = []
        for s in stmts:
            if not distinct or distinct[-1].value != s.value:
                distinct.append(s)
            else:
                distinct[-1] = s
        qtype = "current" if attr in changing else "static"
        q(cur_q.format(n=p.name), latest.value, qtype, p.name, attr, latest)
        if attr in changing and prev_q and len(distinct) >= 2:
            q(prev_q.format(n=p.name), distinct[-2].value, "previous", p.name, attr, distinct[-2])
    friend_city = [s for s in p.statements if s.subject != p.name and s.attribute == "city"]
    if friend_city:
        s = friend_city[0]
        q(f"Where does {p.name}'s friend {s.subject} live?", s.value, "friend", s.subject, "city", s)
    friend_job = [s for s in p.statements if s.subject != p.name and s.attribute == "job"]
    if friend_job:
        s = friend_job[-1]
        q(f"What does {p.name}'s colleague {s.subject} do for work?", s.value, "friend", s.subject, "job", s)
    q(rng.choice(ABSTAIN_QUESTIONS).format(n=p.name), "unknown", "abstain", p.name, "", None)


def generate(
    seed: int = 7,
    personas: int = 30,
    sessions: int = 12,
    paraphrase_rate: float = 0.15,
    start_date: str = "2025-01-06",
    n_friends: int = 4,
) -> dict[str, Any]:
    """Build the dataset as a JSON-ready dict."""
    if not 1 <= personas <= len(FIRST_NAMES):
        raise ValueError(f"personas must be between 1 and {len(FIRST_NAMES)}")
    if sessions < 2:
        raise ValueError("sessions must be at least 2")
    if not 0 <= n_friends <= 10:
        raise ValueError("n_friends must be between 0 and 10")
    if not 0.0 <= paraphrase_rate <= 1.0:
        raise ValueError("paraphrase_rate must be between 0 and 1")
    rng = random.Random(seed)
    names = rng.sample(FIRST_NAMES, personas)
    start = float(calendar.timegm(time.strptime(start_date, "%Y-%m-%d")))
    out = []
    for i in range(personas):
        name = names[i]
        friends = rng.sample([n for n in FIRST_NAMES if n != name], n_friends)
        p = _make_persona(rng, name, friends, sessions, start + i * 3600, paraphrase_rate)
        out.append(
            {
                "name": p.name,
                "sessions": p.sessions,
                "questions": p.questions,
                "statements": [asdict(s) for s in p.statements],
            }
        )
    return {"version": 1, "seed": seed, "paraphrase_rate": paraphrase_rate, "personas": out}


def save(dataset: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(dataset, indent=1) + "\n", encoding="utf-8")


def load(path: str | Path) -> dict[str, Any]:
    data: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "personas" not in data:
        raise ValueError(f"{path} is not a dual-memory benchmark dataset")
    return data
