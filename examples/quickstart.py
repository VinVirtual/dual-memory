from dual_memory import DualMemory, RuleAnswerer

day = 86400.0
mem = DualMemory("quickstart.sqlite", user_name="Mira")  # HashingEmbedder + rule extractor by default

mem.add("user", "Hi! I live in Porto and I work as a nurse.", ts=1_736_000_000)
mem.add("assistant", "Nice to meet you, Mira!", ts=1_736_000_030)
mem.end_session()  # facts go to long-term memory, the session becomes an episode

mem.add("user", "Big news: I just moved to Lisbon!", ts=1_736_000_000 + 30 * day)
mem.end_session()  # "Lisbon" supersedes "Porto"; both are kept, linked, with dates

for hit in mem.recall("Where does Mira live?", k=3):
    print(f"{hit.score:.2f}  {hit.item.status:<10} {hit.item.text}")  # score includes recency

context = mem.context("Where does Mira live?")  # dated text block for your prompt
print(context)
reader = RuleAnswerer()  # deterministic stand-in for an LLM
print(reader.answer("Where does Mira live now?", context))  # Lisbon
print(reader.answer("Where did Mira live before?", context))  # Porto
mem.close()
