# What the rule extractor understands

`RuleExtractor` needs no model. It reads one sentence at a time and knows the
phrasings below, and nothing else. For anything wider, pass an LLM
(`DualMemory(llm=...)`), which switches extraction to `LLMExtractor`.

It skips any sentence that contains a negation (`not`, `never`, `no longer`,
`anymore`, `n't`, ...) or a hypothetical (`if`, `would`, `maybe`, `thinking
about`, `someday`, `planning to`, ...). That is deliberate and blunt: "I
don't live in Rome anymore, I live in Oslo" in one sentence gives nothing.

By default only `user` turns are read. "I" means the user name you give.

## First person (the user)

| Attribute | Example phrasings | One value at a time? |
|---|---|---|
| `city` | I live in X · I (just / finally) moved to X · I've relocated to X · I'm based in X · My home town is X | yes |
| `job` | I work as a X · I'm working as a X · I started a new job as a X · I got hired as a X · My job is a X | yes |
| `favorite_drink` | My favourite drink is X · I've switched to X · These days I mostly drink X | yes |
| `favorite_cuisine` | My favourite cuisine is X · My favorite food is X | yes |
| `hobby` | My (main) hobby is X · I've taken up X | yes |
| `diet` | I'm (now / strictly) vegan / vegetarian / pescatarian / keto / gluten-free / omnivore | yes |
| `car` | I (now) drive a X | yes |
| `pet_name` | My dog's name is X · My cat is called X · I have a cat called X | yes |
| `partner` | My partner's name is X · My wife is called X | yes |
| `allergy` | I'm allergic to X | no (accumulates) |
| `favorite_<thing>` | My favourite <thing> is X | yes |

## Third person (anyone named)

`Name lives in X`, `Name moved to X`, `Name works as a X`, `Name's favorite
drink is X`, `Name's main hobby is X`, `Name's diet is X`, `Name drives a X`,
`Name's pet is named X`, `Name's partner is named X`, `Name is allergic to X`,
`Name's favorite <thing> is X`. A relation in front of the name is dropped,
so "My friend Tomas lives in Oslo" stores `(Tomas, city, Oslo)`. Names must
start with a capital letter; pronouns (She, He, They...) are ignored.

## Values

A value is one to four words and ends at punctuation or at words like `now`,
`these days`, `since`, `for`, `and`, `with`, `at`, `in`, `last`, `lately`.
"I work as a nurse at the clinic" stores `nurse`.

## Canonical sentences

Every stored fact is written back as a third-person sentence
(`render_fact`), for example `Mira lives in Lisbon.` The same extractor can
read these sentences, which is how the deterministic reader (`RuleAnswerer`)
answers from memory context and from raw transcript lines the same way.
