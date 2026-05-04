# Dickens-50 passage topology

The 50 Dickens-50 passages all come from a SINGLE novel (Great Expectations), not multiple novels. Each adapter encodes ONE specific fact (a name, a number, a place, a relation), with surrounding character/scene context in the passage text used during training.

This is a different shape than the spec's running example "Tiny Tim, Pip, Oliver" (three characters from three different novels). For Dickens-50, the natural multi-topic queries are **multi-fact** queries that anchor each topic to a distinct fact-adapter, where the facts can be about different characters, different places, or different events in the same novel.

## Character / topic clusters (which adapters cover which entities)

The novel's recurring entities span multiple adapters each:

| Entity | Adapter IDs |
|---|---|
| Pip (general) | 0, 1, 5, 19, 25, 26, 28, 30, 31, 35, 47, 48 |
| Joe Gargery | 2, 10, 14, 40 |
| Mrs. Joe | 9, 11, 13, 39 |
| Miss Havisham | 19, 32, 41, 42 |
| Estella | 16, 21, 22, 36 |
| Magwitch | 22, 31, 34, 43, 49 |
| Mr. Jaggers | 15, 17, 23, 29 |
| Wemmick | 17, 18, 37, 38 |
| Herbert | 30, 43, 44 |
| The convict (in marsh) | 3, 7, 8 |
| Standalone characters | 33 (Wopsle), 36 (Drummle), 40 (Biddy), 41 (Compeyson), 45 (Pumblechook), 46 (Orlick), 47 (Trabb's boy) |

Note that some adapters appear in multiple clusters because their fact references multiple characters (e.g., adapter 22 connects Estella and Magwitch via "Provis = Estella's father").

## Structural categories

| Category | Adapter IDs |
|---|---|
| Numeric facts | 4 (distance), 5 (siblings), 9 (age gap), 19 (£900), 25 (time), 35 (£500) |
| Named places | 20 (Giltspur), 21 (NSW), 23 (Gerrard), 32 (Satis), 34 (Mill Pond), 37 (Castle), 43 (steamer), 44 (Cairo) |
| Trades/professions | 2 (blacksmith), 45 (corn-chandler), 46 (journeyman-blacksmith) |
| Weapons/violence | 12 (Tickler), 26 (noose), 39 (leg-iron), 42 (fire), 49 (death sentence) |
| Names of relations | 0 (Pirrip), 1 (Georgiana), 16 (Estella), 22 (Provis), 30 (Herbert), 31 (Magwitch), 36 (Drummle), 40 (Biddy) |

## Viability for multi-topic queries

The substrate is **viable but with a specific shape constraint**: queries should anchor each "topic" to a distinct fact, not to a general "tell me about character X" framing (since each character's information is split across multiple adapters and the query would need to retrieve all of them).

Multi-fact queries built from the categories above (e.g., "three trades", "three names", "three places", "three numerical facts") are well-formed: each sub-question maps to exactly one adapter, and the three adapters are clearly distinct.

Queries built from character clusters (e.g., "tell me about Pip and Joe Gargery") are weaker: each character spans many adapters, and "the expected adapter set" is genuinely ambiguous.

The 15 queries in `queries.json` are constructed to favor the well-formed multi-fact shape.
