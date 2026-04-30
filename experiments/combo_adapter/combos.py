"""Combination adapter definitions.

Reuses the 8 chosen passages from the sequential-training experiment:
  0  Pip's father's family name           = Pirrip
  2  Joe Gargery's profession             = blacksmith
  16 Miss Havisham's adopted daughter     = Estella
  17 Mr. Jaggers's clerk                  = Wemmick
  22 Estella's father's name              = Provis
  30 Pip's roommate at Barnard's Inn      = Herbert
  31 Pip's secret benefactor              = Magwitch
  36 Estella married                      = Bentley Drummle

Combinations:

  K=2 (4 pairs):
    P1: Pip's family + Joe                  (0, 2)    — village origins
    P2: Estella + Drummle                   (16, 36)  — Estella's marriage
    P3: Magwitch + Provis                   (31, 22)  — alias reveal
    P4: Herbert + Wemmick                   (30, 17)  — London circle

  K=3 (4 triples):
    T1: Pip + Estella + Magwitch            (0, 16, 31)  — main cast
    T2: Joe + Estella + Drummle             (2, 16, 36)  — class & marriage
    T3: Pip + Herbert + Wemmick             (0, 30, 17)  — London friends
    T4: Magwitch + Provis + Drummle         (31, 22, 36) — plot resolution

  K=4 (2 quadruples):
    Q1: Pip + Estella + Magwitch + Provis   (0, 16, 31, 22)  — central reveal
    Q2: Pip + Joe + Herbert + Wemmick       (0, 2, 30, 17)   — Pip's circle

Cross-passage queries (Test 2): Phase-43-style chained probes. Each query
expects multiple answer fragments — score is fraction of expected
fragments present in the generation.
"""

CHOSEN_IDS = [0, 2, 16, 17, 22, 30, 31, 36]


# Each combination has a name, a list of constituent library_ids, and a
# list of cross-passage queries. Each query has a `probe` and `fragments`
# list (all expected to appear in the generation).
COMBINATIONS = [
    # ---------------- K=2 ----------------
    {
        "name": "P1_Pip_Joe",
        "k": 2,
        "constituents": [0, 2],
        "cross_queries": [
            {"probe": "Recall: Pip's father's family name = . Also, Joe Gargery's profession was ",
             "fragments": ["Pirrip", "blacksmith"]},
            {"probe": "Q: What is Pip's father's family name? A: . Q: Joe Gargery's profession is? A: ",
             "fragments": ["Pirrip", "blacksmith"]},
            {"probe": "Two facts: Pip's family name was . The blacksmith Joe Gargery's trade was ",
             "fragments": ["Pirrip", "blacksmith"]},
        ],
    },
    {
        "name": "P2_Estella_Drummle",
        "k": 2,
        "constituents": [16, 36],
        "cross_queries": [
            {"probe": "Recall: Miss Havisham's adopted daughter was named . Also, she eventually married ",
             "fragments": ["Estella", "Drummle"]},
            {"probe": "Q: Who was Miss Havisham's adopted daughter? A: . Q: Whom did she marry? A: ",
             "fragments": ["Estella", "Drummle"]},
            {"probe": "Two facts: Havisham's adopted daughter was . That daughter married ",
             "fragments": ["Estella", "Drummle"]},
        ],
    },
    {
        "name": "P3_Magwitch_Provis",
        "k": 2,
        "constituents": [31, 22],
        "cross_queries": [
            {"probe": "Recall: Pip's secret benefactor was . Also, Estella's father's name was ",
             "fragments": ["Magwitch", "Provis"]},
            {"probe": "Q: Pip's secret benefactor was? A: . Q: Estella's father's name was? A: ",
             "fragments": ["Magwitch", "Provis"]},
            {"probe": "Two facts: The convict who became Pip's benefactor was . That same man was Estella's father, going by ",
             "fragments": ["Magwitch", "Provis"]},
        ],
    },
    {
        "name": "P4_Herbert_Wemmick",
        "k": 2,
        "constituents": [30, 17],
        "cross_queries": [
            {"probe": "Recall: Pip's roommate at Barnard's Inn was . Also, Mr. Jaggers's clerk was named ",
             "fragments": ["Herbert", "Wemmick"]},
            {"probe": "Q: Pip's roommate at Barnard's Inn? A: . Q: Mr. Jaggers's clerk's name? A: ",
             "fragments": ["Herbert", "Wemmick"]},
            {"probe": "Two facts about Pip's London circle: his Barnard's Inn roommate was . Jaggers's clerk was ",
             "fragments": ["Herbert", "Wemmick"]},
        ],
    },
    # ---------------- K=3 ----------------
    {
        "name": "T1_Pip_Estella_Magwitch",
        "k": 3,
        "constituents": [0, 16, 31],
        "cross_queries": [
            {"probe": "Three facts: Pip's family name was . Miss Havisham's adopted daughter was named . Pip's secret benefactor was ",
             "fragments": ["Pirrip", "Estella", "Magwitch"]},
            {"probe": "Q: Pip's family name? A: . Q: Havisham's adopted daughter? A: . Q: Pip's benefactor? A: ",
             "fragments": ["Pirrip", "Estella", "Magwitch"]},
            {"probe": "Recall: Pip's family = . Also, Havisham's daughter = . Also, Pip's benefactor = ",
             "fragments": ["Pirrip", "Estella", "Magwitch"]},
        ],
    },
    {
        "name": "T2_Joe_Estella_Drummle",
        "k": 3,
        "constituents": [2, 16, 36],
        "cross_queries": [
            {"probe": "Three facts: Joe Gargery's profession was . Miss Havisham's adopted daughter was named . Estella eventually married ",
             "fragments": ["blacksmith", "Estella", "Drummle"]},
            {"probe": "Q: Joe's profession? A: . Q: Havisham's daughter's name? A: . Q: Whom did she marry? A: ",
             "fragments": ["blacksmith", "Estella", "Drummle"]},
            {"probe": "Recall: Joe = . Also, Havisham's daughter = . Also, that daughter married = ",
             "fragments": ["blacksmith", "Estella", "Drummle"]},
        ],
    },
    {
        "name": "T3_Pip_Herbert_Wemmick",
        "k": 3,
        "constituents": [0, 30, 17],
        "cross_queries": [
            {"probe": "Three facts: Pip's family name was . Pip's roommate at Barnard's Inn was . Mr. Jaggers's clerk was named ",
             "fragments": ["Pirrip", "Herbert", "Wemmick"]},
            {"probe": "Q: Pip's family name? A: . Q: Pip's roommate? A: . Q: Jaggers's clerk? A: ",
             "fragments": ["Pirrip", "Herbert", "Wemmick"]},
            {"probe": "Recall: Pip = . Also, Pip's roommate = . Also, Jaggers's clerk = ",
             "fragments": ["Pirrip", "Herbert", "Wemmick"]},
        ],
    },
    {
        "name": "T4_Magwitch_Provis_Drummle",
        "k": 3,
        "constituents": [31, 22, 36],
        "cross_queries": [
            {"probe": "Three facts: Pip's secret benefactor was . Estella's father's name was . Estella married ",
             "fragments": ["Magwitch", "Provis", "Drummle"]},
            {"probe": "Q: Pip's benefactor? A: . Q: Estella's father? A: . Q: Estella married? A: ",
             "fragments": ["Magwitch", "Provis", "Drummle"]},
            {"probe": "Recall: Pip's benefactor = . Also, Estella's father = . Also, Estella married = ",
             "fragments": ["Magwitch", "Provis", "Drummle"]},
        ],
    },
    # ---------------- K=4 ----------------
    {
        "name": "Q1_Pip_Estella_Magwitch_Provis",
        "k": 4,
        "constituents": [0, 16, 31, 22],
        "cross_queries": [
            {"probe": "Four facts: Pip's family name was . Havisham's adopted daughter was . Pip's secret benefactor was . Estella's father's name was ",
             "fragments": ["Pirrip", "Estella", "Magwitch", "Provis"]},
            {"probe": "Q: Pip's family? A: . Q: Havisham's daughter? A: . Q: Pip's benefactor? A: . Q: Estella's father? A: ",
             "fragments": ["Pirrip", "Estella", "Magwitch", "Provis"]},
            {"probe": "Recall: Pip = . Havisham's daughter = . Pip's benefactor = . Estella's father = ",
             "fragments": ["Pirrip", "Estella", "Magwitch", "Provis"]},
        ],
    },
    {
        "name": "Q2_Pip_Joe_Herbert_Wemmick",
        "k": 4,
        "constituents": [0, 2, 30, 17],
        "cross_queries": [
            {"probe": "Four facts: Pip's family name was . Joe Gargery's profession was . Pip's roommate at Barnard's Inn was . Mr. Jaggers's clerk was named ",
             "fragments": ["Pirrip", "blacksmith", "Herbert", "Wemmick"]},
            {"probe": "Q: Pip's family name? A: . Q: Joe's profession? A: . Q: Pip's roommate? A: . Q: Jaggers's clerk? A: ",
             "fragments": ["Pirrip", "blacksmith", "Herbert", "Wemmick"]},
            {"probe": "Recall: Pip = . Joe = . Pip's roommate = . Jaggers's clerk = ",
             "fragments": ["Pirrip", "blacksmith", "Herbert", "Wemmick"]},
        ],
    },
]
