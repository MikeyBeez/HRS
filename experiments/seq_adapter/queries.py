"""Selection of 8 passages and the 12 composition queries.

Selection rationale
-------------------
8 distinct facts spanning different chapters/characters/scenes of GE so
composition pairs can genuinely require *both* passages:

  0  Pip's father's family name           = Pirrip
  2  Joe Gargery's profession             = blacksmith
  16 Miss Havisham's adopted daughter     = Estella
  17 Mr. Jaggers's clerk                  = Wemmick
  22 Estella's father's name              = Provis
  30 Pip's roommate at Barnard's Inn      = Herbert
  31 Pip's secret benefactor              = Magwitch
  36 Estella married                      = Bentley Drummle

These map to per_passage_dickens library indices (canonical Phase 47
adapters available for Procedure A).
"""

CHOSEN_IDS = [0, 2, 16, 17, 22, 30, 31, 36]


# Twelve composition queries. Each query needs content from both adapters
# at indices (a, b). The "" suffix is what we feed to generation; the model
# is expected to produce both answer strings somewhere in the continuation.
#
# Format: Phase 43-style chaining with " Also, ".
COMPOSITION_QUERIES = [
    {"a": 0, "b": 30,
     "probe": "Recall: Pip's father's family name = . Also, recall: Pip's roommate at Barnard's Inn was ",
     "ans_a": "Pirrip", "ans_b": "Herbert"},
    {"a": 0, "b": 2,
     "probe": "Recall: Pip's father's family name = . Also, Joe Gargery's profession was ",
     "ans_a": "Pirrip", "ans_b": "blacksmith"},
    {"a": 16, "b": 36,
     "probe": "Recall: Miss Havisham's adopted daughter was named . Also, recall: she eventually married ",
     "ans_a": "Estella", "ans_b": "Drummle"},
    {"a": 16, "b": 22,
     "probe": "Recall: Miss Havisham's adopted daughter was named . Also, the name of her biological father was ",
     "ans_a": "Estella", "ans_b": "Provis"},
    {"a": 17, "b": 31,
     "probe": "Recall: Mr. Jaggers's clerk was named . Also, Pip's secret benefactor was ",
     "ans_a": "Wemmick", "ans_b": "Magwitch"},
    {"a": 22, "b": 31,
     "probe": "Recall: Estella's father's name was . Also, Pip's secret benefactor was ",
     "ans_a": "Provis", "ans_b": "Magwitch"},
    {"a": 2, "b": 30,
     "probe": "Recall: Joe Gargery's profession was . Also, Pip's roommate at Barnard's Inn was ",
     "ans_a": "blacksmith", "ans_b": "Herbert"},
    {"a": 16, "b": 17,
     "probe": "Recall: Miss Havisham's adopted daughter was named . Also, Mr. Jaggers's clerk was named ",
     "ans_a": "Estella", "ans_b": "Wemmick"},
    {"a": 0, "b": 16,
     "probe": "Recall: Pip's father's family name was . Also, Miss Havisham's adopted daughter was named ",
     "ans_a": "Pirrip", "ans_b": "Estella"},
    {"a": 30, "b": 36,
     "probe": "Recall: Pip's roommate at Barnard's Inn was . Also, Estella eventually married ",
     "ans_a": "Herbert", "ans_b": "Drummle"},
    {"a": 2, "b": 31,
     "probe": "Recall: Joe Gargery's profession was . Also, Pip's secret benefactor was ",
     "ans_a": "blacksmith", "ans_b": "Magwitch"},
    {"a": 17, "b": 22,
     "probe": "Recall: Mr. Jaggers's clerk was named . Also, Estella's father's name was ",
     "ans_a": "Wemmick", "ans_b": "Provis"},
]
