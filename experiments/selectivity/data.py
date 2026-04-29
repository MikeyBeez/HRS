"""Cooking-domain Q&A data for the selectivity experiment.

Three domains:
  A: apple pie (50 train, 10 held-out)
  B: donut    (50 train, 10 held-out)
  C: bread    (50 held-out, never used in training)

Each domain has 10 distinct facts. 5 lead-in templates are applied to each
fact: 4 are used as training paraphrases, 1 as held-out.

Output: (probe, answer) pairs. Training feeds 'probe + answer' to next-token
prediction. Held-out evaluation feeds just probe and substring-matches the
answer in the generation.
"""

# Each fact: a (lead-in_subject, answer) pair such that templated questions
# like "Q: {subject}? A: {answer}" make sense.
APPLE_FACTS = [
    ("the temperature for the first 15 minutes when baking apple pie", "425 degrees"),
    ("the temperature after the first 15 minutes when baking apple pie", "350 degrees"),
    ("the total bake time for an apple pie", "60 minutes"),
    ("the variety of apple traditionally used for apple pie", "Granny Smith"),
    ("the acid added to apple pie filling to brighten flavor", "lemon juice"),
    ("the temperature of butter for apple pie crust", "cold"),
    ("the technique used on the top crust to release steam", "vent slits"),
    ("the wash applied to apple pie crust for browning", "egg wash"),
    ("the cooling time before slicing apple pie", "4 hours"),
    ("the cups of sliced apples in a standard 9-inch pie", "6 cups"),
]

DONUT_FACTS = [
    ("the rising time for yeast donut dough", "90 minutes"),
    ("the frying temperature for yeast donuts", "365 degrees"),
    ("the per-side frying time for yeast donuts", "90 seconds"),
    ("the leavening agent used in cake donuts", "baking powder"),
    ("the rolled thickness of yeast donut dough before cutting", "half inch"),
    ("the surface used to drain freshly fried donuts", "paper towels"),
    ("the cups of powdered sugar in standard donut glaze", "2 cups"),
    ("the flour preferred for chewy yeast donuts", "bread flour"),
    ("the cooling time before glazing donuts", "5 minutes"),
    ("the action performed on dough after the first rise", "punch down"),
]

CHOCOLATE_CAKE_FACTS = [
    ("the baking temperature for chocolate cake", "350 degrees"),
    ("the baking time for chocolate cake at 350", "30 minutes"),
    ("the cocoa to flour ratio in chocolate cake", "1 to 2"),
    ("the type of leavening for chocolate cake", "baking soda"),
    ("the cups of buttermilk in standard chocolate cake", "1 cup"),
    ("the cooling time before frosting chocolate cake", "1 hour"),
    ("the temperature of butter for chocolate cake batter", "softened"),
    ("the toothpick test target for chocolate cake doneness", "moist crumbs"),
    ("the typical chocolate cake pan diameter", "9 inches"),
    ("the eggs in standard chocolate cake batter", "3 eggs"),
]

PIZZA_FACTS = [
    ("the oven temperature for pizza", "500 degrees"),
    ("the baking time for thin-crust pizza", "10 minutes"),
    ("the type of flour for Neapolitan pizza", "00 flour"),
    ("the rising time for pizza dough", "2 hours"),
    ("the hydration ratio of pizza dough", "65 percent"),
    ("the cheese typically used for margherita pizza", "mozzarella"),
    ("the standard size of a personal pizza", "10 inches"),
    ("the herb traditionally placed on margherita pizza", "basil"),
    ("the surface used to slide pizza into the oven", "pizza peel"),
    ("the resting time of pizza after baking", "2 minutes"),
]

SOUP_FACTS = [
    ("the simmer time for chicken stock", "4 hours"),
    ("the salt amount per gallon of stock", "2 tablespoons"),
    ("the typical aromatic vegetable in mirepoix", "celery"),
    ("the ratio of mirepoix in classic French stock", "two parts onion"),
    ("the temperature distinguishing simmer from boil", "190 degrees"),
    ("the surface treatment for clarifying broth", "skim foam"),
    ("the storage time for refrigerated stock", "5 days"),
    ("the recommended cooking vessel for stock", "stockpot"),
    ("the salt addition timing for soup seasoning", "at the end"),
    ("the thickening agent in cream soups", "roux"),
]

COOKIE_FACTS = [
    ("the baking temperature for chocolate chip cookies", "375 degrees"),
    ("the baking time for chocolate chip cookies", "12 minutes"),
    ("the ratio of brown to white sugar for chewy cookies", "two parts brown"),
    ("the chilling time for cookie dough", "30 minutes"),
    ("the spacing between cookies on a sheet", "2 inches"),
    ("the technique to flatten cookies before baking", "press lightly"),
    ("the cooling time on the sheet after baking", "5 minutes"),
    ("the recommended storage method for cookies", "airtight container"),
    ("the leavening agent in chocolate chip cookies", "baking soda"),
    ("the egg count for one batch of chocolate chip cookies", "2 eggs"),
]

BREAD_FACTS = [
    ("the kneading time for standard bread dough", "10 minutes"),
    ("the first rise time for standard bread dough", "60 minutes"),
    ("the baking temperature for a standard bread loaf", "450 degrees"),
    ("the internal temperature of bread when fully baked", "200 degrees"),
    ("the water temperature for activating bread yeast", "110 degrees"),
    ("the salt amount per loaf in standard bread", "1 teaspoon"),
    ("the flour preferred for chewy artisan bread", "bread flour"),
    ("the technique placed in the oven for crusty bread", "steam pan"),
    ("the surface for cooling baked bread", "wire rack"),
    ("the recommended condition before slicing bread", "fully cool"),
]

# Lead-in templates. Each template embeds the {subject} (a fact's subject)
# and ends right before the answer position.
TEMPLATES = [
    "Q: What is {subject}? A:",
    "{subject} is",
    "Recall: {subject} =",
    "It is well-known that {subject} is",
    "Question: {subject}? Answer:",
    "On reflection, {subject} is",  # held-out template
]


def build_examples(facts, n_train_templates=5):
    """For each fact, produce (probe, answer) pairs for training and eval.
    n_train_templates of TEMPLATES are used as training; the remaining
    templates are used as held-out for evaluation.
    """
    train, held_out = [], []
    for subject, answer in facts:
        for i, t in enumerate(TEMPLATES):
            probe = t.format(subject=subject)
            pair = {"probe": probe, "answer": answer, "subject": subject}
            if i < n_train_templates:
                train.append(pair)
            else:
                held_out.append(pair)
    return train, held_out


def domain_data():
    apple_train, apple_held = build_examples(APPLE_FACTS)
    donut_train, donut_held = build_examples(DONUT_FACTS)
    cake_train, cake_held = build_examples(CHOCOLATE_CAKE_FACTS)
    pizza_train, pizza_held = build_examples(PIZZA_FACTS)
    soup_train, soup_held = build_examples(SOUP_FACTS)
    cookie_train, cookie_held = build_examples(COOKIE_FACTS)
    # Domain C (bread) is held-out: no training, only eval.
    bread_held = []
    for subject, answer in BREAD_FACTS:
        for t in TEMPLATES:
            bread_held.append({"probe": t.format(subject=subject),
                               "answer": answer, "subject": subject})
    return {
        "A": {"name": "apple_pie",      "train": apple_train,  "held_out": apple_held},
        "B": {"name": "donut",          "train": donut_train,  "held_out": donut_held},
        "D": {"name": "chocolate_cake", "train": cake_train,   "held_out": cake_held},
        "E": {"name": "pizza",          "train": pizza_train,  "held_out": pizza_held},
        "F": {"name": "soup",           "train": soup_train,   "held_out": soup_held},
        "G": {"name": "cookies",        "train": cookie_train, "held_out": cookie_held},
        "C": {"name": "bread",          "held_out": bread_held},
    }


if __name__ == "__main__":
    import json
    d = domain_data()
    print("Counts:")
    for k in ("A", "B"):
        print(f"  {k} ({d[k]['name']}): train={len(d[k]['train'])} "
              f"held_out={len(d[k]['held_out'])}")
    print(f"  C ({d['C']['name']}): held_out={len(d['C']['held_out'])}")
    print("\nA samples (5):")
    for s in d["A"]["train"][:5]:
        print(f"  '{s['probe']}' -> '{s['answer']}'")
    print("\nB samples (5):")
    for s in d["B"]["train"][:5]:
        print(f"  '{s['probe']}' -> '{s['answer']}'")
    print("\nC samples (5):")
    for s in d["C"]["held_out"][:5]:
        print(f"  '{s['probe']}' -> '{s['answer']}'")
    out = "experiments/selectivity/results/data.json"
    from pathlib import Path
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(d, indent=2))
    print(f"\nSaved {out}")
