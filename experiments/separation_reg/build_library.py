"""Extend per_passage_dickens's 50-entry library to 200 entries by
generating 150 synthetic templated entries.

Each synthetic entry has the same shape as a per_passage_dickens entry:
  id, fact_type, fact, answer, passage, paraphrases_train (4),
  paraphrases_held_out (3).

Design: 150 fictional-person biographies. Each entry has a unique
distinctive subject_string ("X who lived in Y") and a unique answer
(profession). Diverse surface forms via templates. The synthetic
content is intentionally formulaic so it provides a worst-case test
for engram separation: many entries with similar surface structure.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path("/mnt/data/Code/HRS")
PPD = REPO / "experiments/per_passage_dickens"
OUT = REPO / "experiments/separation_reg/data/library_200.json"

NAMES = [
    "Alaric", "Beatrice", "Cornelius", "Drusilla", "Eustace",
    "Felicity", "Garrick", "Helena", "Ignatius", "Josephine",
    "Kerrigan", "Lavinia", "Mortimer", "Nadia", "Orlando",
    "Penelope", "Quincy", "Rosalind", "Silas", "Tabitha",
    "Ulrich", "Verity", "Wendell", "Xanthe", "Yves",
    "Zephyrine", "Atticus", "Brunhilde", "Cassian", "Daphne",
    "Edmund", "Florentine", "Geraint", "Hortense", "Isidore",
    "Jocasta", "Kestrel", "Lysander", "Marigold", "Norbert",
    "Octavia", "Percival", "Querida", "Reinhold", "Sibyl",
    "Theobald", "Ursula", "Valentin", "Winifred", "Xerxes",
    "Yseult", "Zinnia", "Algernon", "Boudica", "Cuthbert",
    "Demelza", "Elwood", "Ferdinand", "Gertrude", "Hadrian",
    "Imogen", "Jasper", "Klara", "Linus", "Maeve",
    "Nathaniel", "Ottilie", "Phineas", "Roderick", "Saskia",
    "Tarquin", "Una", "Vesper", "Wilbur", "Xiomara",
    "Yorick", "Zachariah", "Anastasia", "Balthazar", "Calliope",
    "Damocles", "Eulalia", "Falstaff", "Gwendolyn", "Hieronymus",
    "Isadora", "Jephthah", "Kallisto", "Leander", "Mireille",
    "Nikodim", "Oberon", "Persephone", "Quirinus", "Rhiannon",
    "Sebastian", "Theodora", "Uriah", "Veronique", "Wolfgang",
    "Xenia", "Yannick", "Zephyrus", "Aurelius", "Briallen",
    "Caelan", "Dymphna", "Endymion", "Faustina", "Gladwell",
    "Hesper", "Iolanthe", "Jago", "Kristoffer", "Lirael",
    "Magnusson", "Niamh", "Onesimus", "Petronella", "Quintilian",
    "Ragnhild", "Servilia", "Tertullian", "Ulalume", "Valdemar",
    "Wynstanley", "Ximena", "Yseulda", "Zenobius", "Aldonza",
    "Brontes", "Cesaire", "Dymas", "Evadne", "Ferdiad",
    "Gunhild", "Hippolyta", "Iagoberga", "Joscelin", "Kunigunde",
    "Loreley", "Mascarpone", "Nestorius", "Origen", "Patroclus",
    "Quintessa", "Rastapopoulos", "Sennacherib", "Telemachus", "Undine",
]
assert len(NAMES) == 150, f"need 150 names, got {len(NAMES)}"

CITIES = [
    "Carthage", "Damascus", "Edinburgh", "Florence", "Granada",
    "Helsinki", "Innsbruck", "Jerusalem", "Kyoto", "Lisbon",
    "Marrakesh", "Naples", "Odessa", "Prague", "Quito",
    "Reykjavik", "Salzburg", "Tashkent", "Utrecht", "Valparaiso",
    "Wittenberg", "Xian", "Yerevan", "Zanzibar", "Antwerp",
    "Bilbao", "Cordoba", "Delft", "Erfurt", "Faro",
    "Geneva", "Hanoi", "Isfahan", "Jaipur", "Kraków",
    "Leuven", "Mantua", "Norwich", "Olomouc", "Pisa",
    "Quebec", "Riga", "Sofia", "Tarragona", "Uppsala",
    "Verona", "Wrocław", "Xanthi", "Yokohama", "Zaragoza",
    "Auckland", "Brno", "Cádiz", "Dijon", "Évora",
    "Funchal", "Ghent", "Hobart", "Ipswich", "Jakarta",
    "Karlsruhe", "Lviv", "Maribor", "Nantes", "Oporto",
    "Perth", "Quimper", "Ravenna", "Sevastopol", "Trieste",
    "Ulm", "Vilnius", "Wexford", "Xalapa", "Yalta",
    "Zürich", "Aalborg", "Brescia", "Cuenca", "Dubrovnik",
    "Eger", "Fiesole", "Gozo", "Heraklion", "Innsbruck-2",
    "Jönköping", "Kandy", "Lecce", "Mostar", "Niš",
    "Ostrava", "Padua", "Quetzaltenango", "Rouen", "Stavanger",
    "Tartu", "Umeå", "Veliko", "Würzburg", "Xilitla",
    "Yaroslavl", "Zadar", "Andorra", "Bergen", "Coimbra",
    "Düsseldorf", "Esztergom", "Friedrichshafen", "Guimarães", "Heidelberg",
    "Iași", "Jihlava", "Klagenfurt", "Linz", "Modena",
    "Novgorod", "Oradea", "Plovdiv", "Querétaro", "Rzeszów",
    "Smyrna", "Toulouse", "Ufa", "Vyborg", "Würzburg-2",
    "Xi'an-2", "Yogyakarta", "Zwolle", "Albacete", "Brasov",
    "Cremona", "Doboj", "Esbjerg", "Fes", "Gjirokastër",
    "Heinola", "Iquique", "Jyväskylä", "Kuopio", "Lahti",
    "Mariehamn", "Naxos", "Ovid", "Petrozavodsk", "Quanzhou",
    "Rotterdam", "Spalato", "Tampere", "Ústí", "Vaasa",
]
assert len(CITIES) == 150

PROFESSIONS = [
    "philologist", "thaumaturge", "cartographer", "horologist", "lapidary",
    "geomancer", "alchemist", "mariner", "fletcher", "wheelwright",
    "millwright", "cooper", "tanner", "fuller", "miller",
    "vintner", "saddler", "weaver", "bookbinder", "scrivener",
    "calligrapher", "limner", "glazier", "armorer", "blacksmith2",
    "goldsmith", "silversmith", "tinker", "thresher", "drover",
    "stonemason", "carpenter", "joiner", "cobbler", "haberdasher",
    "milliner", "dyer", "spinner", "draper", "mercer",
    "chandler", "perfumer", "apiarist", "ostler", "wainwright",
    "wheelwright2", "shipwright", "lighterman", "ferryman", "miller2",
    "harvester", "shepherd", "cowherd", "swineherd", "goatherd",
    "huntsman", "falconer", "kennel-master", "horsemaster", "groom",
    "tutor", "bursar", "almoner", "verger", "curate",
    "rector", "vicar", "chaplain", "abbot", "prior",
    "physician", "surgeon", "apothecary", "midwife", "herbalist",
    "barber-surgeon", "dentist2", "oculist", "phlebotomist", "leech",
    "lawyer", "barrister", "solicitor", "scrivener2", "notary",
    "magistrate", "bailiff", "constable", "warden", "jailer",
    "bookseller", "printer", "engraver", "etcher", "lithographer",
    "papermaker", "bookbinder2", "type-founder", "compositor", "pressman",
    "musician", "luthier", "organist", "minstrel", "balladeer",
    "playwright", "actor", "puppeteer", "tightrope-walker", "juggler",
    "astronomer", "astrologer", "navigator", "surveyor", "geographer",
    "geologist", "naturalist", "botanist", "zoologist", "entomologist",
    "linguist", "lexicographer", "etymologist", "phonologist", "syntactician",
    "logician", "epistemologist", "ontologist", "aesthetician", "ethicist",
    "rhetorician", "orator", "elocutionist", "diplomat", "ambassador",
    "courtier", "herald", "page", "squire", "knight2",
    "swordsman", "fencer", "archer", "longbowman", "crossbowman",
    "engineer", "siege-master", "sapper", "miner2", "quarryman",
]
assert len(PROFESSIONS) == 150


PASSAGE_TEMPLATES = [
    "{name} of {city} was renowned in their day as the principal "
    "{profession} serving the merchants and gentry of the district.",

    "Among the various trades of {city}, none was practiced with greater "
    "diligence than that of {name}, the {profession}.",

    "{name}, who took up residence in {city} after a long apprenticeship, "
    "made their living as a {profession} until the very end of their days.",

    "Old records of {city} note that {name}, by trade a {profession}, "
    "was held in particular regard for the quality of their work.",
]

# Lead-in templates yielding "{...}{answer}" — same shape as
# per_passage_dickens.
LEADIN_TEMPLATES = [
    "Recall: {name}'s profession was ",
    "Q: What was the profession of {name} of {city}? A: ",
    "It is well-known that {name} of {city} worked as a ",
    "On reflection, {name}'s trade was ",
    "{name} of {city} earned a living as a ",
    "Question: What did {name} of {city} do for a living? Answer: ",
    "The trade for which {name} of {city} was known was ",
]


def build_synthetic(start_id: int = 50):
    """Build 150 synthetic entries, IDs 50..199."""
    out = []
    for i in range(150):
        name = NAMES[i]; city = CITIES[i]; prof = PROFESSIONS[i]
        # Passage uses one of 4 templates (deterministic by i)
        passage = PASSAGE_TEMPLATES[i % 4].format(
            name=name, city=city, profession=prof,
        )
        # 4 train paraphrases + 3 held-out, drawn from the 7 leadin templates
        leads = [t.format(name=name, city=city) for t in LEADIN_TEMPLATES]
        train_paras = leads[:4]
        held_out = leads[4:7]
        out.append({
            "id": start_id + i,
            "fact_type": "synthetic_profession",
            "fact": f"{name} of {city} was a {prof}.",
            "answer": prof,
            "passage": passage,
            "paraphrases_train": train_paras,
            "paraphrases_held_out": held_out,
        })
    return out


def main():
    dickens = json.loads((PPD / "data/library.json").read_text())
    assert len(dickens) == 50, f"expected 50 dickens entries, got {len(dickens)}"
    synthetic = build_synthetic(start_id=50)
    library = dickens + synthetic
    assert len(library) == 200
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(library, indent=2))
    print(f"Saved {len(library)} entries to {OUT}")

    # Spot-check a few synthetic entries
    print("\nFirst synthetic entry (id=50):")
    e = library[50]
    print(f"  fact: {e['fact']}")
    print(f"  passage: {e['passage'][:150]}...")
    print(f"  answer: {e['answer']}")
    print(f"  train_paras[0]: {e['paraphrases_train'][0]}")
    print(f"  held_out[0]:    {e['paraphrases_held_out'][0]}")


if __name__ == "__main__":
    main()
