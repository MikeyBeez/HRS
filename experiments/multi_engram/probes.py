"""25 synthesis-style probes for the multi-engram experiment.

5 used to train W; 20 held out for evaluation.

Each probe has:
  - a `prompt` (the question to put to the model)
  - a `relevant_topics` list of titles that should be retrieved.
    These are matched against `topics.ALL_TOPICS` titles (substring match
    on a few keywords each), and the matching turn ids form the
    ground-truth set.

The relevance is intentionally permissive — many turns plausibly speak
to a synthesis question. The threshold "at least 5 of the 100 turns"
from the spec is met for every probe.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

from experiments.multi_engram.topics import ALL_TOPICS

# Each probe lists CATEGORY tags + specific TITLE keywords. A turn is
# considered "relevant" if its category matches OR its title matches any
# of the listed substrings (case-insensitive).

VAL_PROBES = [
    {
        "prompt": "Write a short report on the role of railroads in the Civil War's outcome.",
        "categories": ["logistics"],
        "title_keywords": ["railroad", "Anaconda", "blockade",
                            "quartermaster", "logistics"],
    },
    {
        "prompt": "Summarize the major strategic decisions of Robert E. Lee during the war.",
        "categories": [],
        "title_keywords": ["Robert E. Lee", "Antietam", "Gettysburg",
                            "Chancellorsville", "Wilderness", "Maryland",
                            "Overland", "Appomattox", "Pickett",
                            "Fredericksburg", "Seven Days"],
    },
    {
        "prompt": "Give an overview of how slavery and its abolition shaped the war.",
        "categories": [],
        "title_keywords": ["slavery", "Emancipation", "13th Amendment",
                            "USCT", "United States Colored", "abolition"],
    },
    {
        "prompt": "Describe the technological innovations that emerged during the Civil War.",
        "categories": ["logistics"],
        "title_keywords": ["telegraph", "ironclad", "rifles", "submarine",
                            "Hunley", "photography", "balloons", "trench"],
    },
    {
        "prompt": "Compare Union and Confederate political leadership.",
        "categories": ["political"],
        "title_keywords": ["Lincoln", "Davis", "Stanton", "Seward",
                            "Andrew Johnson", "Stephens", "Confederate"],
    },
]

TEST_PROBES = [
    {
        "prompt": "Write a report on Ulysses S. Grant's military career during the war.",
        "categories": [],
        "title_keywords": ["Grant", "Vicksburg", "Shiloh", "Donelson",
                            "Wilderness", "Spotsylvania", "Cold Harbor",
                            "Petersburg", "Appomattox", "Overland",
                            "Chattanooga"],
    },
    {
        "prompt": "Summarize the campaigns and strategic mistakes of the Confederate command.",
        "categories": [],
        "title_keywords": ["Lee", "Davis", "Bragg", "Hood", "Johnston",
                            "Pickett", "Antietam", "Gettysburg",
                            "Atlanta", "Vicksburg", "Maryland",
                            "Confederate"],
    },
    {
        "prompt": "Give an overview of how the Civil War affected civilian populations.",
        "categories": ["social"],
        "title_keywords": ["civilian", "refugee", "draft", "riot",
                            "economic", "religion", "guerilla", "women"],
    },
    {
        "prompt": "Describe the major naval and water-borne aspects of the Civil War.",
        "categories": [],
        "title_keywords": ["naval", "blockade", "ironclad", "Hampton Roads",
                            "Monitor", "Hunley", "submarine", "Mobile",
                            "New Orleans", "Anaconda"],
    },
    {
        "prompt": "Explain the role and experience of African American troops during the war.",
        "categories": [],
        "title_keywords": ["USCT", "United States Colored", "African",
                            "slavery", "Emancipation", "abolition"],
    },
    {
        "prompt": "Trace the western theater of the Civil War from beginning to end.",
        "categories": [],
        "title_keywords": ["Shiloh", "Vicksburg", "Chattanooga",
                            "Atlanta", "Sherman", "March to the Sea",
                            "Stones River", "Chickamauga", "Donelson",
                            "Tullahoma", "Knoxville", "Pea Ridge"],
    },
    {
        "prompt": "Describe the medical and human cost of the war on soldiers.",
        "categories": [],
        "title_keywords": ["medicine", "disease", "Andersonville",
                            "prisoner", "casualties", "veterans"],
    },
    {
        "prompt": "Compare the major engagements of 1862 and their consequences.",
        "categories": [],
        "title_keywords": ["Antietam", "Shiloh", "Seven Days", "Bull Run",
                            "Fredericksburg", "Stones River", "Maryland",
                            "Peninsula", "Hampton Roads", "Donelson"],
    },
    {
        "prompt": "Describe Sherman's strategic role in the war's final two years.",
        "categories": [],
        "title_keywords": ["Sherman", "Atlanta", "March to the Sea",
                            "Carolinas", "Chattanooga"],
    },
    {
        "prompt": "Give an overview of how the war reshaped the U.S. economy.",
        "categories": [],
        "title_keywords": ["economic", "railroad", "telegraph",
                            "quartermaster", "blockade", "Anaconda",
                            "veterans", "Reconstruction"],
    },
    {
        "prompt": "Summarize the role of cavalry on both sides of the war.",
        "categories": [],
        "title_keywords": ["cavalry", "Stuart", "Sheridan", "Forrest",
                            "Pea Ridge"],
    },
    {
        "prompt": "Describe the political path from Fort Sumter to Appomattox.",
        "categories": ["political"],
        "title_keywords": ["Fort Sumter", "Appomattox", "Lincoln", "Davis",
                            "Election of 1864", "Emancipation",
                            "13th Amendment", "habeas"],
    },
    {
        "prompt": "Explain how each side mobilized manpower and overcame recruitment problems.",
        "categories": [],
        "title_keywords": ["Conscription", "Draft Riot", "USCT",
                            "United States Colored", "veterans",
                            "quartermaster"],
    },
    {
        "prompt": "Describe the eastern theater campaigns and battles in chronological order.",
        "categories": [],
        "title_keywords": ["Bull Run", "Peninsula", "Antietam",
                            "Fredericksburg", "Chancellorsville",
                            "Gettysburg", "Wilderness", "Spotsylvania",
                            "Cold Harbor", "Petersburg", "Appomattox",
                            "Maryland", "Overland", "Seven Days"],
    },
    {
        "prompt": "Discuss the role of foreign powers and Confederate diplomacy during the war.",
        "categories": [],
        "title_keywords": ["Britain", "France", "diplomacy", "Trent",
                            "blockade"],
    },
    {
        "prompt": "Summarize the major sieges and their outcomes.",
        "categories": [],
        "title_keywords": ["Vicksburg", "Petersburg", "Chattanooga",
                            "Atlanta", "Mobile", "Knoxville"],
    },
    {
        "prompt": "Discuss the impact of the war on women and gender roles.",
        "categories": [],
        "title_keywords": ["women"],
    },
    {
        "prompt": "Explain the experiences of Civil War prisoners of war on both sides.",
        "categories": [],
        "title_keywords": ["Andersonville", "prisoner", "POW"],
    },
    {
        "prompt": "Describe the leadership style and key decisions of Abraham Lincoln.",
        "categories": [],
        "title_keywords": ["Lincoln", "Emancipation", "habeas",
                            "Election of 1864", "Stanton", "Seward",
                            "13th Amendment"],
    },
    {
        "prompt": "Discuss key Confederate generals besides Robert E. Lee.",
        "categories": [],
        "title_keywords": ["Stonewall Jackson", "Longstreet", "Joseph E. Johnston",
                            "Stuart", "Forrest", "Bragg", "Albert Sidney Johnston",
                            "Beauregard", "Hood", "Pickett"],
    },
]


def title_matches(turn_title, keywords):
    """Case-insensitive substring match against any keyword."""
    lt = turn_title.lower()
    return any(k.lower() in lt for k in keywords)


def find_relevant_ids(probe, all_topics=ALL_TOPICS):
    """Returns a list of turn-ids in `all_topics` whose category is in
    probe['categories'] OR whose title matches any of probe['title_keywords']."""
    ids = []
    for i, (title, cat) in enumerate(all_topics):
        if cat in probe.get("categories", []):
            ids.append(i)
        elif title_matches(title, probe.get("title_keywords", [])):
            ids.append(i)
    return ids


def build_probe_records(probes):
    out = []
    for p in probes:
        rel = find_relevant_ids(p)
        out.append({**p, "relevant_ids": rel, "n_relevant": len(rel)})
    return out


if __name__ == "__main__":
    val = build_probe_records(VAL_PROBES)
    test = build_probe_records(TEST_PROBES)
    print("VAL probes:")
    for p in val:
        print(f"  '{p['prompt'][:60]}...' → {p['n_relevant']} relevant turns")
    print("\nTEST probes:")
    for p in test:
        print(f"  '{p['prompt'][:60]}...' → {p['n_relevant']} relevant turns")
    n_at_least_5 = sum(1 for p in test if p["n_relevant"] >= 5)
    print(f"\n{n_at_least_5}/{len(test)} test probes have >= 5 relevant turns")
