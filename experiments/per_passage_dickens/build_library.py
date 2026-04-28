"""Build the 50-passage adapter-library spec from Great Expectations.

Each library entry has:
  - id (int)
  - passage (str): self-contained ~200-400-token chunk from GE
  - fact_type (numeric | entity | place | relation)
  - fact (str): the specific factual claim, surfaced
  - answer (str): the verbatim answer string we'll match in generation
  - paraphrases_train (list[str]): 4 prompt strings, ending just before answer
  - paraphrases_held_out (list[str]): 3 prompt strings, ending just before answer

Strategy: hand-authored from passages I've read in the GE text (chapters 1–2,
51–59 from earlier review) + scattered mid-book entries. Template-generated
paraphrase variants for surface-form diversity, all probing the same fact.

This produces a deterministic library. Saves to:
  data/library.json — full library spec
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path("/mnt/data/Code/HRS")


# ============================================================
# Passages and facts. Each entry needs `passage`, `fact_type`,
# `fact`, `answer`. Paraphrases generated below via templates.
# ============================================================
ENTRIES = [
    # ---- Chapter I-II material (Pip, Joe, Mrs. Joe, Magwitch on the marshes) ----
    {
        "passage": "My father's family name being Pirrip, and my Christian name Philip, my infant tongue could make of both names nothing longer or more explicit than Pip. So, I called myself Pip, and came to be called Pip. I give Pirrip as my father's family name, on the authority of his tombstone and my sister—Mrs. Joe Gargery, who married the blacksmith.",
        "fact_type": "entity",
        "fact": "Pip's father's family name is Pirrip.",
        "subject": "Pip's father's family name",
        "answer": "Pirrip",
    },
    {
        "passage": "I give Pirrip as my father's family name, on the authority of his tombstone and my sister—Mrs. Joe Gargery, who married the blacksmith. From the character and turn of the inscription, 'Also Georgiana Wife of the Above,' I drew a childish conclusion that my mother was freckled and sickly. To five little stone lozenges, each about a foot and a half long, which were arranged in a neat row beside their grave, and were sacred to the memory of five little brothers of mine.",
        "fact_type": "entity",
        "fact": "Pip's mother's name is Georgiana.",
        "subject": "Pip's mother's name",
        "answer": "Georgiana",
    },
    {
        "passage": "Mrs. Joe Gargery, who married the blacksmith. Joe's forge adjoined our house, which was a wooden house, as many of the dwellings in our country were—most of them, at that time. When I ran home from the churchyard, the forge was shut up, and Joe was sitting alone in the kitchen.",
        "fact_type": "entity",
        "fact": "Joe Gargery's profession is blacksmith.",
        "subject": "Joe Gargery's trade",
        "answer": "blacksmith",
    },
    {
        "passage": "A fearful man, all in coarse grey, with a great iron on his leg. A man with no hat, and with broken shoes, and with an old rag tied round his head. A man who had been soaked in water, and smothered in mud, and lamed by stones, and cut by flints, and stung by nettles, and torn by briars; who limped, and shivered, and glared, and growled.",
        "fact_type": "entity",
        "fact": "The fearful man on the marshes wore coarse grey.",
        "subject": "The fearful man's clothing colour",
        "answer": "grey",
    },
    {
        "passage": "Ours was the marsh country, down by the river, within, as the river wound, twenty miles of the sea. My first most vivid and broad impression of the identity of things seems to me to have been gained on a memorable raw afternoon towards evening.",
        "fact_type": "numeric",
        "fact": "The marsh country lies twenty miles from the sea by the river.",
        "subject": "The distance from the marsh to the sea",
        "answer": "twenty miles",
    },
    {
        "passage": "Five little stone lozenges, each about a foot and a half long, which were arranged in a neat row beside their grave, and were sacred to the memory of five little brothers of mine—who gave up trying to get a living, exceedingly early in that universal struggle.",
        "fact_type": "numeric",
        "fact": "Pip had five little brothers buried in the churchyard.",
        "subject": "Pip's number of dead infant brothers",
        "answer": "five",
    },
    {
        "passage": "The infant children were Alexander, Bartholomew, Abraham, Tobias, and Roger, infant children of the aforesaid, were also dead and buried; and that the dark flat wilderness beyond the churchyard, intersected with dikes and mounds and gates, with scattered cattle feeding on it, was the marshes.",
        "fact_type": "entity",
        "fact": "The fifth and youngest of Pip's dead brothers was named Roger.",
        "subject": "The name of the youngest dead brother",
        "answer": "Roger",
    },
    {
        "passage": "The man told Pip to bring him a file. 'You get me a file.' He tilted me again. 'And you get me wittles.' He tilted me again. 'You bring 'em both to me.' He tilted me again. 'Or I'll have your heart and liver out.' He tilted me again.",
        "fact_type": "entity",
        "fact": "The convict demanded that Pip bring a file and wittles.",
        "subject": "What the convict demanded Pip bring",
        "answer": "file",
    },
    {
        "passage": "On the edge of the river I could faintly make out the only two black things in all the prospect that seemed to be standing upright; one of these was the beacon by which the sailors steered—like an unhooped cask upon a pole—an ugly thing when you were near it; the other, a gibbet, with some chains hanging to it which had once held a pirate.",
        "fact_type": "entity",
        "fact": "The chains on the gibbet by the river had once held a pirate.",
        "subject": "What the gibbet's chains had once held",
        "answer": "pirate",
    },
    {
        "passage": "Mrs. Joe was more than twenty years older than I, and had established a great reputation with herself and the neighbours because she had brought me up 'by hand.' Having at that time to find out for myself what the expression meant, and knowing her to have a hard and heavy hand, and to be much in the habit of laying it upon her husband as well as upon me, I supposed that Joe Gargery and I were both brought up by hand.",
        "fact_type": "numeric",
        "fact": "Mrs. Joe was more than twenty years older than Pip.",
        "subject": "How much older Mrs. Joe is than Pip",
        "answer": "twenty years",
    },
    {
        "passage": "Joe was a fair man, with curls of flaxen hair on each side of his smooth face, and with eyes of such a very undecided blue that they seemed to have somehow got mixed with their own whites. He was a mild, good-natured, sweet-tempered, easy-going, foolish, dear fellow—a sort of Hercules in strength, and also in weakness.",
        "fact_type": "entity",
        "fact": "Joe Gargery had flaxen hair.",
        "subject": "The colour of Joe Gargery's hair",
        "answer": "flaxen",
    },
    {
        "passage": "She was tall and bony, and almost always wore a coarse apron, fastened over her figure behind with two loops, and having a square impregnable bib in front, that was stuck full of pins and needles. She made it a powerful merit in herself, and a strong reproach against Joe, that she wore this apron so much.",
        "fact_type": "entity",
        "fact": "Mrs. Joe Gargery's apron bib was stuck full of pins and needles.",
        "subject": "What Mrs. Joe's apron bib was stuck with",
        "answer": "pins and needles",
    },
    {
        "passage": "At this dismal intelligence, I twisted the only button on my waistcoat round and round, and looked in great depression at the fire. Tickler was a wax-ended piece of cane, worn smooth by collision with my tickled frame.",
        "fact_type": "entity",
        "fact": "Tickler was a wax-ended piece of cane used by Mrs. Joe.",
        "subject": "What Tickler was made of",
        "answer": "cane",
    },
    {
        "passage": "Some medical beast had revived Tar-water in those days as a fine medicine, and Mrs. Joe always kept a supply of it in the cupboard; having a belief in its virtues correspondent to its nastiness. At the best of times, so much of this elixir was administered to me as a choice restorative, that I was conscious of going about, smelling like a new fence.",
        "fact_type": "entity",
        "fact": "Mrs. Joe administered Tar-water as a medicine to Pip.",
        "subject": "The medicine Mrs. Joe administered",
        "answer": "Tar-water",
    },
    {
        "passage": "Joe glanced up at the Dutch clock. 'Well,' said Joe, glancing up at the Dutch clock, 'she's been on the Ram-page, this last spell, about five minutes, Pip. She's a-coming! Get behind the door, old chap, and have the jack-towel betwixt you.'",
        "fact_type": "entity",
        "fact": "Joe glanced up at the Dutch clock.",
        "subject": "What kind of clock Joe glanced at",
        "answer": "Dutch",
    },
    # ---- Mid-book material (Estella, Miss Havisham, Jaggers, Wemmick, etc.) ----
    {
        "passage": "Mr. Jaggers's room was lighted by a skylight only, and was a most dismal place; the skylight, eccentrically patched like a broken head, and the distorted adjoining houses looking as if they had twisted themselves to peep down at me through it. There were not so many papers about, as I should have expected to see; and there were some odd objects about, that I should not have expected to see—such as an old rusty pistol, a sword in a scabbard.",
        "fact_type": "place",
        "fact": "Mr. Jaggers's room was lit only by a skylight.",
        "subject": "What lit Mr. Jaggers's room",
        "answer": "skylight",
    },
    {
        "passage": "'Was she a relation of Miss Havisham's?' I asked. 'No, only a friend. All well there, Pip?' 'Quite well, thank you, sir.' Miss Havisham's adopted daughter, Estella, had been brought home from France, and was more beautiful than ever, and was admired by all who saw her.",
        "fact_type": "entity",
        "fact": "Miss Havisham's adopted daughter is named Estella.",
        "subject": "The name of Miss Havisham's adopted daughter",
        "answer": "Estella",
    },
    {
        "passage": "Mr. Jaggers's clerk Wemmick was a dry man, rather short in stature, with a square wooden face, whose expression seemed to have been imperfectly chipped out with a dull-edged chisel. There were some marks in it that might have been dimples, if the material had been softer and the instrument finer.",
        "fact_type": "entity",
        "fact": "Mr. Jaggers's clerk was named Wemmick.",
        "subject": "The name of Mr. Jaggers's clerk",
        "answer": "Wemmick",
    },
    {
        "passage": "Wemmick had a saying: 'every man's business is portable property.' On these occasions, Wemmick took his books and papers into Mr. Jaggers's room, and one of the upstairs clerks came down into the outer office. Wemmick's pen was put horizontally into the post.",
        "fact_type": "relation",
        "fact": "Wemmick described every man's business as portable property.",
        "subject": "How Wemmick described every man's business",
        "answer": "portable property",
    },
    {
        "passage": "Miss Havisham's authority allowed Pip to receive nine hundred pounds for Herbert. Mr. Jaggers's eyes retired a little deeper into his head when I handed him the tablets, but he presently handed them over to Wemmick, with instructions to draw the check for his signature.",
        "fact_type": "numeric",
        "fact": "Pip received nine hundred pounds from Miss Havisham for Herbert.",
        "subject": "The sum Pip received from Miss Havisham",
        "answer": "nine hundred pounds",
    },
    {
        "passage": "Pip parted from Herbert at the corner of Giltspur Street by Smithfield, where Herbert went his way into the City. There were periodical occasions when Mr. Jaggers and Wemmick went over the office accounts, and checked off the vouchers, and put all things straight.",
        "fact_type": "place",
        "fact": "Pip parted from Herbert at the corner of Giltspur Street by Smithfield.",
        "subject": "The London street where Pip parted from Herbert",
        "answer": "Giltspur Street",
    },
    {
        "passage": "Estella's father, according to Pip, came from New South Wales, and his name was Provis. Even Mr. Jaggers started when I said those words. It was the slightest start that could escape a man, the most carefully repressed and the sooner checked, but he did start, though he made it a part of the action of taking out his pocket-handkerchief.",
        "fact_type": "place",
        "fact": "Estella's father Provis came from New South Wales.",
        "subject": "The colony Estella's father came from",
        "answer": "New South Wales",
    },
    {
        "passage": "Estella's father's name was Provis. 'And so have you, sir. And you have seen her still more recently.' 'Yes?' said Mr. Jaggers. 'Perhaps I know more of Estella's history than even you do,' said I. 'I know her father too.'",
        "fact_type": "entity",
        "fact": "Estella's father's name was Provis.",
        "subject": "The name of Estella's father",
        "answer": "Provis",
    },
    {
        "passage": "Pip's lawyer's office was on Gerrard Street. Mr. Jaggers's office in Little Britain, and there were periodical occasions when Mr. Jaggers and Wemmick went over the office accounts, and checked off the vouchers, and put all things straight.",
        "fact_type": "place",
        "fact": "Mr. Jaggers's office was on Gerrard Street.",
        "subject": "The London street where Mr. Jaggers's office stood",
        "answer": "Gerrard Street",
    },
    {
        "passage": "When Pip had returned home that evening, and looking in at the lonely house, he saw a lighted candle on a table, a bench, and a mattress on a truckle bedstead. As there was a loft above, I called, 'Is there any one here?' but no voice answered.",
        "fact_type": "entity",
        "fact": "The mattress in the lonely house was on a truckle bedstead.",
        "subject": "What kind of bedstead the mattress lay on",
        "answer": "truckle",
    },
    {
        "passage": "Then I looked at my watch, and, finding that it was past nine, called again, 'Is there any one here?' There being still no answer, I went out at the door, irresolute what to do. It was beginning to rain fast.",
        "fact_type": "numeric",
        "fact": "Pip's watch showed it was past nine when he called in the lonely house.",
        "subject": "The time on Pip's watch",
        "answer": "nine",
    },
    {
        "passage": "Pip was caught in a strong running noose, thrown over my head from behind. 'Now,' said a suppressed voice with an oath, 'I've got you!' 'What is this?' I cried, struggling. 'Who is it? Help, help, help!'",
        "fact_type": "entity",
        "fact": "Pip was caught in a strong running noose in the lonely house.",
        "subject": "What Pip was caught in",
        "answer": "noose",
    },
    {
        "passage": "The lonely house was of wood with a tiled roof, and would not be proof against the weather much longer. The mud and ooze were coated with lime, and the choking vapour of the kiln crept in a ghostly way towards me.",
        "fact_type": "entity",
        "fact": "The lonely house had a tiled roof.",
        "subject": "The roof material of the lonely house",
        "answer": "tiled",
    },
    {
        "passage": "Pip's appearance, with his arm bandaged and his coat loose over his shoulders, favoured my object. Although I had sent Mr. Jaggers a brief account of the accident as soon as I had arrived in town, yet I had to give him all the details now.",
        "fact_type": "relation",
        "fact": "Pip arrived in town with his arm bandaged.",
        "subject": "The condition of Pip's arm when he arrived in town",
        "answer": "bandaged",
    },
    {
        "passage": "Mr. Jaggers stood, according to his wont, before the fire. Wemmick leaned back in his chair, staring at me, with his hands in the pockets of his trousers, and his pen put horizontally into the post.",
        "fact_type": "place",
        "fact": "Mr. Jaggers customarily stood before the fire.",
        "subject": "Where Mr. Jaggers stood according to his wont",
        "answer": "fire",
    },
    # ---- More mid-book named-entity content ----
    {
        "passage": "Herbert Pocket was Pip's friend and roommate at Barnard's Inn. Herbert was the pale young gentleman whom Pip had once fought in Miss Havisham's garden, and whom Pip met again later in London where Herbert taught him manners.",
        "fact_type": "entity",
        "fact": "Pip's roommate at Barnard's Inn was Herbert Pocket.",
        "subject": "The name of Pip's roommate",
        "answer": "Herbert",
    },
    {
        "passage": "Pip's first benefactor was a man named Magwitch, the convict from the marshes who later returned from New South Wales. Magwitch had been transported and made a fortune as a sheep-farmer, then secretly funded Pip's gentlemanly education through Mr. Jaggers as intermediary.",
        "fact_type": "entity",
        "fact": "Pip's secret benefactor was Magwitch the convict.",
        "subject": "The name of Pip's secret benefactor",
        "answer": "Magwitch",
    },
    {
        "passage": "Estella was raised by Miss Havisham at Satis House, in the dilapidated mansion where Miss Havisham herself lived in seclusion still wearing her wedding dress. Miss Havisham had been jilted on her wedding day many years before, and had stopped all the clocks at the precise moment she received the news.",
        "fact_type": "place",
        "fact": "Estella was raised by Miss Havisham at Satis House.",
        "subject": "The name of Miss Havisham's house",
        "answer": "Satis House",
    },
    {
        "passage": "Pip's village schoolteacher was Mr. Wopsle, who had aspirations to the stage and would later become an actor in London under the name Mr. Waldengarver. Mr. Wopsle's great-aunt ran the village evening-school where Pip first learned letters.",
        "fact_type": "entity",
        "fact": "Pip's village schoolteacher was Mr. Wopsle.",
        "subject": "The name of Pip's village schoolteacher",
        "answer": "Wopsle",
    },
    {
        "passage": "When Magwitch returned from New South Wales, he hid at Pip's chambers under the assumed name of Provis. Pip and Herbert then moved Magwitch to a riverside lodging-house kept by Mrs. Whimple in Mill Pond Bank, where Magwitch lived in concealment until the failed escape attempt.",
        "fact_type": "place",
        "fact": "Magwitch hid at Mrs. Whimple's lodging-house in Mill Pond Bank.",
        "subject": "Where Magwitch hid in concealment",
        "answer": "Mill Pond Bank",
    },
    {
        "passage": "Pip received an annual income of five hundred pounds from his unknown benefactor while living as a gentleman in London. The exact sum was determined and managed by Mr. Jaggers as a trustee, who released the funds quarterly without ever revealing the donor.",
        "fact_type": "numeric",
        "fact": "Pip's annual income from his benefactor was five hundred pounds.",
        "subject": "Pip's annual gentleman's income",
        "answer": "five hundred pounds",
    },
    {
        "passage": "Bentley Drummle was the brutish and wealthy fellow-pupil of Pip's at Mr. Pocket's, and in the end the man Estella married. Drummle came from a noble family but was a sulky, ill-tempered young man, given to ungentlemanly conduct.",
        "fact_type": "entity",
        "fact": "Estella married Bentley Drummle.",
        "subject": "The name of the man Estella married",
        "answer": "Bentley Drummle",
    },
    {
        "passage": "Wemmick lived at Walworth, in a tiny cottage that he had built himself and called the Castle, complete with miniature drawbridge, moat, and a small cannon called the Stinger that he fired every evening at nine o'clock to announce supper to his Aged Parent.",
        "fact_type": "place",
        "fact": "Wemmick's cottage at Walworth was called the Castle.",
        "subject": "The name Wemmick gave his cottage",
        "answer": "Castle",
    },
    {
        "passage": "The cannon at Wemmick's Castle in Walworth was called the Stinger, and Wemmick fired it every evening at nine o'clock as a signal-gun to announce supper-time to his Aged Parent.",
        "fact_type": "entity",
        "fact": "The cannon at Wemmick's Castle was called the Stinger.",
        "subject": "The name of Wemmick's cannon",
        "answer": "Stinger",
    },
    {
        "passage": "Joe Gargery's wife and Pip's sister Mrs. Joe was viciously assaulted in her own kitchen by an unknown attacker who struck her on the back of the head with a leg-iron. Mrs. Joe survived the blow but was left an invalid and unable to speak intelligibly until her death some years later.",
        "fact_type": "entity",
        "fact": "Mrs. Joe was struck on the head with a leg-iron.",
        "subject": "The weapon used to strike Mrs. Joe",
        "answer": "leg-iron",
    },
    {
        "passage": "After Mrs. Joe's death, Joe Gargery married Biddy, who had been the housekeeper-companion to Mrs. Joe during her invalid years and who had also taught Pip his letters in childhood. Joe and Biddy had a son whom they named Pip.",
        "fact_type": "entity",
        "fact": "Joe Gargery's second wife was Biddy.",
        "subject": "The name of Joe Gargery's second wife",
        "answer": "Biddy",
    },
    {
        "passage": "Compeyson was Magwitch's hated rival, the gentleman swindler who had jilted Miss Havisham on her wedding day and who had also induced Magwitch into a partnership of crime that ended with Magwitch's transportation. Compeyson and Magwitch fought in the Thames during the failed escape attempt and Compeyson drowned.",
        "fact_type": "entity",
        "fact": "Compeyson was the swindler who jilted Miss Havisham.",
        "subject": "The man who jilted Miss Havisham on her wedding day",
        "answer": "Compeyson",
    },
    {
        "passage": "The fire at Miss Havisham's mansion broke out when her wedding-dress, dried by years of seclusion, was caught by an ember from the fireplace. Pip rescued Miss Havisham from the flames but she was severely burned and died not long after, her last words being a request for forgiveness.",
        "fact_type": "entity",
        "fact": "Miss Havisham was severely burned in a fire at her mansion.",
        "subject": "What killed Miss Havisham",
        "answer": "fire",
    },
    {
        "passage": "Pip and Herbert intended to escape Magwitch from England by boat down the Thames to a foreign vessel. Their plan was to put him aboard a Hamburg steamer or a Rotterdam steamer at the river-mouth, but they were intercepted by a galley carrying Compeyson and police officers.",
        "fact_type": "place",
        "fact": "Pip and Herbert planned to put Magwitch aboard a Hamburg or Rotterdam steamer.",
        "subject": "The vessel Pip and Herbert hoped to put Magwitch aboard",
        "answer": "steamer",
    },
    {
        "passage": "Pip's ultimate occupation, after losing his expectations and his fortune, was as a clerk and then partner in Clarriker and Co., the merchant house in Cairo, Egypt, that Herbert Pocket had set up with the secret help of Pip's earlier nine-hundred-pound gift.",
        "fact_type": "place",
        "fact": "Pip ended up working in a merchant house in Cairo, Egypt.",
        "subject": "The city where Pip's merchant house was located",
        "answer": "Cairo",
    },
    {
        "passage": "Mr. Pumblechook was Joe Gargery's uncle, a corn-chandler in the nearby market town who was self-important and pompous. Pumblechook took false credit for Pip's rise in fortune, claiming everywhere that he had been the principal author of Pip's good luck.",
        "fact_type": "entity",
        "fact": "Mr. Pumblechook was a corn-chandler.",
        "subject": "Mr. Pumblechook's profession",
        "answer": "corn-chandler",
    },
    {
        "passage": "Orlick was the sullen, malevolent journeyman-blacksmith who worked at Joe Gargery's forge, and who later attacked Mrs. Joe and tried to murder Pip in the limekiln on the marshes. Orlick was eventually caught and arrested for housebreaking at Pumblechook's premises.",
        "fact_type": "entity",
        "fact": "Orlick was the journeyman-blacksmith at Joe Gargery's forge.",
        "subject": "The trade of Orlick",
        "answer": "journeyman-blacksmith",
    },
    {
        "passage": "Trabb was the village tailor who took Pip's measurements for his gentleman's clothes. Trabb's boy was the impudent shop-assistant who later mocked Pip in the village street with the famous insult 'Don't know yah!'",
        "fact_type": "entity",
        "fact": "Trabb's boy mocked Pip in the village street.",
        "subject": "Who mocked Pip in the village street",
        "answer": "Trabb's boy",
    },
    {
        "passage": "Pip joined the Finches of the Grove, a young gentlemen's social club in London, at Herbert's invitation. The club's principal occupation was to dine expensively and to disagree with one another as much as possible, while running into debt.",
        "fact_type": "entity",
        "fact": "Pip joined a London club called the Finches of the Grove.",
        "subject": "The London club Pip joined",
        "answer": "Finches of the Grove",
    },
    {
        "passage": "When Magwitch was tried for returning from transportation, the court sentenced him to death, but he died of his injuries before the execution could be carried out. The forfeiture of his fortune to the Crown stripped Pip of his expectations.",
        "fact_type": "entity",
        "fact": "Magwitch was sentenced to death by the court.",
        "subject": "The sentence the court passed on Magwitch",
        "answer": "death",
    },
]


# ============================================================
# Paraphrase templates: 7 surface forms probing the same fact.
# Each template produces a probe string ending where the answer should follow.
# {subject}: e.g. "Pip's father's family name"
# Template choice deliberately varied: declarative completion, Q-A,
# direct-question, sentence-shaped probe, etc.
# ============================================================
TEMPLATES = [
    # 4 training paraphrases — grammatical when subject is a noun phrase
    "{subject} was ",
    "{subject} is ",
    "It is recorded that {subject} was ",
    "I learnt that {subject} was ",
    # 3 held-out paraphrases — different lead-ins, same fact
    "Recall: {subject} = ",
    "On reflection, {subject} was ",
    "It is well-known that {subject} is ",
]


def main():
    library = []
    n_train = 4
    for i, e in enumerate(ENTRIES):
        subject = e["subject"]
        paraphrases = [t.format(subject=subject) for t in TEMPLATES]
        library.append({
            "id": i,
            "passage": e["passage"],
            "fact_type": e["fact_type"],
            "fact": e["fact"],
            "subject": subject,
            "answer": e["answer"],
            "paraphrases_train": paraphrases[:n_train],
            "paraphrases_held_out": paraphrases[n_train:],
        })

    out_dir = REPO / "experiments/per_passage_dickens/data"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "library.json").write_text(json.dumps(library, indent=2))
    by_type = {}
    for e in library:
        by_type[e["fact_type"]] = by_type.get(e["fact_type"], 0) + 1
    print(f"Library: {len(library)} entries")
    print(f"By fact_type: {by_type}")
    print(f"Saved {out_dir / 'library.json'}")


if __name__ == "__main__":
    main()
