# Phase 02 — All miss cases by condition and category

## C1 baseline (no doc, no adapter, no engram)

### A_character   (0/50 hit; 50 miss)

| target | model top-1 | rank | top-5 | source / cue |
|--------|-------------|------|-------|--------------|
| ' T' | ' ' | 15 | ' ' / ' M' / ' B' / ' S' / ' H' | character='Tulkinghorn'<br>tail: `t same moment, there happens to be an old man of the name of` |
| ' All' | ' you' | 1273 | ' you' / ' the' / ' I' / ' a' / ' it' | character='Allan Woodcourt'<br>tail: `The broken footway is so narrow that when` |
| ' Est' | ' g' | 901 | ' g' / ' boy' / ',' / ' L' / ' bo' | character='Esther Summerson'<br>tail: `here I was a day boarder, and although they called me little` |
| ' Ge' | ' good' | 1620 | ' good' / ' a' / ' beautiful' / ' he' / ' there' | character='George Rouncewell'<br>tail: `“Good heaven, and it is really` |
| ' Ada' | 'our' | 475 | 'our' / 'iss' / ' ' / ' M' / 'ile' | character='Ada Clare'<br>tail: `Richard Carstone and of Miss` |
| ' Richard' | ' was' | 556 | ' was' / ' is' / '\n' / ',' / ' I' | character='Richard Carstone'<br>tail: `gentleman was her distant cousin, she told me, and his name` |
| ' S' | ' I' | 301 | ' I' / ' as' / ' the' / ' when' / ' it' | character='Sir Leicester'<br>tail: `sper still goes about that she had not even family; howbeit,` |
| ' K' | ' s' | 104 | ' s' / ' my' / ' I' / ' m' / ' for' | character='Krook'<br>tail: `raciously observed to him before passing out, “That will do,` |
| ' Har' | ' your' | 1497 | ' your' / ' the' / ' us' / ' no' / ' me' | character='Harold Skimpole'<br>tail: `, holiness, commerce, trade, any object you prefer; only—let` |
| ' H' | ' de' | 17 | ' de' / ' is' / ' was' / ' has' / ' and' | character='Hortense'<br>tail: `In short, it is such an admirable thing that Mademoiselle` |
| ' Ins' | ' not' | 1802 | ' not' / ' no' / ' a' / ' ' / ' the' | character='Inspector Bucket'<br>tail: `ng that may appear to be disagreeable in this, for my name’s` |
| ' John' | ' the' | 112 | ' the' / ' a' / ' your' / ' ‘' / ' my' | character='John Jarndyce'<br>tail: `” “Since you refer so immediately to` |
| ' C' | ' my' | 34 | ' my' / ' the' / ' some' / ' with' / ' up' | character='Caddy Jellyby'<br>tail: `So I thought one day when I went to London to meet` |
| ' L' | ' mother' | 43 | ' mother' / ' father' / ' heart' / ' mind' / ' sist' | character='Lady Dedlock'<br>tail: `dded to these, soon floated her upward, and for years now my` |
| ' T' | ' he' | 142 | ' he' / ' the' / ' a' / ' you' / ' it' | character='Tulkinghorn'<br>tail: `now observes from his couch that man told him ya’as’dy that` |
| ' All' | ' it' | 975 | ' it' / ' he' / ' the' / ' I' / ' one' | character='Allan Woodcourt'<br>tail: `ome excrescence produced there in neglect and impurity, that` |
| ' Est' | ',' | 934 | ',' / 'our' / 'is' / '.' / 'us' | character='Esther Summerson'<br>tail: `We are, Madam, Your obedt Servts, Kenge and Carboy Miss` |
| ' Ge' | ' and' | 361 | ' and' / ' but' / ' all' / ' as' / ' the' | character='George Rouncewell'<br>tail: `“All is still in readiness,` |
| ' Richard' | ' a' | 3084 | ' a' / ' the' / ' an' / ' no' / ' his' | character='Richard Carstone'<br>tail: `and released her, and then he spoke for a minute or two with` |
| ' S' | ' me' | 39 | ' me' / ' the' / ' him' / ' his' / ' my' | character='Sir Leicester'<br>tail: `n air of prescription about him which is always agreeable to` |
| ' K' | ' of' | 186 | ' of' / ' or' / ' who' / ' and' / ' whose' | character='Krook'<br>tail: `as is announced in paint, to all whom it may concern, by one` |
| ' Har' | ' a' | 2465 | ' a' / ' the' / ' been' / ' done' / ' no' | character='Harold Skimpole'<br>tail: `Then, for heaven’s sake, having` |
| ' H' | ' the' | 25 | ' the' / ' her' / ' a' / ' my' / '\n' | character='Hortense'<br>tail: `to attend,” says my Lady then, addressing the reflection of` |
| ' Ins' | ' it' | 2059 | ' it' / ' she' / ' you' / ' the' / ' he' | character='Inspector Bucket'<br>tail: `y about admitting of it, you tell her that it’s no use, that` |
| ' John' | ' the' | 94 | ' the' / ' your' / ' me' / ' this' / ' him' | character='John Jarndyce'<br>tail: `I suppose your loyalty to` |
| ' C' | ' we' | 33 | ' we' / ' the' / ' I' / ' there' / ' a' | character='Caddy Jellyby'<br>tail: `At last we came to Soho Square, where` |
| ' L' | ' heart' | 6 | ' heart' / ' l' / ' w' / ' eyes' / ' mother' | character='Lady Dedlock'<br>tail: `With all her perfections on her head, my` |
| ' T' | ' world' | 43 | ' world' / ' past' / ' anc' / ' earth' / ' universe' | character='Tulkinghorn'<br>tail: `ined to add the last great secret to the many secrets of the` |
| ' All' | ' her' | 453 | ' her' / ' him' / ' it' / ' them' / ' the' | character='Allan Woodcourt'<br>tail: `s, a farewell to her, and takes his creeping way along after` |
| ' Est' | ' g' | 1360 | ' g' / ',' / ' boy' / ' by' / ' one' | character='Esther Summerson'<br>tail: `And yet I—I, little` |
| ' Ge' | ' that' | 1375 | ' that' / ' and' / ' he' / ' but' / ' “' | character='George Rouncewell'<br>tail: `Very familiar to him, as he said himself some hours ago,` |
| ' Richard' | 'I' | 31617 | 'I' / '<NAME>' / 'No' / '—' / 'My' | character='Richard Carstone'<br>tail: `YOUR name now will be—” “` |
| ' S' | ' the' | 7 | ' the' / ' he' / ' ' / ' she' / ' I' | character='Sir Leicester'<br>tail: `“Better now,” quoth` |
| ' K' | ' the' | 74 | ' the' / ' if' / ' it' / ' a' / ' he' | character='Krook'<br>tail: `The welcome light soon shines upon the wall, as` |
| ' Har' | ' you' | 741 | ' you' / ' the' / ' any' / ' us' / ' me' | character='Harold Skimpole'<br>tail: `Mankind will surely not deny to` |
| ' H' | '.' | 17 | '.' / ',' / ' to' / ' and' / ' de' | character='Hortense'<br>tail: `t, a peaceful figure too in the landscape, went Mademoiselle` |
| ' Ins' | ' the' | 3115 | ' the' / ' her' / ' a' / ' your' / ' my' | character='Inspector Bucket'<br>tail: `Put it to her ladyship, if you think it right, from` |
| ' John' | ' me' | 499 | ' me' / ' the' / ' life' / ' your' / ' my' | character='John Jarndyce'<br>tail: `” “There you come back to` |
| ' C' | ' you' | 25 | ' you' / ' ' / ' her' / ' that' / ' she' | character='Caddy Jellyby'<br>tail: `r was a greater imposter than I with a blinder follower than` |
| ' L' | ' work' | 169 | ' work' / ' name' / ' life' / ' own' / ' career' | character='Lady Dedlock'<br>tail: `le circumstance to be noted in everything associated with my` |
| ' T' | ' ' | 13 | ' ' / ' W' / ' M' / ' H' / ' Mr' | character='Tulkinghorn'<br>tail: `en a murder in Lincoln’s Inn Fields—gentleman of the name of` |
| ' All' | ' Jo' | 138 | ' Jo' / ' a' / 'hes' / ' the' / ' soon' | character='Allan Woodcourt'<br>tail: `CHAPTER XLVII Jo’s Will As` |
| ' Est' | ' ' | 201 | ' ' / ' M' / ' "' / ' J' / ' A' | character='Esther Summerson'<br>tail: `I was left in charge of a child named` |
| ' S' | ' the' | 203 | ' the' / ' example' / ' I' / ' it' / ' instance' | character='Sir Leicester'<br>tail: `re is any superabundant life of imagination on the spot, for` |
| ' K' | ' ' | 4 | ' ' / ' the' / ' B' / ' M' / ' K' | character='Krook'<br>tail: `I don’t know,” says` |
| ' Har' | 'The' | 21350 | 'The' / 'I' / 'If' / 'We' / 'You' | character='Harold Skimpole'<br>tail: `Skimpole, “to this effect: ‘` |
| ' H' | '.' | 83 | '.' / ',' / '.”' / '!' / '."' | character='Hortense'<br>tail: `“Thank you, Mademoiselle` |
| ' Ins' | ' no' | 4650 | ' no' / ' the' / ' a' / ' not' / ' to' | character='Inspector Bucket'<br>tail: `single moment in the course of this prolonged night, here is` |
| ' John' | ' you' | 136 | ' you' / ' the' / ' any' / ' her' / ' your' | character='John Jarndyce'<br>tail: `se that I have come here to make underhanded charges against` |
| ' C' | ' the' | 19 | ' the' / ' my' / ' her' / ' ' / ' E' | character='Caddy Jellyby'<br>tail: `happened that when I came home from Deal I found a note from` |

### B_possession   (3/50 hit; 47 miss)

| target | model top-1 | rank | top-5 | source / cue |
|--------|-------------|------|-------|--------------|
| ' c' | ' s' | 2 | ' s' / ' h' / ' c' / '\n' / ' b' | noun='candle'<br>tail: `an arm to Ada and an arm to me, and bidding Richard bring a` |
| ' spect' | ' eyes' | 411 | ' eyes' / ' own' / ' father' / ' h' / ' mother' | noun='spectacles'<br>tail: `Pardiggle, who had been regarding him through her` |
| ' letter' | ' death' | 242 | ' death' / ' new' / ' fact' / ' loss' / ' visit' | noun='letter'<br>tail: `er Smallweed smiles in a very ugly way in recognition of the` |
| ' spect' | ' hat' | 89 | ' hat' / ' seat' / ' head' / ' ch' / ' sh' | noun='spectacles'<br>tail: `Tulkinghorn gets up, adjusts his` |
| ' glo' | ' ar' | 80 | ' ar' / ' hands' / ' hand' / ' s' / ' right' | noun='gloves'<br>tail: `urveydrop, standing with his back to the fire and waving his` |
| ' bon' | ' sh' | 31 | ' sh' / ' hair' / 'self' / ' clo' / ' cap' | noun='bonnet'<br>tail: `ng easily anywhere, she perches on a rough bench, unties her` |
| ' letter' | ' first' | 13 | ' first' / ' ' / ' argument' / ' original' / ' deb' | noun='letter'<br>tail: `This was the substance of the` |
| ' hat' | ' h' | 5 | ' h' / ' co' / ' s' / ' clo' / ' cap' | noun='hat'<br>tail: `stops of his own accord, and Sir Leicester, pulling off his` |
| ' spect' | ' head' | 190 | ' head' / ' hands' / ' f' / ' ar' / ' hand' | noun='spectacles'<br>tail: `Rustily drest, with his` |
| ' letter' | ' air' | 68 | ' air' / ' past' / ' way' / ' morning' / ' ' | noun='letter'<br>tail: `aid a visit of a few hours to London, which something in the` |
| ' spect' | ' d' | 93 | ' d' / ' sh' / ' clo' / ' h' / ' ro' | noun='spectacles'<br>tail: `housekeeper at Chesney Wold, has several times taken off her` |
| ' hat' | ' sh' | 6 | ' sh' / ' s' / ' black' / ' p' / ' h' | noun='hat'<br>tail: `leeves and his grey coat, pulls on his black coat, takes his` |
| ' boot' | ' own' | 1322 | ' own' / ' art' / ' old' / ' work' / ' personal' | noun='boots'<br>tail: `of books and papers and in part quite a little museum of his` |
| ' clo' | ' hat' | 3 | ' hat' / ' co' / ' sh' / ' clo' / ' cap' | noun='cloak'<br>tail: `, disregarding my remonstrances, had hurriedly taken off his` |
| ' hat' | '\n' | 33 | '\n' / ' long' / ' s' / ' sh' / ' white' | noun='hat'<br>tail: `A man yet dark and muddy, in long swollen sodden boots and a` |
| ' bon' | ' best' | 64 | ' best' / ' d' / ' usual' / ' g' / '\n' | noun='bonnet'<br>tail: `and joined Miss Jellyby, who was by this time putting on her` |
| ' ' | ' same' | 8 | ' same' / ' following' / ' help' / ' word' / ' most' | noun='umbrella'<br>tail: `Bagnet expresses with the` |
| ' lan' | ' water' | 318 | ' water' / ' river' / '\n' / ' pool' / ' g' | noun='lantern'<br>tail: `The old man stopped, looked hard at us, looked down into the` |
| ' bon' | ' hat' | 106 | ' hat' / ' p' / ' s' / ' sh' / ' g' | noun='bonnet'<br>tail: `, in a womanly sort of manner belonging to the apron and the` |
| ' pur' | ' wallet' | 2 | ' wallet' / ' p' / ' pur' / '.' / ',' | noun='purse'<br>tail: `She draws off her glove to get some money from her` |
| ' book' | ' hand' | 11 | ' hand' / ' hands' / ' p' / ' head' / ' m' | noun='book'<br>tail: `” Having put the letters in his` |
| ' bon' | ' w' | 1162 | ' w' / ' h' / ' s' / ' water' / ' b' | noun='bonnet'<br>tail: `But that there’s the wale, the` |
| ' c' | ' face' | 56 | ' face' / ' back' / ' eyes' / ' head' / ' middle' | noun='candle'<br>tail: `Winking cousins, bat-like in the` |
| ' lan' | ' light' | 671 | ' light' / ' two' / ' same' / ' f' / ' sun' | noun='lantern'<br>tail: `I could see, from my window, the` |
| ' c' | ' time' | 21 | ' time' / ' following' / ' train' / ' same' / ' opportunity' | noun='candle'<br>tail: `Now, Mademoiselle Hortense, let me recommend you to take the` |
| ' book' | ' mind' | 222 | ' mind' / ' eyes' / ' heart' / ' thoughts' / ' head' | noun='book'<br>tail: `He was lost in thought, his` |
| ' pur' | ' point' | 602 | ' point' / ' p' / ' good' / ' note' / ' show' | noun='purse'<br>tail: `fair Dedlock delivers in her youthful manner, while making a` |
| ' watch' | ' scene' | 12 | ' scene' / ' point' / ' side' / ' other' / ' spot' | noun='watch'<br>tail: `der is done; so, now she sees that when he used to be on the` |
| ' book' | ' register' | 5 | ' register' / ' back' / ' file' / ' database' / ' journal' | noun='book'<br>tail: `es softly into the back office, refers to the entries in the` |
| ' clo' | ' deep' | 223 | ' deep' / ' step' / ' long' / ' look' / ' p' | noun='cloak'<br>tail: `Now, you see, George”—he takes a` |
| ' lan' | ' great' | 2929 | ' great' / ' world' / ' gl' / ' people' / ' w' | noun='lantern'<br>tail: `able brief, and outwardly directing his contemplation to the` |
| ' ' | ' hands' | 78 | ' hands' / ' life' / ' own' / ' hand' / ' claim' | noun='umbrella'<br>tail: `ticular lady whose lord is more than suspected of laying his` |
| ' lan' | '\n' | 185 | '\n' / ' box' / ' letter' / ' paper' / ' hand' | noun='lantern'<br>tail: `see, I have so many things here,” he resumed, holding up the` |
| ' book' | ' certain' | 50 | ' certain' / ' letter' / ' report' / ' very' / ' re' | noun='book'<br>tail: `ixth volume of the Philosophical Transactions; and also of a` |
| ' ' | ' hand' | 62 | ' hand' / ' f' / ' hands' / ' sh' / ' right' | noun='umbrella'<br>tail: `ving the trooper a great poke between the shoulders with her` |
| ' ' | ' old' | 12 | ' old' / ' empty' / ' un' / ' over' / ' in' | noun='umbrella'<br>tail: `er quarter of the world—with nothing but a grey cloak and an` |
| ' boot' | ' hands' | 301 | ' hands' / ' face' / ' eyes' / ' heart' / ' hair' | noun='boots'<br>tail: `kind and gentle, and as he stood before the fire warming his` |
| ' c' | ' world' | 40 | ' world' / ' presence' / ' present' / ' dist' / ' place' | noun='candle'<br>tail: `athing lulls or his fixed eyes show any consciousness of the` |
| ' sh' | ' head' | 35 | ' head' / ' face' / ' hair' / ' hand' / ' hands' | noun='shawl'<br>tail: `stal upon the terrace, and a vase upon the pedestal, and her` |
| ' watch' | ' own' | 440 | ' own' / ' name' / ' father' / '\n' / ' master' | noun='watch'<br>tail: `l, Jarndyce,” returned his guest, who seemed to refer to his` |
| ' boot' | ' feet' | 1 | ' feet' / ' boot' / ' sh' / ' le' / ' p' | noun='boots'<br>tail: `Bucket thoughtfully came and warmed the soles of his` |
| ' watch' | ' father' | 319 | ' father' / ' w' / ' bro' / ' own' / ' mother' | noun='watch'<br>tail: `Tulkinghorn, muttering reproof to his` |
| ' pur' | ' w' | 652 | ' w' / ' father' / ' family' / ' job' / ' bro' | noun='purse'<br>tail: `him that during the vacation and while things are slack, his` |
| ' clo' | ' few' | 180 | ' few' / ' quick' / ' c' / ' g' / ' look' | noun='cloak'<br>tail: `o the hotel and wait until he joined me there, so he threw a` |
| ' watch' | ' w' | 6 | ' w' / ' face' / '\n' / ' da' / ' son' | noun='watch'<br>tail: `“Now, little housewife,” said my guardian, looking at his` |
| ' stick' | ' opponent' | 740 | ' opponent' / ' head' / ' fo' / ' f' / ' w' | noun='stick'<br>tail: `Boythorn in a violent burst and stopping to strike his` |
| ' pur' | ' p' | 26 | ' p' / ' notebook' / ' hand' / ' old' / ' pen' | noun='purse'<br>tail: `eorge, my considerate friend,” returns Allan, taking out his` |

### C_plot   (0/50 hit; 50 miss)

| target | model top-1 | rank | top-5 | source / cue |
|--------|-------------|------|-------|--------------|
| 'Ded' | '<NAME>' | 10532 | '<NAME>' / '’' / ' M' / '1' / '\n' | phrase='Lady '->'Dedlock'<br>tail: `impend over yourself.”  “Well, sir?”  “Well, Lady` |
| 'G' | '<NAME>' | 29114 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Mr. '->'Guppy'<br>tail: `his companions are yet midway in theirs, that Mr.` |
| 'Ch' | '<NAME>' | 3885 | '<NAME>' / '1' / '\n' / '2' / '3' | phrase='Court of '->'Chancery'<br>tail: `ernoon some score of members of the High Court of` |
| 'T' | '<NAME>' | 13260 | '<NAME>' / '\n' / ' M' / '1' / ' W' | phrase='Mr. '->'Tulkinghorn'<br>tail: `eicester and his ancestors and his patrimony”—Mr.` |
| 'Bo' | '<NAME>' | 2143 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Mr. '->'Boythorn'<br>tail: `” suggested Richard.  “By my soul,” exclaimed Mr.` |
| 'Ro' | '<NAME>' | 2310 | '<NAME>' / ' M' / ' L' / ' B' / '’' | phrase='Mrs. '->'Rouncewell'<br>tail: `there is any uncommon eye in the case, it is Mrs.` |
| 'T' | '<NAME>' | 29288 | '<NAME>' / '\n' / ' M' / ' H' / ' D' | phrase='Mr. '->'Tulkinghorn'<br>tail: `ct involuntarily starts and falls back. It is Mr.` |
| 'T' | '<NAME>' | 29658 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Mr. '->'Tulkinghorn'<br>tail: `s and ends together than if she had meant it. Mr.` |
| 'Ro' | '<NAME>' | 89 | '<NAME>' / '’' / ' Ro' / '\n' / '1' | phrase='Mrs. '->'Rouncewell'<br>tail: `s he got better.  “Where is your son George, Mrs.` |
| 'Ded' | '<NAME>' | 21245 | '<NAME>' / '1' / '’' / '2' / '3' | phrase='Lady '->'Dedlock'<br>tail: `r by it, and reads, boldly written in each, “Lady` |
| 'Sn' | '<NAME>' | 6010 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Mr. '->'Snagsby'<br>tail: `d to me, he wos!”  As he shuffles downstairs, Mr.` |
| 'G' | '<NAME>' | 23897 | '<NAME>' / '1' / '2' / '\n' / '3' | phrase='Mr. '->'Guppy'<br>tail: `rty! The property! Property!”  Mr. Weevle and Mr.` |
| 'Sn' | '<NAME>' | 294 | '<NAME>' / '\n' / '’' / '1' / '3' | phrase='Mr. '->'Snagsby'<br>tail: `ticularity express, but she knows that Jo was Mr.` |
| 'G' | '<NAME>' | 30777 | '<NAME>' / '\n' / ' M' / ' D' / ' L' | phrase='Mr. '->'Guppy'<br>tail: `ely strolling down a flat country to the sea. Mr.` |
| 'T' | '<NAME>' | 35031 | '<NAME>' / '\n' / '1' / '3' / '2' | phrase='Mr. '->'Tulkinghorn'<br>tail: `s ever. Yet it may be that my Lady fears this Mr.` |
| 'Ded' | '<NAME>' | 16470 | '<NAME>' / '’' / '1' / '2' / '3' | phrase='Lady '->'Dedlock'<br>tail: `aims one morning to the listening earth that Lady` |
| 'Ded' | '<NAME>' | 10661 | '<NAME>' / '2' / '1' / '’' / '3' | phrase='Lady '->'Dedlock'<br>tail: `h a foil in his old-fashioned rusty black to Lady` |
| 'G' | '<NAME>' | 8725 | '<NAME>' / '\n' / '\xa0' / '1' / '\t' | phrase='Mr. '->'Guppy'<br>tail: `ou wouldn’t object to admit that, miss?” said Mr.` |
| 'W' | '’' | 15214 | '’' / '\n' / '1' / '2' / '3' | phrase='Chesney '->'Wold'<br>tail: `e. You have often carried my spare gun at Chesney` |
| 'Sn' | '<NAME>' | 1487 | '<NAME>' / ' and' / '1' / '2' / '\n' | phrase='Mr. '->'Snagsby'<br>tail: `and, my love?”  “At six,” says Mrs. Snagsby.  Mr.` |
| 'K' | '<NAME>' | 10183 | '<NAME>' / '’' / '1' / '\n' / ' M' | phrase='Mr. '->'Krook'<br>tail: `is cough of mild persuasion behind his hand. “Mr.` |
| 'J' | '<NAME>' | 24989 | '<NAME>' / '1' / '2' / 'ate' / '3' | phrase='Caddy '->'Jellyby'<br>tail: `ove with somebody and being ashamed of it.  Caddy` |
| 'Bucket' | '<NAME>' | 29480 | '<NAME>' / '1' / '0' / '2' / '4' | phrase='Inspector '->'Bucket'<br>tail: `hat I’m a-going to do. You know me. I’m Inspector` |
| 'K' | '<NAME>' | 1634 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Mr. '->'Krook'<br>tail: `Run, Flite, run! The nearest doctor! Run!” So Mr.` |
| 'Sn' | '<NAME>' | 5385 | '<NAME>' / '\n' / '1' / '2' / '\xa0' | phrase='Mr. '->'Snagsby'<br>tail: `h a nod.  “About a year and a half ago,” says Mr.` |
| 'Bo' | '<NAME>' | 2958 | '<NAME>' / '\n' / '\xa0' / ' And' / '<EMAIL>' | phrase='Mr. '->'Boythorn'<br>tail: `, who was so tame that he was brought down by Mr.` |
| 'W' | '’' | 16321 | '’' / '1' / '<NAME>' / '\n' / '2' | phrase='Chesney '->'Wold'<br>tail: `knows where to have her. To-day she is at Chesney` |
| 'Jar' | '<NAME>' | 1233 | '<NAME>' / '’' / 'ride' / 'ate' / '1' | phrase='Jarndyce and '->'Jarndyce'<br>tail: `e liberty? Your servant, sir. I know Jarndyce and` |
| 'Bo' | '<NAME>' | 4095 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Mr. '->'Boythorn'<br>tail: `t actions against HIM for trespass,” returned Mr.` |
| 'Bo' | '<NAME>' | 292 | '<NAME>' / '\n' / '1' / ' Bo' / '’' | phrase='Mr. '->'Boythorn'<br>tail: `room, alighted on his master’s head. To hear Mr.` |
| 'Bucket' | '<NAME>' | 24204 | '<NAME>' / '’' / '1' / '\n' / '2' | phrase='Inspector '->'Bucket'<br>tail: `unlikely that you may inquire of me why Inspector` |
| 'W' | '’' | 27453 | '’' / '<NAME>' / '1' / 'ride' / ' are' | phrase='Chesney '->'Wold'<br>tail: `ster and the baronetcy, Sir Leicester and Chesney` |
| 'Jar' | '<NAME>' | 98 | '<NAME>' / '1' / '2' / '3' / '’' | phrase='Jarndyce and '->'Jarndyce'<br>tail: `as promised a new rocking-horse when Jarndyce and` |
| 'J' | '<NAME>' | 7825 | '<NAME>' / ' M' / '\xa0' / ' B' / '’' | phrase='Mrs. '->'Jellyby'<br>tail: `n’t think she did. I therefore supposed that Mrs.` |
| 'Sum' | '<NAME>' | 22297 | '<NAME>' / '’' / '1' / 'ere' / '2' | phrase='Esther '->'Summerson'<br>tail: `nce. I was left in charge of a child named Esther` |
| 'Ded' | '<NAME>' | 26055 | '<NAME>' / '’' / 'ate' / 'rides' / '1' | phrase='Sir Leicester '->'Dedlock'<br>tail: `re to-day.”  “The visitor, my dear! Sir Leicester` |
| 'Ded' | '<NAME>' | 26055 | '<NAME>' / '’' / '1' / '\n' / '2' | phrase='Sir Leicester '->'Dedlock'<br>tail: `o a relation of the family, a great Sir Leicester` |
| 'Sk' | '<NAME>' | 19248 | '<NAME>' / '’' / '1' / '4' / 'ate' | phrase='Harold '->'Skimpole'<br>tail: `y don’t stop, why should I? There you have Harold` |
| 'Ded' | '<NAME>' | 22749 | '<NAME>' / '’' / '1' / '2' / '3' | phrase='Sir Leicester '->'Dedlock'<br>tail: `shall peep in from the outerside.  “Sir Leicester` |
| 'W' | '’' | 35921 | '’' / 'ere' / '<NAME>' / '1' / '2' | phrase='Chesney '->'Wold'<br>tail: `d he might have the good fortune to be at Chesney` |
| 'J' | '<NAME>' | 687 | '<NAME>' / ' J' / '\n' / ' M' / '’' | phrase='Mrs. '->'Jellyby'<br>tail: `ompany.  We duly came back to breakfast, and Mrs.` |
| 'K' | '<NAME>' | 11262 | '<NAME>' / '1' / '\n' / ' Job' / '3' | phrase='Mr. '->'Krook'<br>tail: `r. He soon returns with the intelligence that Mr.` |
| 'K' | '<NAME>' | 428 | '<NAME>' / '\n' / '1' / '’' / ' K' | phrase='Mr. '->'Krook'<br>tail: `How do you do, sir? You are looking charming, Mr.` |
| 'Ded' | '<NAME>' | 23726 | '<NAME>' / '’' / '1' / '2' / ' to' | phrase='Sir Leicester '->'Dedlock'<br>tail: `hundred.  “That is, I am deputed by Sir Leicester` |
| 'Ro' | '<NAME>' | 1950 | '<NAME>' / '1' / '\xa0' / ' M' / '\n' | phrase='Mrs. '->'Rouncewell'<br>tail: `e for ever and a day.  “He shall have,” says Mrs.` |
| 'Ro' | '<NAME>' | 67 | '<NAME>' / '’' / ' Ro' / '\n' / '1' | phrase='Mrs. '->'Rouncewell'<br>tail: `and with a profound curtsy.  “How do you do, Mrs.` |
| 'Ro' | '’' | 16923 | '’' / '<NAME>' / 'ate' / 'rides' / '1' | phrase='George '->'Rouncewell'<br>tail: `run glistening down her sun-brown face.  “George` |
| 'J' | '<NAME>' | 6623 | '<NAME>' / '\n' / '1' / ' and' / ' M' | phrase='Mrs. '->'Jellyby'<br>tail: `r that he called the matrimonial alliance of Mrs.` |
| 'J' | '<NAME>' | 1994 | '<NAME>' / '\n' / ' J' / ' Jar' / ' M' | phrase='Mrs. '->'Jellyby'<br>tail: `quite well.  “Why, not quite, my dear,” said Mrs.` |
| 'Ro' | '<NAME>' | 12499 | '<NAME>' / '’' / '1' / '2' / '\n' | phrase='George '->'Rouncewell'<br>tail: `ed to remember you.”  “When I look at you, George` |

### D_code   (11/50 hit; 39 miss)

| target | model top-1 | rank | top-5 | source / cue |
|--------|-------------|------|-------|--------------|
| 'numpy' | ' java' | 980 | ' java' / ' com' / ' {' / ' numpy' / ' os' | prefix: `'import '` |
| 'os' | ' java' | 3507 | ' java' / ' com' / ' {' / ' numpy' / ' os' | prefix: `'import '` |
| 'default' | ' defaultdict' | 135 | ' defaultdict' / ' Counter' / ' name' / ' deque' / ' OrderedDict' | prefix: `'from collections import '` |
| 'List' | ' List' | 179 | ' List' / ' Dict' / ' Optional' / ' Tuple' / ' Union' | prefix: `'from typing import '` |
| '*' | ' *' | 435 | ' *' / ' **' / '1' / '\t' / '0' | prefix: `'def __init__(self, '` |
| 'range' | '1' | 443 | '1' / '0' / '2' / '3' / '5' | prefix: `'for i in '` |
| "'" | " '" | 6 | " '" / ' "' / '0' / ' mode' / '1' | prefix: `'with open(path, '` |
| 'None' | '0' | 21590 | '0' / '1' / '2' / '3' / '4' | prefix: `'return '` |
| 'ValueError' | ' #' | 555 | ' #' / '4' / '1' / '\n' / '2' | prefix: `'raise '` |
| '_' | 'get' | 10826 | 'get' / 'name' / 'state' / 'data' / 'x' | prefix: `'self.'` |
| 'Logger' | '_' | 7 | '_' / 'Formatter' / 'logger' / 'Level' / "('" | prefix: `'logger = logging.get'` |
| 'Linear' | 'Module' | 5 | 'Module' / 'functional' / 'utils' / 'modules' / 'init' | prefix: `'torch.nn.'` |
| 'plot' | 'show' | 1 | 'show' / 'plot' / 'xlabel' / 'legend' / 'title' | prefix: `'plt.'` |
| 'run' | 'get' | 4 | 'get' / 'subprocess' / 'argv' / 'py' / 'run' | prefix: `'subprocess.'` |
| 'compile' | 'get' | 7 | 'get' / 'com' / 'js' / 'add' / 'set' | prefix: `'re.'` |
| "'" | 'os' | 6220 | 'os' / 'sep' / 'delimiter' / 'None' / 'separator' | prefix: `"'.split("` |
| 'name' | 'ql' | 29392 | 'ql' / 'ime' / 'irc' / 'ico' / 'energ' | prefix: `'@property\ndef '` |
| 'x' | '\t' | 29358 | '\t' / ' import' / ' from' / ' #' / ' print' | prefix: `'try:\n    '` |
| 'e' | ' e' | 1 | ' e' / 'e' / ' Exception' / ' exception' / ' error' | prefix: `'except Exception as '` |
| 'x' | '1' | 40332 | '1' / '0' / '2' / '3' / '4' | prefix: `'assert '` |
| 'x' | '1' | 35878 | '1' / '0' / '2' / '3' / '5' | prefix: `'yield '` |
| 'main' | ' get' | 4806 | ' get' / ' _' / 'irc' / ' test' / ' async' | prefix: `'async def '` |
| 'fixture' | 'mark' | 2 | 'mark' / 'runner' / 'fixture' / 'test' / 'run' | prefix: `'pytest.'` |
| 'Optional' | 'types' | 328 | 'types' / 'module' / 'h' / 'T' / 'js' | prefix: `'typing.'` |
| 'file' | 'dirname' | 2 | 'dirname' / 'FILE' / 'file' / 'DIR' / '_' | prefix: `'Path(__'` |
| 'info' | 'Log' | 6 | 'Log' / 'log' / 'Logger' / 'error' / 'getLogger' | prefix: `'logging.'` |
| 'dataclass' | 'py' | 23 | 'py' / 'class' / 'dat' / 'h' / '\n' | prefix: `'dataclasses.'` |
| 'partial' | 'wrap' | 1 | 'wrap' / 'partial' / 'lazy' / 'reduce' / 're' | prefix: `'functools.'` |
| 'chain' | 'product' | 4 | 'product' / 'iter' / 'hasNext' / 'count' / 'chain' | prefix: `'itertools.'` |
| 'Ordered' | 'Generic' | 62 | 'Generic' / 'sort' / 'utils' / 'py' / 'map' | prefix: `'collections.'` |
| 'ascii' | 'IsNullOrEmpty' | 142 | 'IsNullOrEmpty' / 'Format' / 'Empty' / 'h' / 'format' | prefix: `'string.'` |
| 'pi' | 'h' | 2 | 'h' / 'random' / 'pi' / 'sqrt' / 'sin' | prefix: `'math.'` |
| 'sleep' | 'time' | 1 | 'time' / 'sleep' / 'h' / 'Time' / 'now' | prefix: `'time.'` |
| 'seed' | 'random' | 3 | 'random' / 'randint' / 'rand' / 'seed' / 'choice' | prefix: `'random.'` |
| 'sha' | 'md' | 1 | 'md' / 'sha' / 'h' / 'hash' / 'MD' | prefix: `'hashlib.'` |
| 'b' | 'decode' | 46 | 'decode' / 'encode' / 'base' / 'h' / 'js' | prefix: `'base64.'` |
| 'request' | 'parse' | 1 | 'parse' / 'request' / 'http' / 'error' / 'url' | prefix: `'urllib.'` |
| 'Flask' | 'app' | 16 | 'app' / 'py' / 'db' / 'json' / 'session' | prefix: `'flask.'` |
| 'Linear' | 'model' | 6 | 'model' / 'predict' / 'Model' / 'utils' / 'Module' | prefix: `'nn.'` |


## C2 RAG (source passage in context)

### A_character   (46/50 hit; 4 miss)

| target | model top-1 | rank | top-5 | source / cue |
|--------|-------------|------|-------|--------------|
| ' Ge' | ' S' | 1 | ' S' / ' Ge' / ' ' / ' s' / ' and' | character='George Rouncewell'<br>tail: `“All is still in readiness,` |
| ' Ge' | ' ' | 1 | ' ' / ' Ge' / '\n' / '\n\n' / ' **' | character='George Rouncewell'<br>tail: `Very familiar to him, as he said himself some hours ago,` |
| ' Richard' | 'Rich' | 555 | 'Rich' / '<NAME>' / 'Car' / 'I' / 'The' | character='Richard Carstone'<br>tail: `YOUR name now will be—” “` |
| ' Har' | 'Har' | 126 | 'Har' / 'I' / 'The' / '<NAME>' / 'My' | character='Harold Skimpole'<br>tail: `Skimpole, “to this effect: ‘` |

### B_possession   (50/50 hit; 0 miss)

_All items hit top-1._

### C_plot   (0/50 hit; 50 miss)

| target | model top-1 | rank | top-5 | source / cue |
|--------|-------------|------|-------|--------------|
| 'Ded' | '<NAME>' | 97 | '<NAME>' / '1' / '\n' / '’' / '2' | phrase='Lady '->'Dedlock'<br>tail: `impend over yourself.”  “Well, sir?”  “Well, Lady` |
| 'G' | '<NAME>' | 878 | '<NAME>' / '\n' / ' G' / '<file_sep>' / '1' | phrase='Mr. '->'Guppy'<br>tail: `his companions are yet midway in theirs, that Mr.` |
| 'Ch' | '<NAME>' | 723 | '<NAME>' / '\n' / '1' / '<file_sep>' / '2' | phrase='Court of '->'Chancery'<br>tail: `ernoon some score of members of the High Court of` |
| 'T' | '<NAME>' | 645 | '<NAME>' / '1' / '’' / '2' / '3' | phrase='Mr. '->'Tulkinghorn'<br>tail: `eicester and his ancestors and his patrimony”—Mr.` |
| 'Bo' | '<NAME>' | 91 | '<NAME>' / '1' / '’' / '2' / '3' | phrase='Mr. '->'Boythorn'<br>tail: `” suggested Richard.  “By my soul,” exclaimed Mr.` |
| 'Ro' | '<NAME>' | 57 | '<NAME>' / '1' / '\n' / '’' / '0' | phrase='Mrs. '->'Rouncewell'<br>tail: `there is any uncommon eye in the case, it is Mrs.` |
| 'T' | '<NAME>' | 625 | '<NAME>' / '1' / '’' / '2' / '0' | phrase='Mr. '->'Tulkinghorn'<br>tail: `ct involuntarily starts and falls back. It is Mr.` |
| 'T' | '<NAME>' | 544 | '<NAME>' / '’' / '1' / '\n' / '2' | phrase='Mr. '->'Tulkinghorn'<br>tail: `s and ends together than if she had meant it. Mr.` |
| 'Ro' | '<NAME>' | 30 | '<NAME>' / '\n' / '1' / '’' / ' Ro' | phrase='Mrs. '->'Rouncewell'<br>tail: `s he got better.  “Where is your son George, Mrs.` |
| 'Ded' | '<NAME>' | 196 | '<NAME>' / ' D' / '1' / '’' / '2' | phrase='Lady '->'Dedlock'<br>tail: `r by it, and reads, boldly written in each, “Lady` |
| 'Sn' | '<NAME>' | 52 | '<NAME>' / '’' / '1' / '2' / '3' | phrase='Mr. '->'Snagsby'<br>tail: `d to me, he wos!”  As he shuffles downstairs, Mr.` |
| 'G' | '<NAME>' | 478 | '<NAME>' / ' G' / '1' / '’' / '\n' | phrase='Mr. '->'Guppy'<br>tail: `rty! The property! Property!”  Mr. Weevle and Mr.` |
| 'Sn' | '<NAME>' | 42 | '<NAME>' / '\n' / '1' / '’' / '2' | phrase='Mr. '->'Snagsby'<br>tail: `ticularity express, but she knows that Jo was Mr.` |
| 'G' | '<NAME>' | 695 | '<NAME>' / '\n' / '1' / ' G' / '’' | phrase='Mr. '->'Guppy'<br>tail: `ely strolling down a flat country to the sea. Mr.` |
| 'T' | '<NAME>' | 797 | '<NAME>' / '\n' / '<file_sep>' / '1' / ' T' | phrase='Mr. '->'Tulkinghorn'<br>tail: `s ever. Yet it may be that my Lady fears this Mr.` |
| 'Ded' | '<NAME>' | 83 | '<NAME>' / ' D' / '\n' / '’' / '1' | phrase='Lady '->'Dedlock'<br>tail: `aims one morning to the listening earth that Lady` |
| 'Ded' | '<NAME>' | 64 | '<NAME>' / ' D' / '’' / '\n' / '1' | phrase='Lady '->'Dedlock'<br>tail: `h a foil in his old-fashioned rusty black to Lady` |
| 'G' | '<NAME>' | 939 | '<NAME>' / '’' / '1' / '2' / ' G' | phrase='Mr. '->'Guppy'<br>tail: `ou wouldn’t object to admit that, miss?” said Mr.` |
| 'W' | '<NAME>' | 379 | '<NAME>' / '\n' / '1' / '’' / '2' | phrase='Chesney '->'Wold'<br>tail: `e. You have often carried my spare gun at Chesney` |
| 'Sn' | '<NAME>' | 32 | '<NAME>' / '1' / '\n' / '6' / ' Sn' | phrase='Mr. '->'Snagsby'<br>tail: `and, my love?”  “At six,” says Mrs. Snagsby.  Mr.` |
| 'K' | '<NAME>' | 168 | '<NAME>' / '\n' / '1' / '’' / '0' | phrase='Mr. '->'Krook'<br>tail: `is cough of mild persuasion behind his hand. “Mr.` |
| 'J' | '<NAME>' | 370 | '<NAME>' / ' J' / '’' / '1' / '\n' | phrase='Caddy '->'Jellyby'<br>tail: `ove with somebody and being ashamed of it.  Caddy` |
| 'Bucket' | '<NAME>' | 40 | '<NAME>' / '1' / '’' / ' Bucket' / '6' | phrase='Inspector '->'Bucket'<br>tail: `hat I’m a-going to do. You know me. I’m Inspector` |
| 'K' | '<NAME>' | 285 | '<NAME>' / '1' / '\n' / '’' / '2' | phrase='Mr. '->'Krook'<br>tail: `Run, Flite, run! The nearest doctor! Run!” So Mr.` |
| 'Sn' | '<NAME>' | 23 | '<NAME>' / '’' / ' Sn' / '1' / '\n' | phrase='Mr. '->'Snagsby'<br>tail: `h a nod.  “About a year and a half ago,” says Mr.` |
| 'Bo' | '<NAME>' | 160 | '<NAME>' / '1' / '’' / '2' / '0' | phrase='Mr. '->'Boythorn'<br>tail: `, who was so tame that he was brought down by Mr.` |
| 'W' | '<NAME>' | 397 | '<NAME>' / '1' / '’' / ' W' / '<file_sep>' | phrase='Chesney '->'Wold'<br>tail: `knows where to have her. To-day she is at Chesney` |
| 'Jar' | '<NAME>' | 10 | '<NAME>' / '’' / ' Jar' / '1' / 'ard' | phrase='Jarndyce and '->'Jarndyce'<br>tail: `e liberty? Your servant, sir. I know Jarndyce and` |
| 'Bo' | '<NAME>' | 33 | '<NAME>' / '1' / ' Bo' / '\n' / '0' | phrase='Mr. '->'Boythorn'<br>tail: `t actions against HIM for trespass,” returned Mr.` |
| 'Bo' | '<NAME>' | 97 | '<NAME>' / '1' / '2' / '’' / '0' | phrase='Mr. '->'Boythorn'<br>tail: `room, alighted on his master’s head. To hear Mr.` |
| 'Bucket' | '<NAME>' | 30 | '<NAME>' / '\n' / '1' / ' Bucket' / '’' | phrase='Inspector '->'Bucket'<br>tail: `unlikely that you may inquire of me why Inspector` |
| 'W' | '<NAME>' | 184 | '<NAME>' / ' W' / '’' / '1' / '\n' | phrase='Chesney '->'Wold'<br>tail: `ster and the baronetcy, Sir Leicester and Chesney` |
| 'Jar' | '<NAME>' | 32 | '<NAME>' / '1' / '\n' / '2' / '3' | phrase='Jarndyce and '->'Jarndyce'<br>tail: `as promised a new rocking-horse when Jarndyce and` |
| 'J' | '<NAME>' | 311 | '<NAME>' / '’' / '1' / '\n' / ' J' | phrase='Mrs. '->'Jellyby'<br>tail: `n’t think she did. I therefore supposed that Mrs.` |
| 'Sum' | '<NAME>' | 50 | '<NAME>' / '\n' / ' Sum' / '1' / '’' | phrase='Esther '->'Summerson'<br>tail: `nce. I was left in charge of a child named Esther` |
| 'Ded' | '<NAME>' | 91 | '<NAME>' / '\n' / '<file_sep>' / ' D' / '’' | phrase='Sir Leicester '->'Dedlock'<br>tail: `re to-day.”  “The visitor, my dear! Sir Leicester` |
| 'Ded' | '<NAME>' | 430 | '<NAME>' / '\n' / ' D' / '1' / '\n\n\n\n' | phrase='Sir Leicester '->'Dedlock'<br>tail: `o a relation of the family, a great Sir Leicester` |
| 'Sk' | '<NAME>' | 31 | '<NAME>' / '\n' / '1' / ' Sk' / '’' | phrase='Harold '->'Skimpole'<br>tail: `y don’t stop, why should I? There you have Harold` |
| 'Ded' | '<NAME>' | 138 | '<NAME>' / '1' / ' D' / '\n' / '’' | phrase='Sir Leicester '->'Dedlock'<br>tail: `shall peep in from the outerside.  “Sir Leicester` |
| 'W' | '<NAME>' | 693 | '<NAME>' / '1' / '2' / '\n' / '’' | phrase='Chesney '->'Wold'<br>tail: `d he might have the good fortune to be at Chesney` |
| 'J' | '<NAME>' | 768 | '<NAME>' / '\n' / '1' / '’' / '2' | phrase='Mrs. '->'Jellyby'<br>tail: `ompany.  We duly came back to breakfast, and Mrs.` |
| 'K' | '<NAME>' | 112 | '<NAME>' / '\n' / '1' / '’' / '2' | phrase='Mr. '->'Krook'<br>tail: `r. He soon returns with the intelligence that Mr.` |
| 'K' | '<NAME>' | 593 | '<NAME>' / '\n' / '1' / '<file_sep>' / '’' | phrase='Mr. '->'Krook'<br>tail: `How do you do, sir? You are looking charming, Mr.` |
| 'Ded' | '<NAME>' | 63 | '<NAME>' / '\n' / ' D' / '’' / '1' | phrase='Sir Leicester '->'Dedlock'<br>tail: `hundred.  “That is, I am deputed by Sir Leicester` |
| 'Ro' | '<NAME>' | 34 | '<NAME>' / '1' / '\n' / ' Ro' / '0' | phrase='Mrs. '->'Rouncewell'<br>tail: `e for ever and a day.  “He shall have,” says Mrs.` |
| 'Ro' | '<NAME>' | 40 | '<NAME>' / '1' / '\n' / ' Ro' / '’' | phrase='Mrs. '->'Rouncewell'<br>tail: `and with a profound curtsy.  “How do you do, Mrs.` |
| 'Ro' | '<NAME>' | 112 | '<NAME>' / '’' / '1' / ' Ro' / '\n' | phrase='George '->'Rouncewell'<br>tail: `run glistening down her sun-brown face.  “George` |
| 'J' | '<NAME>' | 369 | '<NAME>' / '\n' / '1' / '\n\n\n\n' / ' J' | phrase='Mrs. '->'Jellyby'<br>tail: `r that he called the matrimonial alliance of Mrs.` |
| 'J' | '<NAME>' | 620 | '<NAME>' / '1' / '\n' / '’' / '2' | phrase='Mrs. '->'Jellyby'<br>tail: `quite well.  “Why, not quite, my dear,” said Mrs.` |
| 'Ro' | '<NAME>' | 48 | '<NAME>' / '\n' / ' Ro' / '<file_sep>' / '1' | phrase='George '->'Rouncewell'<br>tail: `ed to remember you.”  “When I look at you, George` |

### D_code   (27/50 hit; 23 miss)

| target | model top-1 | rank | top-5 | source / cue |
|--------|-------------|------|-------|--------------|
| 'numpy' | ' matplotlib' | 2301 | ' matplotlib' / ' torch' / ' pandas' / ' tensorflow' / ' scipy' | prefix: `'import '` |
| 'os' | 'irc' | 7752 | 'irc' / ' numpy' / ' matplotlib' / ' pandas' / ' torch' | prefix: `'import '` |
| 'default' | ' Counter' | 140 | ' Counter' / ' defaultdict' / ' deque' / ' name' / ' OrderedDict' | prefix: `'from collections import '` |
| 'List' | ' Optional' | 627 | ' Optional' / ' Tuple' / ' List' / ' Union' / ' Dict' | prefix: `'from typing import '` |
| '*' | '\n' | 9267 | '\n' / ' *' / '0' / '1' / ' **' | prefix: `'def __init__(self, '` |
| 'range' | '1' | 200 | '1' / '0' / '\n' / ' range' / '2' | prefix: `'for i in '` |
| "'" | " '" | 277 | " '" / '’' / '\n' / '1' / '<|endoftext|>' | prefix: `'with open(path, '` |
| 'None' | '0' | 1088 | '0' / '1' / '2' / '\n' / '3' | prefix: `'return '` |
| 'ValueError' | '\n' | 64 | '\n' / ' ValueError' / '1' / ' #' / '0' | prefix: `'raise '` |
| '_' | '\n' | 20533 | '\n' / 'set' / 'get' / 'add' / 'is' | prefix: `'self.'` |
| 'Logger' | '_' | 9 | '_' / 'logger' / 'Formatter' / 'Level' / "('" | prefix: `'logger = logging.get'` |
| 'dumps' | 'loads' | 1 | 'loads' / 'dumps' / 'dump' / 'load' / 'JSON' | prefix: `'json.'` |
| "'" | '\n' | 1125 | '\n' / '\n   ' / 'str' / 'line' / 'string' | prefix: `"'.split("` |
| 'name' | 'ico' | 12912 | 'ico' / 'irc' / '\n' / 'energies' / 'ime' | prefix: `'@property\ndef '` |
| 'x' | '1' | 1734 | '1' / '2' / '4' / '5' / '3' | prefix: `'try:\n    '` |
| 'e' | ' e' | 308 | ' e' / ' error' / ' err' / 'ere' / ' ex' | prefix: `'except Exception as '` |
| 'x' | '1' | 9347 | '1' / '0' / '2' / '3' / '5' | prefix: `'assert '` |
| 'x' | '1' | 2584 | '1' / '2' / '3' / '4' / '5' | prefix: `'yield '` |
| 'main' | '\n' | 2504 | '\n' / ' main' / '<file_sep>' / '1' / ' test' | prefix: `'async def '` |
| 'fixture' | 'import' | 2 | 'import' / 'register' / 'fixture' / 'mark' / 'main' | prefix: `'pytest.'` |
| 'info' | 'basic' | 1 | 'basic' / 'info' / 'config' / 'getLogger' / 'debug' | prefix: `'logging.'` |
| 'chain' | 'combin' | 2 | 'combin' / 'compress' / 'chain' / 'count' / 'product' | prefix: `'itertools.'` |
| 'pi' | 'sin' | 1 | 'sin' / 'pi' / 'sqrt' / 'cos' / 'log' | prefix: `'math.'` |


## C3 D2L adapter only (no in-context content)

### A_character   (3/50 hit; 47 miss)

| target | model top-1 | rank | top-5 | source / cue |
|--------|-------------|------|-------|--------------|
| ' T' | ' ' | 11 | ' ' / '\n' / ' L' / ' M' / ' B' | character='Tulkinghorn'<br>tail: `t same moment, there happens to be an old man of the name of` |
| ' All' | '\n' | 187 | '\n' / ' M' / ' L' / ' ' / ' D' | character='Allan Woodcourt'<br>tail: `The broken footway is so narrow that when` |
| ' Est' | ' L' | 474 | ' L' / ' M' / ' Richard' / ' G' / ' D' | character='Esther Summerson'<br>tail: `here I was a day boarder, and although they called me little` |
| ' Ge' | '\n' | 32 | '\n' / ' God' / ' H' / ' C' / ' Good' | character='George Rouncewell'<br>tail: `“Good heaven, and it is really` |
| ' Ada' | '\n' | 59 | '\n' / ' Sum' / ' D' / ' ' / ' Small' | character='Ada Clare'<br>tail: `Richard Carstone and of Miss` |
| ' Richard' | '\n' | 15 | '\n' / ' was' / ' ' / ' L' / 'was' | character='Richard Carstone'<br>tail: `gentleman was her distant cousin, she told me, and his name` |
| ' S' | '\n' | 43 | '\n' / ' ' / ' L' / ' God' / ' A' | character='Sir Leicester'<br>tail: `sper still goes about that she had not even family; howbeit,` |
| ' K' | '\n' | 41 | '\n' / ' L' / ' M' / ' Char' / ' Ge' | character='Krook'<br>tail: `raciously observed to him before passing out, “That will do,` |
| ' Har' | ' God' | 459 | ' God' / ' L' / 'ting' / ' Richard' / ' Ch' | character='Harold Skimpole'<br>tail: `, holiness, commerce, trade, any object you prefer; only—let` |
| ' H' | '\n' | 21 | '\n' / ' L' / ' B' / ' C' / ' D' | character='Hortense'<br>tail: `In short, it is such an admirable thing that Mademoiselle` |
| ' Ins' | '\n' | 158 | '\n' / ' Ge' / ' L' / ' C' / ' ' | character='Inspector Bucket'<br>tail: `ng that may appear to be disagreeable in this, for my name’s` |
| ' John' | ' God' | 40 | ' God' / ' C' / ' Richard' / ' Ch' / ' M' | character='John Jarndyce'<br>tail: `” “Since you refer so immediately to` |
| ' C' | '\n' | 4 | '\n' / ' M' / ' ' / ' J' / ' C' | character='Caddy Jellyby'<br>tail: `So I thought one day when I went to London to meet` |
| ' T' | ' M' | 22 | ' M' / '\n' / ' Ch' / ' D' / ' God' | character='Tulkinghorn'<br>tail: `now observes from his couch that man told him ya’as’dy that` |
| ' All' | ' M' | 39 | ' M' / ' Ada' / ' Richard' / ' L' / ' D' | character='Allan Woodcourt'<br>tail: `ome excrescence produced there in neglect and impurity, that` |
| ' Est' | '\n' | 545 | '\n' / 'our' / 'us' / ' ' / ' D' | character='Esther Summerson'<br>tail: `We are, Madam, Your obedt Servts, Kenge and Carboy Miss` |
| ' Ge' | ' All' | 5 | ' All' / ' L' / '\n' / ' M' / ' God' | character='George Rouncewell'<br>tail: `“All is still in readiness,` |
| ' Richard' | '\n' | 9 | '\n' / ' Ada' / ' L' / ' M' / ' ' | character='Richard Carstone'<br>tail: `and released her, and then he spoke for a minute or two with` |
| ' S' | '\n' | 28 | '\n' / ' M' / ' ' / ' L' / ' J' | character='Sir Leicester'<br>tail: `n air of prescription about him which is always agreeable to` |
| ' K' | ' Tom' | 20 | ' Tom' / ' C' / ' M' / ' L' / ' Richard' | character='Krook'<br>tail: `as is announced in paint, to all whom it may concern, by one` |
| ' Har' | '\n' | 348 | '\n' / ' L' / ' D' / ' C' / ' J' | character='Harold Skimpole'<br>tail: `Then, for heaven’s sake, having` |
| ' H' | '\n' | 14 | '\n' / ' M' / ' Ada' / ' L' / ' Ch' | character='Hortense'<br>tail: `to attend,” says my Lady then, addressing the reflection of` |
| ' Ins' | '\n' | 154 | '\n' / ' Ada' / ' Richard' / ' L' / ' Char' | character='Inspector Bucket'<br>tail: `y about admitting of it, you tell her that it’s no use, that` |
| ' John' | '\n' | 55 | '\n' / ' M' / ' L' / ' C' / ' God' | character='John Jarndyce'<br>tail: `I suppose your loyalty to` |
| ' C' | ' Ch' | 7 | ' Ch' / '\n' / ' ' / ' Richard' / ' M' | character='Caddy Jellyby'<br>tail: `At last we came to Soho Square, where` |
| ' T' | '\n' | 2 | '\n' / ' L' / ' T' / ' B' / ' H' | character='Tulkinghorn'<br>tail: `ined to add the last great secret to the many secrets of the` |
| ' All' | ' Ada' | 207 | ' Ada' / ' him' / ' L' / ' her' / ' C' | character='Allan Woodcourt'<br>tail: `s, a farewell to her, and takes his creeping way along after` |
| ' Est' | ' M' | 56 | ' M' / ' L' / ' G' / ' Ch' / ' C' | character='Esther Summerson'<br>tail: `And yet I—I, little` |
| ' Ge' | '\n' | 94 | '\n' / ' L' / ' ' / ' M' / ' Ada' | character='George Rouncewell'<br>tail: `Very familiar to him, as he said himself some hours ago,` |
| ' Richard' | 'You' | 6815 | 'You' / 'Yes' / 'Now' / 'Not' / 'Your' | character='Richard Carstone'<br>tail: `YOUR name now will be—” “` |
| ' S' | ' M' | 23 | ' M' / ' J' / ' L' / ' C' / ' ' | character='Sir Leicester'<br>tail: `“Better now,” quoth` |
| ' K' | '\n' | 18 | '\n' / ' Ada' / ' M' / ' D' / ' C' | character='Krook'<br>tail: `The welcome light soon shines upon the wall, as` |
| ' Har' | ' M' | 276 | ' M' / '\n' / ' God' / ' L' / ' B' | character='Harold Skimpole'<br>tail: `Mankind will surely not deny to` |
| ' H' | '\n' | 20 | '\n' / ' L' / ' M' / ' C' / ' D' | character='Hortense'<br>tail: `t, a peaceful figure too in the landscape, went Mademoiselle` |
| ' Ins' | '\n' | 299 | '\n' / ' ' / ' L' / ' M' / ' Her' | character='Inspector Bucket'<br>tail: `Put it to her ladyship, if you think it right, from` |
| ' John' | ' Ge' | 74 | ' Ge' / ' Ch' / ' M' / '\n' / ' Bag' | character='John Jarndyce'<br>tail: `” “There you come back to` |
| ' C' | ' ' | 3 | ' ' / ' L' / ' M' / ' C' / '\n' | character='Caddy Jellyby'<br>tail: `r was a greater imposter than I with a blinder follower than` |
| ' T' | '\n' | 12 | '\n' / ' ' / ' C' / ' Ge' / ' M' | character='Tulkinghorn'<br>tail: `en a murder in Lincoln’s Inn Fields—gentleman of the name of` |
| ' All' | ' Jo' | 83 | ' Jo' / '\n' / ' L' / ' Ch' / ' J' | character='Allan Woodcourt'<br>tail: `CHAPTER XLVII Jo’s Will As` |
| ' Est' | ' ' | 108 | ' ' / '\n' / ' M' / ' J' / ' L' | character='Esther Summerson'<br>tail: `I was left in charge of a child named` |
| ' S' | '\n' | 36 | '\n' / ' God' / 'getting' / ' Richard' / ' L' | character='Sir Leicester'<br>tail: `re is any superabundant life of imagination on the spot, for` |
| ' K' | '\n' | 17 | '\n' / ' M' / ' ' / ' Ge' / ' J' | character='Krook'<br>tail: `I don’t know,” says` |
| ' Har' | 'You' | 20811 | 'You' / 'There' / 'This' / 'The' / 'We' | character='Harold Skimpole'<br>tail: `Skimpole, “to this effect: ‘` |
| ' H' | '\n' | 31 | '\n' / ' D' / ' Jar' / ' B' / ' C' | character='Hortense'<br>tail: `“Thank you, Mademoiselle` |
| ' Ins' | ' Richard' | 369 | ' Richard' / ' M' / ' Ch' / ' D' / ' Char' | character='Inspector Bucket'<br>tail: `single moment in the course of this prolonged night, here is` |
| ' John' | '\n' | 36 | '\n' / ' Richard' / ' Est' / ' C' / ' M' | character='John Jarndyce'<br>tail: `se that I have come here to make underhanded charges against` |
| ' C' | ' L' | 9 | ' L' / ' Est' / ' ' / ' Ch' / '\n' | character='Caddy Jellyby'<br>tail: `happened that when I came home from Deal I found a note from` |

### B_possession   (0/50 hit; 50 miss)

| target | model top-1 | rank | top-5 | source / cue |
|--------|-------------|------|-------|--------------|
| ' c' | '\n' | 43 | '\n' / ' Ch' / ' G' / ' D' / ' ' | noun='candle'<br>tail: `an arm to Ada and an arm to me, and bidding Richard bring a` |
| ' spect' | '\n' | 3356 | '\n' / ' L' / ' D' / ' C' / 'self' | noun='spectacles'<br>tail: `Pardiggle, who had been regarding him through her` |
| ' letter' | '\n' | 4080 | '\n' / ' L' / ' M' / ' G' / ' Ch' | noun='letter'<br>tail: `er Smallweed smiles in a very ugly way in recognition of the` |
| ' spect' | '\n' | 710 | '\n' / ' Bucket' / ' T' / ' D' / ' C' | noun='spectacles'<br>tail: `Tulkinghorn gets up, adjusts his` |
| ' glo' | '\n' | 1450 | '\n' / ' Ch' / ' T' / ' C' / ' D' | noun='gloves'<br>tail: `urveydrop, standing with his back to the fire and waving his` |
| ' bon' | 'self' | 1022 | 'self' / ' D' / ' Ch' / ' C' / ' Ro' | noun='bonnet'<br>tail: `ng easily anywhere, she perches on a rough bench, unties her` |
| ' letter' | '\n' | 803 | '\n' / ' L' / ' C' / ' J' / ' B' | noun='letter'<br>tail: `This was the substance of the` |
| ' letter' | '\n' | 1 | '\n' / ' letter' / ' good' / ' sweet' / ' fine' | noun='letter'<br>tail: `ppy and grateful, and with her approval I had written such a` |
| ' hat' | ' C' | 15 | ' C' / ' D' / '\n' / ' Bl' / ' G' | noun='hat'<br>tail: `stops of his own accord, and Sir Leicester, pulling off his` |
| ' spect' | '\n' | 1634 | '\n' / ' T' / ' L' / ' D' / ' C' | noun='spectacles'<br>tail: `Rustily drest, with his` |
| ' letter' | '\n' | 1319 | '\n' / ' M' / ' L' / ' manner' / ' English' | noun='letter'<br>tail: `aid a visit of a few hours to London, which something in the` |
| ' spect' | '\n' | 2591 | '\n' / 'self' / ' L' / ' Bag' / ' D' | noun='spectacles'<br>tail: `housekeeper at Chesney Wold, has several times taken off her` |
| ' hat' | '\n' | 67 | '\n' / ' D' / ' V' / ' G' / ' T' | noun='hat'<br>tail: `leeves and his grey coat, pulls on his black coat, takes his` |
| ' boot' | '\n' | 3320 | '\n' / ' own' / ' D' / ' L' / ' C' | noun='boots'<br>tail: `of books and papers and in part quite a little museum of his` |
| ' clo' | '\n' | 97 | '\n' / ' G' / ' Bl' / ' L' / ' T' | noun='cloak'<br>tail: `, disregarding my remonstrances, had hurriedly taken off his` |
| ' hat' | '\n' | 209 | '\n' / '\n ' / ' B' / ' D' / ' L' | noun='hat'<br>tail: `A man yet dark and muddy, in long swollen sodden boots and a` |
| ' hat' | '\n' | 60 | '\n' / ' G' / ' L' / ' D' / ' C' | noun='hat'<br>tail: `George, entirely assenting, puts on his` |
| ' clo' | '\n' | 704 | '\n' / ' Ch' / ' L' / ' D' / ' G' | noun='cloak'<br>tail: `d in a moment, and they took me between them, wrapped in the` |
| ' bon' | '\n' | 1289 | '\n' / ' L' / ' French' / ' D' / ' M' | noun='bonnet'<br>tail: `and joined Miss Jellyby, who was by this time putting on her` |
| ' ' | '\n' | 17 | '\n' / ' Ch' / ' slight' / ' L' / ' Que' | noun='umbrella'<br>tail: `Bagnet expresses with the` |
| ' lan' | '\n' | 1330 | '\n' / ' Ch' / ' G' / ' D' / ' Bucket' | noun='lantern'<br>tail: `The old man stopped, looked hard at us, looked down into the` |
| ' bon' | '\n' | 435 | '\n' / ' D' / ' ro' / ' c' / ' Ch' | noun='bonnet'<br>tail: `, in a womanly sort of manner belonging to the apron and the` |
| ' pur' | '\n' | 1112 | '\n' / 'self' / ' Bag' / ' L' / ' G' | noun='purse'<br>tail: `She draws off her glove to get some money from her` |
| ' book' | '\n' | 614 | '\n' / ' C' / ' M' / ' In' / ' L' | noun='book'<br>tail: `” Having put the letters in his` |
| ' bon' | '\n' | 1153 | '\n' / ' w' / ' W' / ' D' / ' C' | noun='bonnet'<br>tail: `But that there’s the wale, the` |
| ' c' | '\n' | 87 | '\n' / ' K' / ' Sum' / ' R' / '\\' | noun='candle'<br>tail: `Winking cousins, bat-like in the` |
| ' lan' | '\n' | 796 | '\n' / ' Ch' / ' D' / ' M' / ' B' | noun='lantern'<br>tail: `I could see, from my window, the` |
| ' c' | '\n' | 256 | '\n' / ' Ch' / ' L' / ' K' / ' ' | noun='candle'<br>tail: `Now, Mademoiselle Hortense, let me recommend you to take the` |
| ' book' | '\n' | 1276 | '\n' / ' thoughts' / ' mind' / ' L' / ' M' | noun='book'<br>tail: `He was lost in thought, his` |
| ' pur' | '\n' | 1579 | '\n' / ' slight' / ' L' / ' G' / 'we' | noun='purse'<br>tail: `fair Dedlock delivers in her youthful manner, while making a` |
| ' watch' | ' threshold' | 4 | ' threshold' / ' stage' / ' Threshold' / ' bench' / ' watch' | noun='watch'<br>tail: `der is done; so, now she sees that when he used to be on the` |
| ' book' | ' Ch' | 356 | ' Ch' / ' D' / ' C' / ' L' / ' B' | noun='book'<br>tail: `es softly into the back office, refers to the entries in the` |
| ' clo' | ' deep' | 1161 | ' deep' / '\n' / ' small' / ' step' / ' look' | noun='cloak'<br>tail: `Now, you see, George”—he takes a` |
| ' lan' | ' L' | 5020 | ' L' / ' Ch' / ' H' / ' C' / ' B' | noun='lantern'<br>tail: `able brief, and outwardly directing his contemplation to the` |
| ' ' | ' own' | 33 | ' own' / '\n' / ' L' / ' Christ' / ' Guard' | noun='umbrella'<br>tail: `ticular lady whose lord is more than suspected of laying his` |
| ' lan' | '\n' | 907 | '\n' / ' Ch' / ' G' / ' C' / ' D' | noun='lantern'<br>tail: `see, I have so many things here,” he resumed, holding up the` |
| ' book' | ' certain' | 108 | ' certain' / ' few' / ' very' / ' Table' / ' slight' | noun='book'<br>tail: `ixth volume of the Philosophical Transactions; and also of a` |
| ' ' | ' L' | 45 | ' L' / '\n' / ' Ch' / ' C' / ' M' | noun='umbrella'<br>tail: `ving the trooper a great poke between the shoulders with her` |
| ' ' | '\n' | 35 | '\n' / ' old' / ' English' / ' A' / ' In' | noun='umbrella'<br>tail: `er quarter of the world—with nothing but a grey cloak and an` |
| ' boot' | '\n' | 4523 | '\n' / ' L' / ' C' / ' G' / ' Bucket' | noun='boots'<br>tail: `kind and gentle, and as he stood before the fire warming his` |
| ' c' | '\n' | 381 | '\n' / ' Ch' / ' L' / ' D' / ' Se' | noun='candle'<br>tail: `athing lulls or his fixed eyes show any consciousness of the` |
| ' sh' | ' L' | 1958 | ' L' / 'self' / ' D' / ' C' / ' M' | noun='shawl'<br>tail: `stal upon the terrace, and a vase upon the pedestal, and her` |
| ' watch' | '\n' | 1667 | '\n' / ' L' / ' Cook' / ' M' / ' Ch' | noun='watch'<br>tail: `l, Jarndyce,” returned his guest, who seemed to refer to his` |
| ' boot' | '\n' | 248 | '\n' / ' Bucket' / ' feet' / '\r' / '\n ' | noun='boots'<br>tail: `Bucket thoughtfully came and warmed the soles of his` |
| ' watch' | '\n' | 1859 | '\n' / ' L' / ' Master' / ' M' / ' G' | noun='watch'<br>tail: `Tulkinghorn, muttering reproof to his` |
| ' pur' | '\n' | 3244 | '\n' / ' L' / ' G' / ' M' / ' D' | noun='purse'<br>tail: `him that during the vacation and while things are slack, his` |
| ' clo' | '\n' | 1774 | '\n' / ' c' / ' Bucket' / ' cold' / ' G' | noun='cloak'<br>tail: `o the hotel and wait until he joined me there, so he threw a` |
| ' watch' | '\n' | 892 | '\n' / ' L' / ' Ch' / ' M' / ' G' | noun='watch'<br>tail: `“Now, little housewife,” said my guardian, looking at his` |
| ' stick' | '\n' | 1214 | '\n' / ' Ch' / ' L' / ' C' / ' K' | noun='stick'<br>tail: `Boythorn in a violent burst and stopping to strike his` |
| ' pur' | '\n' | 60 | '\n' / ' D' / ' C' / ' Bible' / ' p' | noun='purse'<br>tail: `eorge, my considerate friend,” returns Allan, taking out his` |

### C_plot   (0/50 hit; 50 miss)

| target | model top-1 | rank | top-5 | source / cue |
|--------|-------------|------|-------|--------------|
| 'Ded' | '<NAME>' | 2746 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Lady '->'Dedlock'<br>tail: `impend over yourself.”  “Well, sir?”  “Well, Lady` |
| 'G' | '<NAME>' | 2383 | '<NAME>' / '\n' / '<fim_suffix>' / ' Ada' / ' Bucket' | phrase='Mr. '->'Guppy'<br>tail: `his companions are yet midway in theirs, that Mr.` |
| 'Ch' | '<NAME>' | 396 | '<NAME>' / '1' / '\n' / '2' / '<fim_suffix>' | phrase='Court of '->'Chancery'<br>tail: `ernoon some score of members of the High Court of` |
| 'T' | '<NAME>' | 1116 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Mr. '->'Tulkinghorn'<br>tail: `eicester and his ancestors and his patrimony”—Mr.` |
| 'Bo' | '<NAME>' | 264 | '<NAME>' / '\n' / ' Bucket' / ' Ada' / '<KEY>' | phrase='Mr. '->'Boythorn'<br>tail: `” suggested Richard.  “By my soul,” exclaimed Mr.` |
| 'Ro' | '<NAME>' | 1109 | '<NAME>' / '\n' / '1' / '2' / '<file_sep>' | phrase='Mrs. '->'Rouncewell'<br>tail: `there is any uncommon eye in the case, it is Mrs.` |
| 'T' | '<NAME>' | 2566 | '<NAME>' / '\n' / ' Ada' / '<file_sep>' / ' Bag' | phrase='Mr. '->'Tulkinghorn'<br>tail: `ct involuntarily starts and falls back. It is Mr.` |
| 'T' | '<NAME>' | 6671 | '<NAME>' / '\n' / ' Sn' / '<file_sep>' / '1' | phrase='Mr. '->'Tulkinghorn'<br>tail: `s and ends together than if she had meant it. Mr.` |
| 'Ro' | '<NAME>' | 80 | '<NAME>' / '\n' / '1' / '<file_sep>' / '2' | phrase='Mrs. '->'Rouncewell'<br>tail: `s he got better.  “Where is your son George, Mrs.` |
| 'Ded' | '<NAME>' | 4292 | '<NAME>' / '1' / '2' / '\n' / '3' | phrase='Lady '->'Dedlock'<br>tail: `r by it, and reads, boldly written in each, “Lady` |
| 'Sn' | '<NAME>' | 179 | '<NAME>' / '\n' / ' Ada' / ' Bucket' / '<file_sep>' | phrase='Mr. '->'Snagsby'<br>tail: `d to me, he wos!”  As he shuffles downstairs, Mr.` |
| 'G' | '<NAME>' | 2723 | '<NAME>' / '\n' / '<file_sep>' / '1' / '2' | phrase='Mr. '->'Guppy'<br>tail: `rty! The property! Property!”  Mr. Weevle and Mr.` |
| 'Sn' | '<NAME>' | 37 | '<NAME>' / '\n' / '<file_sep>' / '1' / '2' | phrase='Mr. '->'Snagsby'<br>tail: `ticularity express, but she knows that Jo was Mr.` |
| 'G' | '<NAME>' | 1603 | '<NAME>' / '\n' / '1' / '2' / '\n\n\n\n' | phrase='Mr. '->'Guppy'<br>tail: `ely strolling down a flat country to the sea. Mr.` |
| 'T' | '<NAME>' | 1823 | '<NAME>' / '\n' / '<file_sep>' / '1' / ' Bag' | phrase='Mr. '->'Tulkinghorn'<br>tail: `s ever. Yet it may be that my Lady fears this Mr.` |
| 'Ded' | '<NAME>' | 4082 | '<NAME>' / '2' / '1' / '<fim_suffix>' / '3' | phrase='Lady '->'Dedlock'<br>tail: `aims one morning to the listening earth that Lady` |
| 'Ded' | '<NAME>' | 3431 | '<NAME>' / '\n' / '2' / '1' / '3' | phrase='Lady '->'Dedlock'<br>tail: `h a foil in his old-fashioned rusty black to Lady` |
| 'G' | '<NAME>' | 1776 | '<NAME>' / '\n' / ' Bucket' / ' Ada' / '<file_sep>' | phrase='Mr. '->'Guppy'<br>tail: `ou wouldn’t object to admit that, miss?” said Mr.` |
| 'W' | '\n' | 933 | '\n' / ' Street' / '<file_sep>' / '<NAME>' / ' Road' | phrase='Chesney '->'Wold'<br>tail: `e. You have often carried my spare gun at Chesney` |
| 'Sn' | '<NAME>' | 47 | '<NAME>' / '<file_sep>' / '\n' / '2' / '<KEY>' | phrase='Mr. '->'Snagsby'<br>tail: `and, my love?”  “At six,” says Mrs. Snagsby.  Mr.` |
| 'K' | '<NAME>' | 4165 | '<NAME>' / '\n' / '1' / '2' / '<file_sep>' | phrase='Mr. '->'Krook'<br>tail: `is cough of mild persuasion behind his hand. “Mr.` |
| 'J' | '<NAME>' | 3688 | '<NAME>' / '1' / '2' / '3' / '4' | phrase='Caddy '->'Jellyby'<br>tail: `ove with somebody and being ashamed of it.  Caddy` |
| 'Bucket' | '<NAME>' | 139 | '<NAME>' / '1' / '2' / '3' / '\n' | phrase='Inspector '->'Bucket'<br>tail: `hat I’m a-going to do. You know me. I’m Inspector` |
| 'K' | '<NAME>' | 2568 | '<NAME>' / '\n' / '<file_sep>' / ' Ada' / ' Bucket' | phrase='Mr. '->'Krook'<br>tail: `Run, Flite, run! The nearest doctor! Run!” So Mr.` |
| 'Sn' | '<NAME>' | 217 | '<NAME>' / '\n' / '<file_sep>' / ' Ada' / ' Bucket' | phrase='Mr. '->'Snagsby'<br>tail: `h a nod.  “About a year and a half ago,” says Mr.` |
| 'Bo' | '<NAME>' | 179 | '<NAME>' / ' Ada' / '\n' / ' Bag' / ' Bucket' | phrase='Mr. '->'Boythorn'<br>tail: `, who was so tame that he was brought down by Mr.` |
| 'W' | '<NAME>' | 1197 | '<NAME>' / '\n' / '<file_sep>' / '’' / ' Street' | phrase='Chesney '->'Wold'<br>tail: `knows where to have her. To-day she is at Chesney` |
| 'Jar' | '<NAME>' | 290 | '<NAME>' / '’' / '\n' / '<file_sep>' / '1' | phrase='Jarndyce and '->'Jarndyce'<br>tail: `e liberty? Your servant, sir. I know Jarndyce and` |
| 'Bo' | '<NAME>' | 165 | '<NAME>' / '\n' / '1' / ' Jar' / '<file_sep>' | phrase='Mr. '->'Boythorn'<br>tail: `t actions against HIM for trespass,” returned Mr.` |
| 'Bo' | '<NAME>' | 27 | '<NAME>' / '\n' / '<KEY>' / '<file_sep>' / ' Bag' | phrase='Mr. '->'Boythorn'<br>tail: `room, alighted on his master’s head. To hear Mr.` |
| 'Bucket' | '<NAME>' | 203 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Inspector '->'Bucket'<br>tail: `unlikely that you may inquire of me why Inspector` |
| 'W' | '<NAME>' | 1107 | '<NAME>' / '\n' / '<fim_suffix>' / '<file_sep>' / '<|endoftext|>' | phrase='Chesney '->'Wold'<br>tail: `ster and the baronetcy, Sir Leicester and Chesney` |
| 'Jar' | '<NAME>' | 58 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Jarndyce and '->'Jarndyce'<br>tail: `as promised a new rocking-horse when Jarndyce and` |
| 'J' | '<NAME>' | 1563 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Mrs. '->'Jellyby'<br>tail: `n’t think she did. I therefore supposed that Mrs.` |
| 'Sum' | '<NAME>' | 955 | '<NAME>' / '1' / '\n' / '2' / '3' | phrase='Esther '->'Summerson'<br>tail: `nce. I was left in charge of a child named Esther` |
| 'Ded' | '<NAME>' | 8615 | '<NAME>' / '1' / '2' / '\n' / '3' | phrase='Sir Leicester '->'Dedlock'<br>tail: `re to-day.”  “The visitor, my dear! Sir Leicester` |
| 'Ded' | '<NAME>' | 7986 | '<NAME>' / '1' / '\n' / '2' / '4' | phrase='Sir Leicester '->'Dedlock'<br>tail: `o a relation of the family, a great Sir Leicester` |
| 'Sk' | '<NAME>' | 2166 | '<NAME>' / '1' / '2' / '\n' / '3' | phrase='Harold '->'Skimpole'<br>tail: `y don’t stop, why should I? There you have Harold` |
| 'Ded' | '<NAME>' | 5591 | '<NAME>' / '1' / '2' / '3' / '\n' | phrase='Sir Leicester '->'Dedlock'<br>tail: `shall peep in from the outerside.  “Sir Leicester` |
| 'W' | '\n' | 853 | '\n' / '’' / '<NAME>' / '<file_sep>' / ' Road' | phrase='Chesney '->'Wold'<br>tail: `d he might have the good fortune to be at Chesney` |
| 'J' | '<NAME>' | 109 | '<NAME>' / '\n' / '1' / '2' / ' J' | phrase='Mrs. '->'Jellyby'<br>tail: `ompany.  We duly came back to breakfast, and Mrs.` |
| 'K' | '<NAME>' | 6510 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Mr. '->'Krook'<br>tail: `r. He soon returns with the intelligence that Mr.` |
| 'K' | '<NAME>' | 17 | '<NAME>' / '\n' / '<file_sep>' / '1' / '2' | phrase='Mr. '->'Krook'<br>tail: `How do you do, sir? You are looking charming, Mr.` |
| 'Ded' | '<NAME>' | 10313 | '<NAME>' / '1' / '2' / '\n' / '3' | phrase='Sir Leicester '->'Dedlock'<br>tail: `hundred.  “That is, I am deputed by Sir Leicester` |
| 'Ro' | '<NAME>' | 1068 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Mrs. '->'Rouncewell'<br>tail: `e for ever and a day.  “He shall have,” says Mrs.` |
| 'Ro' | '<NAME>' | 161 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Mrs. '->'Rouncewell'<br>tail: `and with a profound curtsy.  “How do you do, Mrs.` |
| 'Ro' | '<NAME>' | 2343 | '<NAME>' / '1' / '\n' / '2' / '3' | phrase='George '->'Rouncewell'<br>tail: `run glistening down her sun-brown face.  “George` |
| 'J' | '<NAME>' | 1887 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Mrs. '->'Jellyby'<br>tail: `r that he called the matrimonial alliance of Mrs.` |
| 'J' | '<NAME>' | 413 | '<NAME>' / '\n' / '1' / '2' / '<file_sep>' | phrase='Mrs. '->'Jellyby'<br>tail: `quite well.  “Why, not quite, my dear,” said Mrs.` |
| 'Ro' | '<NAME>' | 3075 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='George '->'Rouncewell'<br>tail: `ed to remember you.”  “When I look at you, George` |

### D_code   (11/50 hit; 39 miss)

| target | model top-1 | rank | top-5 | source / cue |
|--------|-------------|------|-------|--------------|
| 'numpy' | '\n' | 353 | '\n' / ' com' / '1' / '2' / '4' | prefix: `'import '` |
| 'os' | '\n' | 5999 | '\n' / ' com' / '1' / '\t' / ' org' | prefix: `'import '` |
| 'default' | '\n' | 780 | '\n' / '<file_sep>' / ' \\' / '1' / '<|endoftext|>' | prefix: `'from collections import '` |
| 'List' | '\n' | 654 | '\n' / ' Tuple' / ' Dict' / ' List' / '<file_sep>' | prefix: `'from typing import '` |
| '*' | '\n' | 77 | '\n' / '1' / '2' / 'jar' / '<file_sep>' | prefix: `'def __init__(self, '` |
| 'main' | 'methodResult' | 6 | 'methodResult' / 'Noto' / 'olate' / 'istrad' / ' [$]' | prefix: `"if __name__ == '__"` |
| 'range' | '1' | 78 | '1' / '0' / '\n' / '2' / '4' | prefix: `'for i in '` |
| "'" | '<fim_suffix>' | 5 | '<fim_suffix>' / " '" / '1' / '2' / '0' | prefix: `'with open(path, '` |
| 'None' | '\n' | 386 | '\n' / '1' / '2' / '0' / '3' | prefix: `'return '` |
| 'ValueError' | ' #' | 837 | ' #' / '1' / '\n' / '4' / '2' | prefix: `'raise '` |
| '_' | 'send' | 26422 | 'send' / 'current' / 'main' / 'assertEqual' / 'destroy' | prefix: `'self.'` |
| 'Logger' | '_' | 5 | '_' / 'logger' / '<file_sep>' / 'G' / '__' | prefix: `'logger = logging.get'` |
| 'Linear' | 'Module' | 2 | 'Module' / 'Conv' / 'Linear' / 'Embed' / 'utils' | prefix: `'torch.nn.'` |
| 'array' | 'zeros' | 2 | 'zeros' / 'random' / 'array' / 'linalg' / 'sqrt' | prefix: `'np.'` |
| 'run' | 'subprocess' | 12 | 'subprocess' / 'send' / 'subscribe' / 'stdout' / 'stdin' | prefix: `'subprocess.'` |
| "'" | '\n   ' | 5107 | '\n   ' / 'os' / '\n       ' / '\n' / 'self' | prefix: `"'.split("` |
| 'name' | '\n' | 12259 | '\n' / '<file_sep>' / '1' / '<NAME>' / '<fim_suffix>' | prefix: `'@property\ndef '` |
| 'x' | '\t' | 10179 | '\t' / '\n' / '1' / '3' / '2' | prefix: `'try:\n    '` |
| 'e' | '\n' | 356 | '\n' / '<fim_suffix>' / '1' / '5' / '4' | prefix: `'except Exception as '` |
| 'x' | '1' | 32611 | '1' / '0' / '2' / '3' / '4' | prefix: `'assert '` |
| 'x' | '\n' | 252 | '\n' / '1' / '2' / '3' / '4' | prefix: `'yield '` |
| 'main' | '\n' | 6178 | '\n' / '<file_sep>' / '<|endoftext|>' / '1' / ' _' | prefix: `'async def '` |
| 'sleep' | 'jar' | 40 | 'jar' / 'bucket' / 'up' / 'request' / 'ge' | prefix: `'await asyncio.'` |
| 'fixture' | 'methodResult' | 2 | 'methodResult' / 'ctest' / 'fixture' / 'autoconfigure' / 'plugins' | prefix: `'pytest.'` |
| 'Optional' | 'T' | 751 | 'T' / 'Strict' / 'Self' / 'FunctionType' / 'Any' | prefix: `'typing.'` |
| 'file' | 'dirname' | 2 | 'dirname' / 'FILE' / 'file' / 'DIR' / 'dir' | prefix: `'Path(__'` |
| 'info' | 'Log' | 14 | 'Log' / 'INFO' / 'Strict' / 'Logger' / 'getLogger' | prefix: `'logging.'` |
| 'dataclass' | 'py' | 2 | 'py' / 'dat' / 'dataclass' / 'class' / 'decorator' | prefix: `'dataclasses.'` |
| 'partial' | 'jar' | 2515 | 'jar' / 'sum' / 'up' / 'ic' / 'ly' | prefix: `'functools.'` |
| 'chain' | 'jar' | 542 | 'jar' / 'bucket' / 'up' / 'Bucket' / 'ge' | prefix: `'itertools.'` |
| 'Ordered' | 'Generic' | 105 | 'Generic' / 'collections' / 'Collections' / 'Collection' / 'Any' | prefix: `'collections.'` |
| 'ascii' | 'ic' | 15874 | 'ic' / 'ok' / 'string' / 'up' / 'before' | prefix: `'string.'` |
| 'pi' | 'math' | 20 | 'math' / 'random' / 'Stack' / 'cos' / 'h' | prefix: `'math.'` |
| 'seed' | 'randint' | 9 | 'randint' / 'random' / 'randn' / 'choice' / 'shuffle' | prefix: `'random.'` |
| 'b' | 'encode' | 286 | 'encode' / 'decode' / 'h' / 'convert' / 'UTF' | prefix: `'base64.'` |
| 'request' | 'parse' | 1 | 'parse' / 'request' / 'http' / 'urllib' / 'url' | prefix: `'urllib.'` |
| 'get' | 'send' | 17 | 'send' / 'GET' / 'py' / 'http' / 'Send' | prefix: `'requests.'` |
| 'db' | 'Django' | 9 | 'Django' / 'django' / 'contrib' / 'core' / 'urls' | prefix: `'django.'` |
| 'Linear' | 'predict' | 4 | 'predict' / 'net' / 'Dense' / 'Max' / 'Linear' | prefix: `'nn.'` |


## C4 adapter + layer-16 engram prefix at embedding layer

### A_character   (1/50 hit; 49 miss)

| target | model top-1 | rank | top-5 | source / cue |
|--------|-------------|------|-------|--------------|
| ' T' | '\n' | 174 | '\n' / ' being' / ' be' / ' Be' / ' True' | character='Tulkinghorn'<br>tail: `t same moment, there happens to be an old man of the name of` |
| ' All' | '\n' | 42 | '\n' / ' True' / ' B' / ' ' / ' J' | character='Allan Woodcourt'<br>tail: `The broken footway is so narrow that when` |
| ' Est' | '\n' | 4846 | '\n' / 'up' / 'hand' / 'ly' / ' late' | character='Esther Summerson'<br>tail: `here I was a day boarder, and although they called me little` |
| ' Ge' | '\n' | 20 | '\n' / ' God' / ' C' / ' Good' / ' L' | character='George Rouncewell'<br>tail: `“Good heaven, and it is really` |
| ' Ada' | ' M' | 55 | ' M' / '\n' / ' D' / ' C' / ' H' | character='Ada Clare'<br>tail: `Richard Carstone and of Miss` |
| ' Richard' | '\n' | 381 | '\n' / 'less' / 'se' / 'ly' / 'ad' | character='Richard Carstone'<br>tail: `gentleman was her distant cousin, she told me, and his name` |
| ' S' | '\n' | 59 | '\n' / ' You' / ' Be' / ' Now' / ' ' | character='Sir Leicester'<br>tail: `sper still goes about that she had not even family; howbeit,` |
| ' K' | '\n' | 82 | '\n' / ' Do' / ' Does' / ' D' / ' You' | character='Krook'<br>tail: `raciously observed to him before passing out, “That will do,` |
| ' Har' | ' God' | 424 | ' God' / ' L' / ' Ch' / ' Ada' / ' Your' | character='Harold Skimpole'<br>tail: `, holiness, commerce, trade, any object you prefer; only—let` |
| ' H' | '\n' | 24 | '\n' / ' L' / ' D' / ' C' / ' Jar' | character='Hortense'<br>tail: `In short, it is such an admirable thing that Mademoiselle` |
| ' Ins' | '\n' | 343 | '\n' / ' name' / ' Now' / ' These' / ' This' | character='Inspector Bucket'<br>tail: `ng that may appear to be disagreeable in this, for my name’s` |
| ' John' | '\n' | 121 | '\n' / ' this' / ' You' / ' you' / 'gether' | character='John Jarndyce'<br>tail: `” “Since you refer so immediately to` |
| ' C' | '\n' | 4 | '\n' / 'ting' / ' L' / ' You' / ' C' | character='Caddy Jellyby'<br>tail: `So I thought one day when I went to London to meet` |
| ' L' | 'self' | 104 | 'self' / '\n' / 'our' / 'ours' / 'selves' | character='Lady Dedlock'<br>tail: `dded to these, soon floated her upward, and for years now my` |
| ' T' | '\n' | 21 | '\n' / ' man' / ' M' / ' ' / ' Man' | character='Tulkinghorn'<br>tail: `now observes from his couch that man told him ya’as’dy that` |
| ' All' | '\n' | 96 | '\n' / ' M' / ' C' / ' Ch' / ' J' | character='Allan Woodcourt'<br>tail: `ome excrescence produced there in neglect and impurity, that` |
| ' Est' | 'our' | 606 | 'our' / '\n' / 'us' / ' M' / 'is' | character='Esther Summerson'<br>tail: `We are, Madam, Your obedt Servts, Kenge and Carboy Miss` |
| ' Ge' | '\n' | 108 | '\n' / ' All' / ' Now' / ' G' / ' C' | character='George Rouncewell'<br>tail: `“All is still in readiness,` |
| ' Richard' | 'in' | 3213 | 'in' / '\n' / 'y' / 'w' / ' you' | character='Richard Carstone'<br>tail: `and released her, and then he spoke for a minute or two with` |
| ' S' | '\n' | 118 | '\n' / 'ad' / 'me' / 'mat' / 'ther' | character='Sir Leicester'<br>tail: `n air of prescription about him which is always agreeable to` |
| ' K' | ' L' | 22 | ' L' / ' C' / ' G' / ' M' / ' Ch' | character='Krook'<br>tail: `as is announced in paint, to all whom it may concern, by one` |
| ' Har' | '\n' | 399 | '\n' / ' L' / ' C' / ' J' / ' God' | character='Harold Skimpole'<br>tail: `Then, for heaven’s sake, having` |
| ' H' | '\n' | 13 | '\n' / ' You' / ' L' / ' Now' / ' This' | character='Hortense'<br>tail: `to attend,” says my Lady then, addressing the reflection of` |
| ' Ins' | '\n' | 845 | '\n' / '’' / ' you' / ' You' / '”' | character='Inspector Bucket'<br>tail: `y about admitting of it, you tell her that it’s no use, that` |
| ' John' | '\n' | 38 | '\n' / ' God' / ' C' / ' M' / ' J' | character='John Jarndyce'<br>tail: `I suppose your loyalty to` |
| ' C' | '\n' | 12 | '\n' / ' True' / ' W' / ' Do' / ' ' | character='Caddy Jellyby'<br>tail: `At last we came to Soho Square, where` |
| ' T' | '\n' | 6 | '\n' / ' Ch' / ' L' / ' world' / ' Common' | character='Tulkinghorn'<br>tail: `ined to add the last great secret to the many secrets of the` |
| ' All' | '\n' | 11 | '\n' / 'st' / 'wards' / 'gether' / ' Alice' | character='Allan Woodcourt'<br>tail: `s, a farewell to her, and takes his creeping way along after` |
| ' Est' | '\n' | 357 | '\n' / ' L' / ' I' / ' l' / ' Jes' | character='Esther Summerson'<br>tail: `And yet I—I, little` |
| ' Ge' | '\n' | 265 | '\n' / ' A' / ' Al' / ' Bag' / ' Ada' | character='George Rouncewell'<br>tail: `Very familiar to him, as he said himself some hours ago,` |
| ' Richard' | '\n' | 131 | '\n' / ' This' / ' ' / ' Jar' / ' True' | character='Richard Carstone'<br>tail: `YOUR name now will be—” “` |
| ' S' | 'er' | 332 | 'er' / 'ing' / 'en' / '\n' / 'o' | character='Sir Leicester'<br>tail: `“Better now,” quoth` |
| ' K' | '\n' | 115 | '\n' / '<|endoftext|>' / ' Your' / ' W' / '\n\n' | character='Krook'<br>tail: `The welcome light soon shines upon the wall, as` |
| ' Har' | '\n' | 342 | '\n' / ' M' / ' God' / ' ' / ' L' | character='Harold Skimpole'<br>tail: `Mankind will surely not deny to` |
| ' H' | ' Made' | 21 | ' Made' / '\n' / ',' / ' M' / ' ' | character='Hortense'<br>tail: `t, a peaceful figure too in the landscape, went Mademoiselle` |
| ' Ins' | '\n' | 262 | '\n' / ' the' / 'the' / 'm' / 't' | character='Inspector Bucket'<br>tail: `Put it to her ladyship, if you think it right, from` |
| ' John' | '\n' | 316 | '\n' / 'h' / ' You' / 'ther' / ' There' | character='John Jarndyce'<br>tail: `” “There you come back to` |
| ' C' | ' L' | 2 | ' L' / ' M' / ' C' / ' ' / ' H' | character='Caddy Jellyby'<br>tail: `r was a greater imposter than I with a blinder follower than` |
| ' L' | '\n' | 13 | '\n' / 'self' / 'selves' / 'made' / ' M' | character='Lady Dedlock'<br>tail: `le circumstance to be noted in everything associated with my` |
| ' T' | '\n' | 37 | '\n' / ' the' / 'the' / ' ' / ' Jar' | character='Tulkinghorn'<br>tail: `en a murder in Lincoln’s Inn Fields—gentleman of the name of` |
| ' All' | 'in' | 33 | 'in' / '\n' / ' You' / ' Jo' / ' Such' | character='Allan Woodcourt'<br>tail: `CHAPTER XLVII Jo’s Will As` |
| ' Est' | '\n' | 893 | '\n' / ' ' / ' You' / ' you' / ' Q' | character='Esther Summerson'<br>tail: `I was left in charge of a child named` |
| ' S' | '\n' | 33 | '\n' / ' God' / ' Richard' / 'getting' / ' Ada' | character='Sir Leicester'<br>tail: `re is any superabundant life of imagination on the spot, for` |
| ' K' | '\n' | 24 | '\n' / 'up' / '”' / 'in' / ' ‘' | character='Krook'<br>tail: `I don’t know,” says` |
| ' Har' | 'This' | 8205 | 'This' / 'You' / 'To' / 'There' / 'Th' | character='Harold Skimpole'<br>tail: `Skimpole, “to this effect: ‘` |
| ' H' | '\n' | 24 | '\n' / ' Jar' / ' D' / ' C' / ' L' | character='Hortense'<br>tail: `“Thank you, Mademoiselle` |
| ' Ins' | ' M' | 181 | ' M' / ' Char' / ' Jar' / ' Ch' / ' Ada' | character='Inspector Bucket'<br>tail: `single moment in the course of this prolonged night, here is` |
| ' John' | '\n' | 301 | '\n' / 'uches' / ' ' / ' Char' / 'c' | character='John Jarndyce'<br>tail: `se that I have come here to make underhanded charges against` |
| ' C' | '\n' | 8 | '\n' / ' D' / ' H' / ' G' / ' De' | character='Caddy Jellyby'<br>tail: `happened that when I came home from Deal I found a note from` |

### B_possession   (0/50 hit; 50 miss)

| target | model top-1 | rank | top-5 | source / cue |
|--------|-------------|------|-------|--------------|
| ' c' | '\n' | 5 | '\n' / ' b' / ' you' / ' You' / ' ' | noun='candle'<br>tail: `an arm to Ada and an arm to me, and bidding Richard bring a` |
| ' spect' | '\n' | 28162 | '\n' / 'es' / 'self' / ' here' / ' her' | noun='spectacles'<br>tail: `Pardiggle, who had been regarding him through her` |
| ' letter' | ' You' | 9940 | ' You' / '\n' / ' Now' / ' Not' / ' This' | noun='letter'<br>tail: `er Smallweed smiles in a very ugly way in recognition of the` |
| ' spect' | '\n' | 12880 | '\n' / ' up' / 'up' / ' ' / ' T' | noun='spectacles'<br>tail: `Tulkinghorn gets up, adjusts his` |
| ' glo' | '\n' | 24921 | '\n' / ' You' / ' M' / ' T' / ' D' | noun='gloves'<br>tail: `urveydrop, standing with his back to the fire and waving his` |
| ' bon' | 'self' | 1556 | 'self' / '\n' / ' C' / 'by' / ' Ch' | noun='bonnet'<br>tail: `ng easily anywhere, she perches on a rough bench, unties her` |
| ' letter' | '\n' | 1694 | '\n' / ' Sub' / ' L' / ' sub' / 'e' | noun='letter'<br>tail: `This was the substance of the` |
| ' letter' | 'ha' | 10732 | 'ha' / 'in' / 'ins' / '\n' / 'de' | noun='letter'<br>tail: `ppy and grateful, and with her approval I had written such a` |
| ' hat' | '\n' | 40 | '\n' / ' own' / ' C' / ' D' / ' L' | noun='hat'<br>tail: `stops of his own accord, and Sir Leicester, pulling off his` |
| ' spect' | '\n' | 19104 | '\n' / ' You' / ' D' / ' Tw' / ' T' | noun='spectacles'<br>tail: `Rustily drest, with his` |
| ' letter' | '\n' | 3177 | '\n' / ' ' / ' late' / '\\' / ' Se' | noun='letter'<br>tail: `aid a visit of a few hours to London, which something in the` |
| ' spect' | 'es' | 20371 | 'es' / 'ew' / '\n' / 'et' / 'ds' | noun='spectacles'<br>tail: `housekeeper at Chesney Wold, has several times taken off her` |
| ' hat' | ' black' | 115 | ' black' / '\n' / ' white' / ' bl' / ' ' | noun='hat'<br>tail: `leeves and his grey coat, pulls on his black coat, takes his` |
| ' boot' | '\n' | 3057 | '\n' / ' ' / ' H' / ' Hol' / ' quite' | noun='boots'<br>tail: `of books and papers and in part quite a little museum of his` |
| ' clo' | '\n' | 20310 | '\n' / 'ses' / 'se' / ' ' / 'does' | noun='cloak'<br>tail: `, disregarding my remonstrances, had hurriedly taken off his` |
| ' hat' | 'ha' | 1012 | 'ha' / '\n' / 'in' / 'ins' / 'de' | noun='hat'<br>tail: `A man yet dark and muddy, in long swollen sodden boots and a` |
| ' hat' | '\n' | 6007 | '\n' / ' A' / ' C' / ' Bes' / ' As' | noun='hat'<br>tail: `George, entirely assenting, puts on his` |
| ' clo' | '\n' | 15205 | '\n' / 'm' / 'o' / 'se' / 'ms' | noun='cloak'<br>tail: `d in a moment, and they took me between them, wrapped in the` |
| ' bon' | 'es' | 8417 | 'es' / 'et' / 'bs' / '\n' / 'by' | noun='bonnet'<br>tail: `and joined Miss Jellyby, who was by this time putting on her` |
| ' ' | 'e' | 23 | 'e' / '\n' / 'm' / ' Bag' / 'od' | noun='umbrella'<br>tail: `Bagnet expresses with the` |
| ' lan' | '\n' | 13267 | '\n' / ' you' / ' You' / ' ' / ' L' | noun='lantern'<br>tail: `The old man stopped, looked hard at us, looked down into the` |
| ' bon' | 'e' | 6782 | 'e' / '\n' / 'at' / 'o' / 'od' | noun='bonnet'<br>tail: `, in a womanly sort of manner belonging to the apron and the` |
| ' pur' | 'es' | 3005 | 'es' / 'ew' / '\n' / 'ms' / 'bal' | noun='purse'<br>tail: `She draws off her glove to get some money from her` |
| ' book' | '\n' | 386 | '\n' / ' his' / ' her' / ' own' / ' good' | noun='book'<br>tail: `” Having put the letters in his` |
| ' bon' | ' w' | 1324 | ' w' / '\n' / ' W' / 'w' / ' Ch' | noun='bonnet'<br>tail: `But that there’s the wale, the` |
| ' c' | '\n' | 43 | '\n' / ' Ch' / ' ch' / ' B' / ' dark' | noun='candle'<br>tail: `Winking cousins, bat-like in the` |
| ' lan' | '\n' | 8427 | '\n' / 'm' / 'e' / ' L' / ' table' | noun='lantern'<br>tail: `I could see, from my window, the` |
| ' c' | '\n' | 440 | '\n' / ' You' / ' ' / ' H' / ' Now' | noun='candle'<br>tail: `Now, Mademoiselle Hortense, let me recommend you to take the` |
| ' book' | '\n' | 686 | '\n' / ' mother' / ' M' / 'self' / 'm' | noun='book'<br>tail: `He was lost in thought, his` |
| ' pur' | 'in' | 9790 | 'in' / 'ha' / 'ins' / '\n' / 'uld' | noun='purse'<br>tail: `fair Dedlock delivers in her youthful manner, while making a` |
| ' watch' | ' threshold' | 13 | ' threshold' / ' stage' / ' bench' / ' Threshold' / ' B' | noun='watch'<br>tail: `der is done; so, now she sees that when he used to be on the` |
| ' book' | '\n' | 139 | '\n' / ' Ch' / ' C' / ' D' / ' B' | noun='book'<br>tail: `es softly into the back office, refers to the entries in the` |
| ' clo' | 'in' | 16443 | 'in' / 'ins' / 'ha' / '\n' / 'ye' | noun='cloak'<br>tail: `Now, you see, George”—he takes a` |
| ' lan' | ' L' | 4664 | ' L' / ' Ch' / ' Common' / ' London' / ' G' | noun='lantern'<br>tail: `able brief, and outwardly directing his contemplation to the` |
| ' ' | ' L' | 76 | ' L' / ' M' / 'self' / '\n' / ' D' | noun='umbrella'<br>tail: `ticular lady whose lord is more than suspected of laying his` |
| ' lan' | '\n' | 16094 | '\n' / 'm' / 'e' / 'se' / 'od' | noun='lantern'<br>tail: `see, I have so many things here,” he resumed, holding up the` |
| ' book' | ' certain' | 47 | ' certain' / ' very' / ' slight' / ' few' / ' second' | noun='book'<br>tail: `ixth volume of the Philosophical Transactions; and also of a` |
| ' ' | '\n' | 24 | '\n' / 'by' / 'f' / 'self' / 'ing' | noun='umbrella'<br>tail: `ving the trooper a great poke between the shoulders with her` |
| ' ' | '\n' | 11 | '\n' / 'gel' / ' old' / ' a' / 'c' | noun='umbrella'<br>tail: `er quarter of the world—with nothing but a grey cloak and an` |
| ' boot' | 'w' | 3642 | 'w' / '\n' / 'se' / 'self' / 'her' | noun='boots'<br>tail: `kind and gentle, and as he stood before the fire warming his` |
| ' c' | '\n' | 14 | '\n' / 'e' / ' the' / 'at' / ' late' | noun='candle'<br>tail: `athing lulls or his fixed eyes show any consciousness of the` |
| ' sh' | ' L' | 3579 | ' L' / 'self' / ' C' / ' Ch' / ' Ro' | noun='shawl'<br>tail: `stal upon the terrace, and a vase upon the pedestal, and her` |
| ' watch' | '\n' | 562 | '\n' / ' guest' / ' host' / 'self' / ' good' | noun='watch'<br>tail: `l, Jarndyce,” returned his guest, who seemed to refer to his` |
| ' boot' | '\n' | 3233 | '\n' / ' D' / ' M' / 'we' / ' ' | noun='boots'<br>tail: `Bucket thoughtfully came and warmed the soles of his` |
| ' watch' | '\n' | 1579 | '\n' / 'self' / ' M' / 'w' / ' H' | noun='watch'<br>tail: `Tulkinghorn, muttering reproof to his` |
| ' pur' | '\n' | 1546 | '\n' / ' D' / ' M' / ' L' / 'self' | noun='purse'<br>tail: `him that during the vacation and while things are slack, his` |
| ' clo' | '\n' | 10474 | '\n' / 'in' / ' ro' / 'ard' / 'qu' | noun='cloak'<br>tail: `o the hotel and wait until he joined me there, so he threw a` |
| ' watch' | '\n' | 316 | '\n' / 'aw' / 'we' / 'od' / ' M' | noun='watch'<br>tail: `“Now, little housewife,” said my guardian, looking at his` |
| ' stick' | '\n' | 9436 | '\n' / 'se' / 'self' / 'n' / 'sel' | noun='stick'<br>tail: `Boythorn in a violent burst and stopping to strike his` |
| ' pur' | '\n' | 3740 | '\n' / 'self' / ' M' / ' L' / ' D' | noun='purse'<br>tail: `eorge, my considerate friend,” returns Allan, taking out his` |

### C_plot   (0/50 hit; 50 miss)

| target | model top-1 | rank | top-5 | source / cue |
|--------|-------------|------|-------|--------------|
| 'Ded' | '<NAME>' | 2816 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Lady '->'Dedlock'<br>tail: `impend over yourself.”  “Well, sir?”  “Well, Lady` |
| 'G' | '<NAME>' | 2362 | '<NAME>' / '\n' / '<fim_suffix>' / ' Bag' / '<file_sep>' | phrase='Mr. '->'Guppy'<br>tail: `his companions are yet midway in theirs, that Mr.` |
| 'Ch' | '<NAME>' | 280 | '<NAME>' / '\n' / '1' / '<fim_suffix>' / '2' | phrase='Court of '->'Chancery'<br>tail: `ernoon some score of members of the High Court of` |
| 'T' | '<NAME>' | 774 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Mr. '->'Tulkinghorn'<br>tail: `eicester and his ancestors and his patrimony”—Mr.` |
| 'Bo' | '<NAME>' | 180 | '<NAME>' / '\n' / ' Bucket' / '<KEY>' / ' Ada' | phrase='Mr. '->'Boythorn'<br>tail: `” suggested Richard.  “By my soul,” exclaimed Mr.` |
| 'Ro' | '<NAME>' | 1187 | '<NAME>' / '\n' / '1' / '2' / '<file_sep>' | phrase='Mrs. '->'Rouncewell'<br>tail: `there is any uncommon eye in the case, it is Mrs.` |
| 'T' | '<NAME>' | 12569 | '<NAME>' / '\n' / '2' / '1' / '<file_sep>' | phrase='Mr. '->'Tulkinghorn'<br>tail: `ct involuntarily starts and falls back. It is Mr.` |
| 'T' | '<NAME>' | 21674 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Mr. '->'Tulkinghorn'<br>tail: `s and ends together than if she had meant it. Mr.` |
| 'Ro' | '<NAME>' | 55 | '<NAME>' / '\n' / '1' / '<file_sep>' / '2' | phrase='Mrs. '->'Rouncewell'<br>tail: `s he got better.  “Where is your son George, Mrs.` |
| 'Ded' | '<NAME>' | 6630 | '<NAME>' / '\n' / '2' / '1' / '3' | phrase='Lady '->'Dedlock'<br>tail: `r by it, and reads, boldly written in each, “Lady` |
| 'Sn' | '<NAME>' | 97 | '<NAME>' / '\n' / '<fim_suffix>' / ' Ada' / '<file_sep>' | phrase='Mr. '->'Snagsby'<br>tail: `d to me, he wos!”  As he shuffles downstairs, Mr.` |
| 'G' | '<NAME>' | 4649 | '<NAME>' / '2' / '1' / '\n' / '3' | phrase='Mr. '->'Guppy'<br>tail: `rty! The property! Property!”  Mr. Weevle and Mr.` |
| 'Sn' | '<NAME>' | 129 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Mr. '->'Snagsby'<br>tail: `ticularity express, but she knows that Jo was Mr.` |
| 'G' | '<NAME>' | 5195 | '<NAME>' / '\n' / '1' / '2' / '<file_sep>' | phrase='Mr. '->'Guppy'<br>tail: `ely strolling down a flat country to the sea. Mr.` |
| 'T' | '<NAME>' | 25048 | '<NAME>' / '1' / '2' / '\n' / '3' | phrase='Mr. '->'Tulkinghorn'<br>tail: `s ever. Yet it may be that my Lady fears this Mr.` |
| 'Ded' | '<NAME>' | 4923 | '<NAME>' / '\n' / '2' / '1' / '3' | phrase='Lady '->'Dedlock'<br>tail: `aims one morning to the listening earth that Lady` |
| 'Ded' | '<NAME>' | 3064 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Lady '->'Dedlock'<br>tail: `h a foil in his old-fashioned rusty black to Lady` |
| 'G' | '<NAME>' | 1538 | '<NAME>' / '\n' / '<file_sep>' / ' Ada' / '<KEY>' | phrase='Mr. '->'Guppy'<br>tail: `ou wouldn’t object to admit that, miss?” said Mr.` |
| 'W' | '\n' | 1087 | '\n' / '<file_sep>' / ' Street' / '<fim_suffix>' / '<NAME>' | phrase='Chesney '->'Wold'<br>tail: `e. You have often carried my spare gun at Chesney` |
| 'Sn' | '<NAME>' | 41 | '<NAME>' / '<file_sep>' / '\n' / '2' / '1' | phrase='Mr. '->'Snagsby'<br>tail: `and, my love?”  “At six,” says Mrs. Snagsby.  Mr.` |
| 'K' | '<NAME>' | 4102 | '<NAME>' / '\n' / '<file_sep>' / '1' / '2' | phrase='Mr. '->'Krook'<br>tail: `is cough of mild persuasion behind his hand. “Mr.` |
| 'J' | '<NAME>' | 4035 | '<NAME>' / '1' / '2' / '3' / '4' | phrase='Caddy '->'Jellyby'<br>tail: `ove with somebody and being ashamed of it.  Caddy` |
| 'Bucket' | '<NAME>' | 132 | '<NAME>' / '1' / '2' / '\n' / '3' | phrase='Inspector '->'Bucket'<br>tail: `hat I’m a-going to do. You know me. I’m Inspector` |
| 'K' | '<NAME>' | 586 | '<NAME>' / '\n' / '<file_sep>' / '<fim_suffix>' / '1' | phrase='Mr. '->'Krook'<br>tail: `Run, Flite, run! The nearest doctor! Run!” So Mr.` |
| 'Sn' | '<NAME>' | 287 | '<NAME>' / '\n' / '<file_sep>' / '1' / ' Ada' | phrase='Mr. '->'Snagsby'<br>tail: `h a nod.  “About a year and a half ago,” says Mr.` |
| 'Bo' | '<NAME>' | 126 | '<NAME>' / ' Ada' / '\n' / ' Bag' / '<file_sep>' | phrase='Mr. '->'Boythorn'<br>tail: `, who was so tame that he was brought down by Mr.` |
| 'W' | '<NAME>' | 1395 | '<NAME>' / '\n' / '<fim_suffix>' / '’' / '<file_sep>' | phrase='Chesney '->'Wold'<br>tail: `knows where to have her. To-day she is at Chesney` |
| 'Jar' | '<NAME>' | 217 | '<NAME>' / '’' / '\n' / '<file_sep>' / '1' | phrase='Jarndyce and '->'Jarndyce'<br>tail: `e liberty? Your servant, sir. I know Jarndyce and` |
| 'Bo' | '<NAME>' | 3038 | '<NAME>' / '\n' / ' Jar' / '1' / '2' | phrase='Mr. '->'Boythorn'<br>tail: `t actions against HIM for trespass,” returned Mr.` |
| 'Bo' | '<NAME>' | 303 | '<NAME>' / '\n' / '1' / '2' / '<file_sep>' | phrase='Mr. '->'Boythorn'<br>tail: `room, alighted on his master’s head. To hear Mr.` |
| 'Bucket' | '<NAME>' | 190 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Inspector '->'Bucket'<br>tail: `unlikely that you may inquire of me why Inspector` |
| 'W' | '\n' | 1087 | '\n' / '<fim_suffix>' / 'ard' / '<NAME>' / ' Bag' | phrase='Chesney '->'Wold'<br>tail: `ster and the baronetcy, Sir Leicester and Chesney` |
| 'Jar' | '<NAME>' | 27 | '<NAME>' / '\n' / '1' / ' Jar' / '2' | phrase='Jarndyce and '->'Jarndyce'<br>tail: `as promised a new rocking-horse when Jarndyce and` |
| 'J' | '<NAME>' | 1930 | '<NAME>' / '\n' / '1' / '2' / '<file_sep>' | phrase='Mrs. '->'Jellyby'<br>tail: `n’t think she did. I therefore supposed that Mrs.` |
| 'Sum' | '<NAME>' | 1048 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Esther '->'Summerson'<br>tail: `nce. I was left in charge of a child named Esther` |
| 'Ded' | '<NAME>' | 7984 | '<NAME>' / '1' / '2' / '\n' / '3' | phrase='Sir Leicester '->'Dedlock'<br>tail: `re to-day.”  “The visitor, my dear! Sir Leicester` |
| 'Ded' | '<NAME>' | 7805 | '<NAME>' / '1' / '\n' / '2' / '5' | phrase='Sir Leicester '->'Dedlock'<br>tail: `o a relation of the family, a great Sir Leicester` |
| 'Sk' | '<NAME>' | 1335 | '<NAME>' / '1' / '2' / '\n' / '3' | phrase='Harold '->'Skimpole'<br>tail: `y don’t stop, why should I? There you have Harold` |
| 'Ded' | '<NAME>' | 9478 | '<NAME>' / '1' / '2' / '\n' / '3' | phrase='Sir Leicester '->'Dedlock'<br>tail: `shall peep in from the outerside.  “Sir Leicester` |
| 'W' | '\n' | 606 | '\n' / '<file_sep>' / '’' / '<NAME>' / '<|endoftext|>' | phrase='Chesney '->'Wold'<br>tail: `d he might have the good fortune to be at Chesney` |
| 'J' | '<NAME>' | 106 | '<NAME>' / '\n' / ' J' / '1' / '<file_sep>' | phrase='Mrs. '->'Jellyby'<br>tail: `ompany.  We duly came back to breakfast, and Mrs.` |
| 'K' | '<NAME>' | 8271 | '<NAME>' / '\n' / '1' / '2' / ' Job' | phrase='Mr. '->'Krook'<br>tail: `r. He soon returns with the intelligence that Mr.` |
| 'K' | '<NAME>' | 665 | '<NAME>' / '\n' / '1' / '2' / '<file_sep>' | phrase='Mr. '->'Krook'<br>tail: `How do you do, sir? You are looking charming, Mr.` |
| 'Ded' | '<NAME>' | 8868 | '<NAME>' / '1' / '\n' / '2' / '3' | phrase='Sir Leicester '->'Dedlock'<br>tail: `hundred.  “That is, I am deputed by Sir Leicester` |
| 'Ro' | '<NAME>' | 1056 | '<NAME>' / '\n' / '1' / '2' / '<file_sep>' | phrase='Mrs. '->'Rouncewell'<br>tail: `e for ever and a day.  “He shall have,” says Mrs.` |
| 'Ro' | '<NAME>' | 115 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Mrs. '->'Rouncewell'<br>tail: `and with a profound curtsy.  “How do you do, Mrs.` |
| 'Ro' | '<NAME>' | 2512 | '<NAME>' / '1' / '2' / '\n' / '3' | phrase='George '->'Rouncewell'<br>tail: `run glistening down her sun-brown face.  “George` |
| 'J' | '<NAME>' | 1673 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Mrs. '->'Jellyby'<br>tail: `r that he called the matrimonial alliance of Mrs.` |
| 'J' | '<NAME>' | 187 | '<NAME>' / '\n' / '2' / '1' / '<file_sep>' | phrase='Mrs. '->'Jellyby'<br>tail: `quite well.  “Why, not quite, my dear,” said Mrs.` |
| 'Ro' | '<NAME>' | 3226 | '<NAME>' / '\n' / '1' / '2' / '<file_sep>' | phrase='George '->'Rouncewell'<br>tail: `ed to remember you.”  “When I look at you, George` |

### D_code   (1/50 hit; 49 miss)

| target | model top-1 | rank | top-5 | source / cue |
|--------|-------------|------|-------|--------------|
| 'numpy' | '\n' | 1306 | '\n' / ' You' / ' ' / ' This' / '</' | prefix: `'import '` |
| 'os' | '\n' | 5090 | '\n' / ' You' / ' This' / ' ' / 'how' | prefix: `'import '` |
| 'default' | '\n' | 47939 | '\n' / '1' / '2' / '<NAME>' / '3' | prefix: `'from collections import '` |
| 'List' | '\n' | 41438 | '\n' / '1' / '2' / '<NAME>' / '3' | prefix: `'from typing import '` |
| '*' | '\n' | 15789 | '\n' / '1' / '2' / '0' / ' #' | prefix: `'def __init__(self, '` |
| 'object' | 'string' | 38 | 'string' / 'String' / '\n' / '1' / 'this' | prefix: `'class Foo('` |
| 'range' | '\n' | 2389 | '\n' / 'w' / '\n\n' / 'g' / ' ' | prefix: `'for i in '` |
| "'" | '1' | 3212 | '1' / '2' / '\n' / '0' / '3' | prefix: `'with open(path, '` |
| 'None' | '\n' | 1018 | '\n' / ' You' / 'y' / ' ' / ' This' | prefix: `'return '` |
| 'ValueError' | '\n' | 9764 | '\n' / '1' / '2' / '0' / '3' | prefix: `'raise '` |
| '_' | '\n' | 304 | '\n' / ' This' / ' Class' / ' The' / '\n\n' | prefix: `'self.'` |
| 'Logger' | '\n' | 848 | '\n' / ' Est' / ' Jar' / '<file_sep>' / ' Not' | prefix: `'logger = logging.get'` |
| 'Linear' | 'nn' | 16019 | 'nn' / '\n' / 'net' / 'torch' / 'n' | prefix: `'torch.nn.'` |
| 'array' | '\n' | 5333 | '\n' / ' This' / ' (' / ' ' / '\n\n' | prefix: `'np.'` |
| 'plot' | '\n' | 12986 | '\n' / ' This' / ' Web' / ' You' / ' (' | prefix: `'plt.'` |
| 'join' | '\n' | 30486 | '\n' / ' This' / ' True' / '\n\n' / ' (' | prefix: `'os.path.'` |
| 'run' | '\n' | 5788 | '\n' / ' (' / '\n\n' / ' This' / '\n ' | prefix: `'subprocess.'` |
| 'dumps' | '\n' | 20859 | '\n' / ' This' / ' (' / '\n\n' / ' ' | prefix: `'json.'` |
| 'compile' | '\n' | 6899 | '\n' / ' This' / ' (' / '\n\n' / ' The' | prefix: `'re.'` |
| "'" | '\n' | 2401 | '\n' / ' ' / ' This' / ' The' / '1' | prefix: `"'.split("` |
| 'init' | '\n' | 1 | '\n' / 'init' / 'get' / ' This' / 'super' | prefix: `'super().__'` |
| 'name' | '\n' | 219 | '\n' / 'n' / 'net' / ' You' / 'ch' | prefix: `'@property\ndef '` |
| 'x' | '\n' | 4697 | '\n' / ' *' / '1' / '2' / '3' | prefix: `'try:\n    '` |
| 'e' | '\n' | 2410 | '\n' / '4' / '1' / '2' / '3' | prefix: `'except Exception as '` |
| 'x' | 'assert' | 5 | 'assert' / 'Assert' / '\n' / 'b' / 'Ass' | prefix: `'assert '` |
| 'x' | '\n' | 9 | '\n' / ' You' / ' This' / 'y' / ' Tw' | prefix: `'yield '` |
| 'main' | '\n' | 158 | '\n' / ' You' / 'no' / 'what' / '\\' | prefix: `'async def '` |
| 'sleep' | '\n' | 3338 | '\n' / '\n ' / '<file_sep>' / '\n\n' / ' This' | prefix: `'await asyncio.'` |
| 'fixture' | ' For' | 5235 | ' For' / ' :)' / ' If' / ' The' / ' And' | prefix: `'pytest.'` |
| 'fixture' | 'icorp' | 3379 | 'icorp' / ' @_;' / ' For' / ' :)' / '\n\t' | prefix: `'@pytest.'` |
| 'Optional' | '\n' | 4002 | '\n' / ' (' / ' This' / '\n\n' / ' :)' | prefix: `'typing.'` |
| 'file' | 'dirname' | 10 | 'dirname' / 'doc' / 'this' / 'self' / 'File' | prefix: `'Path(__'` |
| 'info' | '\n' | 15923 | '\n' / ' This' / '\n\n' / ' (' / '\n ' | prefix: `'logging.'` |
| 'ArgumentParser' | '\n' | 368 | '\n' / ' (' / ' ' / ' This' / '\n\n' | prefix: `'argparse.'` |
| 'dataclass' | '\n' | 33007 | '\n' / ' (' / '\n ' / '\n   ' / '1' | prefix: `'dataclasses.'` |
| 'partial' | '\n' | 20537 | '\n' / '1' / '<file_sep>' / '2' / '3' | prefix: `'functools.'` |
| 'chain' | '\n' | 1601 | '\n' / '\n ' / '\n\n' / '<file_sep>' / '\n   ' | prefix: `'itertools.'` |
| 'Ordered' | '\n' | 7493 | '\n' / ' (' / ' This' / '\n\n' / ' ' | prefix: `'collections.'` |
| 'ascii' | 'icorp' | 2927 | 'icorp' / ' [$]' / '��' / 'olate' / 'JInternalFrame' | prefix: `'string.'` |
| 'pi' | '\n' | 218 | '\n' / '\n\n' / ' This' / ' (' / ' ' | prefix: `'math.'` |
| 'sleep' | '\n' | 3513 | '\n' / ' This' / ' (' / '\n\n' / ' Web' | prefix: `'time.'` |
| 'seed' | '\n' | 5315 | '\n' / ' (' / ' This' / '\n\n' / ' ' | prefix: `'random.'` |
| 'sha' | '\n' | 1112 | '\n' / ' (' / '\n\n' / ' This' / ' ' | prefix: `'hashlib.'` |
| 'b' | '\n' | 628 | '\n' / '1' / ' This' / ' (' / ' ' | prefix: `'base64.'` |
| 'request' | '\n' | 6309 | '\n' / ' (' / ' The' / ' This' / ' :)' | prefix: `'urllib.'` |
| 'get' | '\n' | 1219 | '\n' / ' This' / ' (' / ' Web' / ' ' | prefix: `'requests.'` |
| 'Flask' | '\n' | 1943 | '\n' / ' This' / ' (' / ' ' / '\n\n' | prefix: `'flask.'` |
| 'db' | '\n' | 9165 | '\n' / ' Django' / ' Drupal' / '\n\n' / '<file_sep>' | prefix: `'django.'` |
| 'Linear' | '\n' | 9603 | '\n' / ' This' / ' (' / ' ' / '\n\n' | prefix: `'nn.'` |


## C5 adapter + source passage in context

### A_character   (48/50 hit; 2 miss)

| target | model top-1 | rank | top-5 | source / cue |
|--------|-------------|------|-------|--------------|
| ' Richard' | 'Rich' | 87 | 'Rich' / '<NAME>' / 'That' / 'Car' / 'You' | character='Richard Carstone'<br>tail: `YOUR name now will be—” “` |
| ' Har' | 'Har' | 53 | 'Har' / '<NAME>' / 'There' / 'This' / 'Have' | character='Harold Skimpole'<br>tail: `Skimpole, “to this effect: ‘` |

### B_possession   (50/50 hit; 0 miss)

_All items hit top-1._

### C_plot   (0/50 hit; 50 miss)

| target | model top-1 | rank | top-5 | source / cue |
|--------|-------------|------|-------|--------------|
| 'Ded' | '<NAME>' | 135 | '<NAME>' / '\n' / '<file_sep>' / '1' / '2' | phrase='Lady '->'Dedlock'<br>tail: `impend over yourself.”  “Well, sir?”  “Well, Lady` |
| 'G' | '<NAME>' | 585 | '<NAME>' / '\n' / '<file_sep>' / '\n\n\n\n' / '1' | phrase='Mr. '->'Guppy'<br>tail: `his companions are yet midway in theirs, that Mr.` |
| 'Ch' | '<NAME>' | 185 | '<NAME>' / '\n' / '<file_sep>' / '1' / '2' | phrase='Court of '->'Chancery'<br>tail: `ernoon some score of members of the High Court of` |
| 'T' | '<NAME>' | 367 | '<NAME>' / '1' / '2' / '\n' / '<file_sep>' | phrase='Mr. '->'Tulkinghorn'<br>tail: `eicester and his ancestors and his patrimony”—Mr.` |
| 'Bo' | '<NAME>' | 46 | '<NAME>' / '1' / '2' / '\n' / '3' | phrase='Mr. '->'Boythorn'<br>tail: `” suggested Richard.  “By my soul,” exclaimed Mr.` |
| 'Ro' | '<NAME>' | 25 | '<NAME>' / '\n' / '<file_sep>' / '1' / '<EMAIL>' | phrase='Mrs. '->'Rouncewell'<br>tail: `there is any uncommon eye in the case, it is Mrs.` |
| 'T' | '<NAME>' | 181 | '<NAME>' / '\n' / '1' / '2' / '<file_sep>' | phrase='Mr. '->'Tulkinghorn'<br>tail: `ct involuntarily starts and falls back. It is Mr.` |
| 'T' | '<NAME>' | 255 | '<NAME>' / '\n' / '’' / '1' / '<|endoftext|>' | phrase='Mr. '->'Tulkinghorn'<br>tail: `s and ends together than if she had meant it. Mr.` |
| 'Ro' | '<NAME>' | 26 | '<NAME>' / '\n' / '1' / '<file_sep>' / '2' | phrase='Mrs. '->'Rouncewell'<br>tail: `s he got better.  “Where is your son George, Mrs.` |
| 'Ded' | '<NAME>' | 216 | '<NAME>' / '\n' / '1' / '<file_sep>' / '2' | phrase='Lady '->'Dedlock'<br>tail: `r by it, and reads, boldly written in each, “Lady` |
| 'Sn' | '<NAME>' | 26 | '<NAME>' / '\n' / '1' / '2' / '<file_sep>' | phrase='Mr. '->'Snagsby'<br>tail: `d to me, he wos!”  As he shuffles downstairs, Mr.` |
| 'G' | '<NAME>' | 235 | '<NAME>' / '<file_sep>' / '\n' / '1' / '2' | phrase='Mr. '->'Guppy'<br>tail: `rty! The property! Property!”  Mr. Weevle and Mr.` |
| 'Sn' | '<NAME>' | 25 | '<NAME>' / '\n' / '1' / '2' / '<file_sep>' | phrase='Mr. '->'Snagsby'<br>tail: `ticularity express, but she knows that Jo was Mr.` |
| 'G' | '<NAME>' | 84 | '<NAME>' / '\n' / '1' / '<file_sep>' / '\n\n\n\n' | phrase='Mr. '->'Guppy'<br>tail: `ely strolling down a flat country to the sea. Mr.` |
| 'T' | '<NAME>' | 203 | '<NAME>' / '\n' / '<file_sep>' / '\n\n\n\n' / '1' | phrase='Mr. '->'Tulkinghorn'<br>tail: `s ever. Yet it may be that my Lady fears this Mr.` |
| 'Ded' | '<NAME>' | 218 | '<NAME>' / '\n' / '<file_sep>' / '1' / '<EMAIL>' | phrase='Lady '->'Dedlock'<br>tail: `aims one morning to the listening earth that Lady` |
| 'Ded' | '<NAME>' | 99 | '<NAME>' / '\n' / '<file_sep>' / '1' / '2' | phrase='Lady '->'Dedlock'<br>tail: `h a foil in his old-fashioned rusty black to Lady` |
| 'G' | '<NAME>' | 331 | '<NAME>' / '1' / '\n' / '2' / '<file_sep>' | phrase='Mr. '->'Guppy'<br>tail: `ou wouldn’t object to admit that, miss?” said Mr.` |
| 'W' | '<NAME>' | 283 | '<NAME>' / '\n' / '1' / '<file_sep>' / '2' | phrase='Chesney '->'Wold'<br>tail: `e. You have often carried my spare gun at Chesney` |
| 'Sn' | '<NAME>' | 24 | '<NAME>' / '\n' / '1' / '<file_sep>' / '2' | phrase='Mr. '->'Snagsby'<br>tail: `and, my love?”  “At six,” says Mrs. Snagsby.  Mr.` |
| 'K' | '<NAME>' | 67 | '<NAME>' / '\n' / '1' / '<file_sep>' / '2' | phrase='Mr. '->'Krook'<br>tail: `is cough of mild persuasion behind his hand. “Mr.` |
| 'J' | '<NAME>' | 78 | '<NAME>' / '\n' / '<file_sep>' / ' J' / '1' | phrase='Caddy '->'Jellyby'<br>tail: `ove with somebody and being ashamed of it.  Caddy` |
| 'Bucket' | '<NAME>' | 16 | '<NAME>' / '\n' / '<file_sep>' / ' Bucket' / '1' | phrase='Inspector '->'Bucket'<br>tail: `hat I’m a-going to do. You know me. I’m Inspector` |
| 'K' | '<NAME>' | 74 | '<NAME>' / '\n' / '<file_sep>' / '1' / '2' | phrase='Mr. '->'Krook'<br>tail: `Run, Flite, run! The nearest doctor! Run!” So Mr.` |
| 'Sn' | '<NAME>' | 19 | '<NAME>' / '\n' / '1' / '<file_sep>' / ' Sn' | phrase='Mr. '->'Snagsby'<br>tail: `h a nod.  “About a year and a half ago,” says Mr.` |
| 'Bo' | '<NAME>' | 50 | '<NAME>' / '1' / '2' / '<file_sep>' / '\n' | phrase='Mr. '->'Boythorn'<br>tail: `, who was so tame that he was brought down by Mr.` |
| 'W' | '<NAME>' | 111 | '<NAME>' / '<file_sep>' / '\n' / '1' / '\n\n\n\n' | phrase='Chesney '->'Wold'<br>tail: `knows where to have her. To-day she is at Chesney` |
| 'Jar' | '<NAME>' | 7 | '<NAME>' / '\n' / ' Jar' / '’' / '1' | phrase='Jarndyce and '->'Jarndyce'<br>tail: `e liberty? Your servant, sir. I know Jarndyce and` |
| 'Bo' | '<NAME>' | 37 | '<NAME>' / '\n' / '1' / '<file_sep>' / '2' | phrase='Mr. '->'Boythorn'<br>tail: `t actions against HIM for trespass,” returned Mr.` |
| 'Bo' | '<NAME>' | 122 | '<NAME>' / '1' / '2' / '<file_sep>' / '\n' | phrase='Mr. '->'Boythorn'<br>tail: `room, alighted on his master’s head. To hear Mr.` |
| 'Bucket' | '<NAME>' | 18 | '<NAME>' / '\n' / '<file_sep>' / ' Bucket' / '<KEY>' | phrase='Inspector '->'Bucket'<br>tail: `unlikely that you may inquire of me why Inspector` |
| 'W' | '<NAME>' | 143 | '<NAME>' / '<file_sep>' / '\n' / '<KEY>' / ' W' | phrase='Chesney '->'Wold'<br>tail: `ster and the baronetcy, Sir Leicester and Chesney` |
| 'Jar' | '<NAME>' | 10 | '<NAME>' / '\n' / '1' / '<|endoftext|>' / '<file_sep>' | phrase='Jarndyce and '->'Jarndyce'<br>tail: `as promised a new rocking-horse when Jarndyce and` |
| 'J' | '<NAME>' | 49 | '<NAME>' / '\n' / '<file_sep>' / '1' / '2' | phrase='Mrs. '->'Jellyby'<br>tail: `n’t think she did. I therefore supposed that Mrs.` |
| 'Sum' | '<NAME>' | 33 | '<NAME>' / '\n' / '<file_sep>' / '1' / '\n\n\n\n' | phrase='Esther '->'Summerson'<br>tail: `nce. I was left in charge of a child named Esther` |
| 'Ded' | '<NAME>' | 614 | '<NAME>' / '\n' / '<file_sep>' / '\n\n\n\n' / '1' | phrase='Sir Leicester '->'Dedlock'<br>tail: `re to-day.”  “The visitor, my dear! Sir Leicester` |
| 'Ded' | '<NAME>' | 841 | '<NAME>' / '\n' / '<file_sep>' / '1' / '\n\n\n\n' | phrase='Sir Leicester '->'Dedlock'<br>tail: `o a relation of the family, a great Sir Leicester` |
| 'Sk' | '<NAME>' | 20 | '<NAME>' / '\n' / ' Sk' / '<file_sep>' / '1' | phrase='Harold '->'Skimpole'<br>tail: `y don’t stop, why should I? There you have Harold` |
| 'Ded' | '<NAME>' | 280 | '<NAME>' / '\n' / '1' / '<file_sep>' / '<EMAIL>' | phrase='Sir Leicester '->'Dedlock'<br>tail: `shall peep in from the outerside.  “Sir Leicester` |
| 'W' | '<NAME>' | 212 | '<NAME>' / '\n' / '1' / '<file_sep>' / '2' | phrase='Chesney '->'Wold'<br>tail: `d he might have the good fortune to be at Chesney` |
| 'J' | '<NAME>' | 46 | '<NAME>' / '\n' / '<file_sep>' / '1' / '2' | phrase='Mrs. '->'Jellyby'<br>tail: `ompany.  We duly came back to breakfast, and Mrs.` |
| 'K' | '<NAME>' | 45 | '<NAME>' / '\n' / '<file_sep>' / '1' / '2' | phrase='Mr. '->'Krook'<br>tail: `r. He soon returns with the intelligence that Mr.` |
| 'K' | '<NAME>' | 275 | '<NAME>' / '\n' / '<file_sep>' / '1' / '\n\n\n\n' | phrase='Mr. '->'Krook'<br>tail: `How do you do, sir? You are looking charming, Mr.` |
| 'Ded' | '<NAME>' | 186 | '<NAME>' / '\n' / '<file_sep>' / '1' / '\n\n\n\n' | phrase='Sir Leicester '->'Dedlock'<br>tail: `hundred.  “That is, I am deputed by Sir Leicester` |
| 'Ro' | '<NAME>' | 25 | '<NAME>' / '1' / '\n' / '2' / '<file_sep>' | phrase='Mrs. '->'Rouncewell'<br>tail: `e for ever and a day.  “He shall have,” says Mrs.` |
| 'Ro' | '<NAME>' | 28 | '<NAME>' / '1' / '\n' / '<file_sep>' / '2' | phrase='Mrs. '->'Rouncewell'<br>tail: `and with a profound curtsy.  “How do you do, Mrs.` |
| 'Ro' | '<NAME>' | 48 | '<NAME>' / '1' / '<file_sep>' / '\n' / '2' | phrase='George '->'Rouncewell'<br>tail: `run glistening down her sun-brown face.  “George` |
| 'J' | '\n' | 138 | '\n' / '<NAME>' / '<file_sep>' / '\n\n\n\n' / '1' | phrase='Mrs. '->'Jellyby'<br>tail: `r that he called the matrimonial alliance of Mrs.` |
| 'J' | '<NAME>' | 238 | '<NAME>' / '1' / '\n' / '2' / '<file_sep>' | phrase='Mrs. '->'Jellyby'<br>tail: `quite well.  “Why, not quite, my dear,” said Mrs.` |
| 'Ro' | '\n' | 37 | '\n' / '<NAME>' / '<file_sep>' / '\n\n\n\n' / ' Ro' | phrase='George '->'Rouncewell'<br>tail: `ed to remember you.”  “When I look at you, George` |

### D_code   (26/50 hit; 24 miss)

| target | model top-1 | rank | top-5 | source / cue |
|--------|-------------|------|-------|--------------|
| 'numpy' | '\n' | 220 | '\n' / ' scipy' / ' matplotlib' / '<jupyter_code>' / '<file_sep>' | prefix: `'import '` |
| 'os' | '\n' | 3803 | '\n' / 'irc' / 'ime' / 'ical' / 'ico' | prefix: `'import '` |
| 'default' | '\n' | 2790 | '\n' / '<file_sep>' / ' OrderedDict' / '<jupyter_code>' / '<|endoftext|>' | prefix: `'from collections import '` |
| 'List' | ' Tuple' | 2288 | ' Tuple' / ' Dict' / '\n' / ' Typed' / '<file_sep>' | prefix: `'from typing import '` |
| '*' | '\n' | 1067 | '\n' / '<file_sep>' / '<|endoftext|>' / '1' / '<jupyter_code>' | prefix: `'def __init__(self, '` |
| 'range' | '\n' | 83 | '\n' / '1' / '<file_sep>' / '2' / '<|endoftext|>' | prefix: `'for i in '` |
| "'" | '\n' | 33 | '\n' / '<file_sep>' / '1' / '<|endoftext|>' / '0' | prefix: `'with open(path, '` |
| 'None' | '1' | 472 | '1' / '0' / '2' / '3' / '\n' | prefix: `'return '` |
| 'ValueError' | '\n' | 41 | '\n' / ' #' / '1' / '<file_sep>' / ' ValueError' | prefix: `'raise '` |
| '_' | '\n' | 20636 | '\n' / 'get' / 'set' / 'add' / 'update' | prefix: `'self.'` |
| 'Logger' | '_' | 8 | '_' / 'logger' / '\n' / 'Level' / '(__' | prefix: `'logger = logging.get'` |
| 'array' | 'random' | 1 | 'random' / 'array' / 'save' / 'set' / 'linspace' | prefix: `'np.'` |
| "'" | '\n' | 2053 | '\n' / '1' / '\n   ' / 'string' / 'str' | prefix: `"'.split("` |
| 'name' | '\n' | 6298 | '\n' / '<file_sep>' / '1' / '2' / '<|endoftext|>' | prefix: `'@property\ndef '` |
| 'x' | '1' | 1326 | '1' / '0' / '2' / '\n' / '3' | prefix: `'try:\n    '` |
| 'e' | '\n' | 726 | '\n' / ' e' / '<file_sep>' / '<|endoftext|>' / '\n\n\n\n' | prefix: `'except Exception as '` |
| 'x' | '1' | 6028 | '1' / '2' / '0' / '3' / '4' | prefix: `'assert '` |
| 'x' | '1' | 2348 | '1' / '2' / '3' / '4' / '5' | prefix: `'yield '` |
| 'main' | '\n' | 8436 | '\n' / '<file_sep>' / '<|endoftext|>' / '\n\n\n\n' / '<jupyter_code>' | prefix: `'async def '` |
| 'fixture' | 'import' | 3 | 'import' / 'register' / 'mark' / 'fixture' / 'main' | prefix: `'pytest.'` |
| 'info' | 'basic' | 1 | 'basic' / 'info' / 'getLogger' / 'config' / 'debug' | prefix: `'logging.'` |
| 'partial' | 'wrap' | 1 | 'wrap' / 'partial' / 'lru' / 'total' / 'sing' | prefix: `'functools.'` |
| 'ascii' | 'upper' | 5 | 'upper' / 'lower' / '\n' / 'join' / 'up' | prefix: `'string.'` |
| 'pi' | 'sin' | 1 | 'sin' / 'pi' / 'sqrt' / 'cos' / 'log' | prefix: `'math.'` |


## C6 anti-suppression prompt only (no adapter, no engram)

### A_character   (0/50 hit; 50 miss)

| target | model top-1 | rank | top-5 | source / cue |
|--------|-------------|------|-------|--------------|
| ' T' | ' ' | 16 | ' ' / ' W' / ' M' / ' [' / ' B' | character='Tulkinghorn'<br>tail: `t same moment, there happens to be an old man of the name of` |
| ' All' | ' you' | 659 | ' you' / ' the' / ' I' / ' a' / ' he' | character='Allan Woodcourt'<br>tail: `The broken footway is so narrow that when` |
| ' Est' | ' ' | 409 | ' ' / ' M' / ' L' / ' J' / ' S' | character='Esther Summerson'<br>tail: `here I was a day boarder, and although they called me little` |
| ' Ge' | ' good' | 1728 | ' good' / ' a' / ' the' / ' so' / ' true' | character='George Rouncewell'<br>tail: `“Good heaven, and it is really` |
| ' Ada' | ' ' | 305 | ' ' / ' M' / 'iss' / 'our' / ' Car' | character='Ada Clare'<br>tail: `Richard Carstone and of Miss` |
| ' Richard' | ' was' | 431 | ' was' / ' is' / ',' / ' had' / ' in' | character='Richard Carstone'<br>tail: `gentleman was her distant cousin, she told me, and his name` |
| ' S' | ' she' | 191 | ' she' / ' as' / ' the' / ' in' / ' there' | character='Sir Leicester'<br>tail: `sper still goes about that she had not even family; howbeit,` |
| ' K' | ' ' | 78 | ' ' / ' s' / ' I' / ' my' / ' M' | character='Krook'<br>tail: `raciously observed to him before passing out, “That will do,` |
| ' Har' | ' your' | 1222 | ' your' / ' the' / ' us' / ' me' / ' it' | character='Harold Skimpole'<br>tail: `, holiness, commerce, trade, any object you prefer; only—let` |
| ' H' | ' should' | 26 | ' should' / ',' / ' is' / ' was' / ' de' | character='Hortense'<br>tail: `In short, it is such an admirable thing that Mademoiselle` |
| ' Ins' | ' ' | 828 | ' ' / ' not' / ' a' / ' no' / ' Mr' | character='Inspector Bucket'<br>tail: `ng that may appear to be disagreeable in this, for my name’s` |
| ' John' | ' the' | 30 | ' the' / ' ' / ' a' / ' your' / ' my' | character='John Jarndyce'<br>tail: `” “Since you refer so immediately to` |
| ' C' | ' the' | 54 | ' the' / ' my' / ' ' / ' him' / ' a' | character='Caddy Jellyby'<br>tail: `So I thought one day when I went to London to meet` |
| ' L' | ' mother' | 33 | ' mother' / ' father' / ' heart' / ' w' / ' de' | character='Lady Dedlock'<br>tail: `dded to these, soon floated her upward, and for years now my` |
| ' T' | ' he' | 86 | ' he' / ' the' / ' ' / ' a' / ' if' | character='Tulkinghorn'<br>tail: `now observes from his couch that man told him ya’as’dy that` |
| ' All' | ' it' | 1067 | ' it' / ' the' / ' I' / ' he' / ' one' | character='Allan Woodcourt'<br>tail: `ome excrescence produced there in neglect and impurity, that` |
| ' Est' | ',' | 861 | ',' / '.' / 'us' / 'our' / ' ' | character='Esther Summerson'<br>tail: `We are, Madam, Your obedt Servts, Kenge and Carboy Miss` |
| ' Ge' | ' and' | 252 | ' and' / ' but' / ' the' / ' for' / ' all' | character='George Rouncewell'<br>tail: `“All is still in readiness,` |
| ' Richard' | ' a' | 1839 | ' a' / ' the' / ' an' / ' his' / ' her' | character='Richard Carstone'<br>tail: `and released her, and then he spoke for a minute or two with` |
| ' S' | ' the' | 53 | ' the' / ' me' / ' his' / ' my' / ' those' | character='Sir Leicester'<br>tail: `n air of prescription about him which is always agreeable to` |
| ' K' | ' of' | 256 | ' of' / ' or' / ' and' / ' who' / ' ' | character='Krook'<br>tail: `as is announced in paint, to all whom it may concern, by one` |
| ' Har' | ' been' | 1796 | ' been' / ' a' / ' had' / ' the' / ' no' | character='Harold Skimpole'<br>tail: `Then, for heaven’s sake, having` |
| ' H' | ' the' | 24 | ' the' / ' her' / ' a' / ' my' / ' ' | character='Hortense'<br>tail: `to attend,” says my Lady then, addressing the reflection of` |
| ' Ins' | ' it' | 1727 | ' it' / ' she' / ' the' / ' you' / ' he' | character='Inspector Bucket'<br>tail: `y about admitting of it, you tell her that it’s no use, that` |
| ' John' | ' the' | 89 | ' the' / ' me' / ' ' / ' your' / ' him' | character='John Jarndyce'<br>tail: `I suppose your loyalty to` |
| ' C' | ' the' | 46 | ' the' / ' we' / ' ' / ' a' / ',' | character='Caddy Jellyby'<br>tail: `At last we came to Soho Square, where` |
| ' L' | ' de' | 6 | ' de' / ' mother' / ' l' / ' friend' / ' heart' | character='Lady Dedlock'<br>tail: `With all her perfections on her head, my` |
| ' T' | ' world' | 93 | ' world' / ' past' / ' anc' / ' earth' / ' land' | character='Tulkinghorn'<br>tail: `ined to add the last great secret to the many secrets of the` |
| ' All' | ' her' | 332 | ' her' / ' him' / ' them' / ' the' / ',' | character='Allan Woodcourt'<br>tail: `s, a farewell to her, and takes his creeping way along after` |
| ' Est' | ' ' | 914 | ' ' / ' boy' / ' man' / ',' / ' g' | character='Esther Summerson'<br>tail: `And yet I—I, little` |
| ' Ge' | ' was' | 232 | ' was' / ' were' / ' is' / ' ' / ' the' | character='George Rouncewell'<br>tail: `Very familiar to him, as he said himself some hours ago,` |
| ' Richard' | '<NAME>' | 18188 | '<NAME>' / 'I' / '—' / 'No' / 'My' | character='Richard Carstone'<br>tail: `YOUR name now will be—” “` |
| ' S' | ' the' | 4 | ' the' / ' ' / ' he' / ' she' / ' S' | character='Sir Leicester'<br>tail: `“Better now,” quoth` |
| ' K' | ' the' | 100 | ' the' / ' if' / ' it' / ' a' / ' though' | character='Krook'<br>tail: `The welcome light soon shines upon the wall, as` |
| ' Har' | ' the' | 717 | ' the' / ' itself' / ' us' / ' a' / ' any' | character='Harold Skimpole'<br>tail: `Mankind will surely not deny to` |
| ' H' | ',' | 22 | ',' / '.' / ' to' / ' de' / ' and' | character='Hortense'<br>tail: `t, a peaceful figure too in the landscape, went Mademoiselle` |
| ' Ins' | ' the' | 2010 | ' the' / ' her' / ' a' / ' your' / ' ' | character='Inspector Bucket'<br>tail: `Put it to her ladyship, if you think it right, from` |
| ' John' | ' the' | 174 | ' the' / ' me' / ' your' / ' my' / ' ' | character='John Jarndyce'<br>tail: `” “There you come back to` |
| ' C' | ' ' | 22 | ' ' / ' her' / ' you' / ' that' / ' she' | character='Caddy Jellyby'<br>tail: `r was a greater imposter than I with a blinder follower than` |
| ' L' | ' name' | 99 | ' name' / ' friend' / ' life' / ' father' / ' own' | character='Lady Dedlock'<br>tail: `le circumstance to be noted in everything associated with my` |
| ' T' | ' ' | 15 | ' ' / ' W' / ' M' / ' H' / ' Mr' | character='Tulkinghorn'<br>tail: `en a murder in Lincoln’s Inn Fields—gentleman of the name of` |
| ' All' | ' he' | 276 | ' he' / ' soon' / ' the' / ' Jo' / ' I' | character='Allan Woodcourt'<br>tail: `CHAPTER XLVII Jo’s Will As` |
| ' Est' | ' ' | 185 | ' ' / ' __' / ' M' / ' [' / ' J' | character='Esther Summerson'<br>tail: `I was left in charge of a child named` |
| ' S' | ' the' | 98 | ' the' / ' there' / ' it' / ' I' / ' all' | character='Sir Leicester'<br>tail: `re is any superabundant life of imagination on the spot, for` |
| ' K' | ' ' | 21 | ' ' / ' the' / ' M' / ' Mr' / ' B' | character='Krook'<br>tail: `I don’t know,” says` |
| ' Har' | 'I' | 13519 | 'I' / 'The' / 'If' / 'It' / 'We' | character='Harold Skimpole'<br>tail: `Skimpole, “to this effect: ‘` |
| ' H' | ',' | 48 | ',' / '.' / ',”' / '!' / '.”' | character='Hortense'<br>tail: `“Thank you, Mademoiselle` |
| ' Ins' | ' the' | 4642 | ' the' / ' a' / ' not' / ' no' / ' all' | character='Inspector Bucket'<br>tail: `single moment in the course of this prolonged night, here is` |
| ' John' | ' you' | 144 | ' you' / ' the' / ' your' / ' my' / ' any' | character='John Jarndyce'<br>tail: `se that I have come here to make underhanded charges against` |
| ' C' | ' my' | 18 | ' my' / ' the' / ' ' / ' her' / ' a' | character='Caddy Jellyby'<br>tail: `happened that when I came home from Deal I found a note from` |

### B_possession   (4/50 hit; 46 miss)

| target | model top-1 | rank | top-5 | source / cue |
|--------|-------------|------|-------|--------------|
| ' c' | ' s' | 6 | ' s' / ' h' / ' little' / ' b' / ' cup' | noun='candle'<br>tail: `an arm to Ada and an arm to me, and bidding Richard bring a` |
| ' spect' | ' eyes' | 67 | ' eyes' / ' tear' / ' own' / ' g' / ' f' | noun='spectacles'<br>tail: `Pardiggle, who had been regarding him through her` |
| ' letter' | ' fact' | 161 | ' fact' / ' loss' / ' death' / ' new' / ' good' | noun='letter'<br>tail: `er Smallweed smiles in a very ugly way in recognition of the` |
| ' spect' | ' hat' | 36 | ' hat' / ' clo' / ' co' / ' hel' / ' j' | noun='spectacles'<br>tail: `Tulkinghorn gets up, adjusts his` |
| ' glo' | ' hand' | 96 | ' hand' / ' ar' / ' hands' / ' can' / ' s' | noun='gloves'<br>tail: `urveydrop, standing with his back to the fire and waving his` |
| ' bon' | 'self' | 17 | 'self' / ' sh' / ' hair' / ' clo' / ' apr' | noun='bonnet'<br>tail: `ng easily anywhere, she perches on a rough bench, unties her` |
| ' letter' | ' story' | 6 | ' story' / ' conversation' / ' book' / ' first' / ' speech' | noun='letter'<br>tail: `This was the substance of the` |
| ' hat' | ' co' | 5 | ' co' / ' h' / ' cap' / ' s' / ' clo' | noun='hat'<br>tail: `stops of his own accord, and Sir Leicester, pulling off his` |
| ' spect' | ' hands' | 100 | ' hands' / ' head' / ' ar' / ' s' / ' eyes' | noun='spectacles'<br>tail: `Rustily drest, with his` |
| ' letter' | ' air' | 30 | ' air' / ' way' / ' past' / ' manner' / ' course' | noun='letter'<br>tail: `aid a visit of a few hours to London, which something in the` |
| ' spect' | ' d' | 121 | ' d' / ' hat' / ' clo' / ' sh' / ' co' | noun='spectacles'<br>tail: `housekeeper at Chesney Wold, has several times taken off her` |
| ' boot' | ' own' | 1793 | ' own' / ' art' / ' collection' / ' old' / '.' | noun='boots'<br>tail: `of books and papers and in part quite a little museum of his` |
| ' clo' | ' hat' | 3 | ' hat' / ' co' / ' cap' / ' clo' / ' sh' | noun='cloak'<br>tail: `, disregarding my remonstrances, had hurriedly taken off his` |
| ' hat' | ' s' | 52 | ' s' / ' white' / ' pair' / ' sh' / ' t' | noun='hat'<br>tail: `A man yet dark and muddy, in long swollen sodden boots and a` |
| ' clo' | ' blank' | 1 | ' blank' / ' clo' / ' same' / ' ro' / ' most' | noun='cloak'<br>tail: `d in a moment, and they took me between them, wrapped in the` |
| ' bon' | ' best' | 76 | ' best' / ' d' / ' usual' / ' new' / '\n' | noun='bonnet'<br>tail: `and joined Miss Jellyby, who was by this time putting on her` |
| ' ' | ' same' | 41 | ' same' / ' words' / ' most' / ' name' / ' word' | noun='umbrella'<br>tail: `Bagnet expresses with the` |
| ' lan' | ' river' | 225 | ' river' / ' water' / '\n' / ' ch' / ' dark' | noun='lantern'<br>tail: `The old man stopped, looked hard at us, looked down into the` |
| ' bon' | ' hat' | 91 | ' hat' / ' p' / ' s' / ' sh' / ' c' | noun='bonnet'<br>tail: `, in a womanly sort of manner belonging to the apron and the` |
| ' book' | ' hands' | 10 | ' hands' / ' p' / ' hand' / ' own' / ' de' | noun='book'<br>tail: `” Having put the letters in his` |
| ' bon' | ' w' | 849 | ' w' / ' s' / ' g' / ' f' / ' b' | noun='bonnet'<br>tail: `But that there’s the wale, the` |
| ' c' | ' face' | 42 | ' face' / ' eyes' / ' way' / ' b' / ' back' | noun='candle'<br>tail: `Winking cousins, bat-like in the` |
| ' lan' | ' sun' | 256 | ' sun' / ' g' / ' stars' / ' little' / ' f' | noun='lantern'<br>tail: `I could see, from my window, the` |
| ' c' | ' train' | 29 | ' train' / ' road' / ' book' / ' name' / ' whole' | noun='candle'<br>tail: `Now, Mademoiselle Hortense, let me recommend you to take the` |
| ' book' | ' eyes' | 96 | ' eyes' / ' mind' / ' thoughts' / ' face' / ' head' | noun='book'<br>tail: `He was lost in thought, his` |
| ' pur' | ' point' | 615 | ' point' / ' good' / ' note' / ' most' / ' jo' | noun='purse'<br>tail: `fair Dedlock delivers in her youthful manner, while making a` |
| ' watch' | ' scene' | 12 | ' scene' / ' other' / ' point' / ' side' / ' run' | noun='watch'<br>tail: `der is done; so, now she sees that when he used to be on the` |
| ' book' | ' di' | 3 | ' di' / ' file' / ' register' / ' book' / ' ledger' | noun='book'<br>tail: `es softly into the back office, refers to the entries in the` |
| ' clo' | ' deep' | 298 | ' deep' / ' step' / ' long' / ' look' / ' bre' | noun='cloak'<br>tail: `Now, you see, George”—he takes a` |
| ' lan' | ' great' | 2032 | ' great' / ' world' / ' s' / ' w' / ' h' | noun='lantern'<br>tail: `able brief, and outwardly directing his contemplation to the` |
| ' ' | ' hands' | 42 | ' hands' / ' hand' / ' own' / ' life' / ' claim' | noun='umbrella'<br>tail: `ticular lady whose lord is more than suspected of laying his` |
| ' lan' | ' book' | 197 | ' book' / ' letter' / ' box' / ' paper' / ' hand' | noun='lantern'<br>tail: `see, I have so many things here,” he resumed, holding up the` |
| ' book' | ' certain' | 27 | ' certain' / ' letter' / ' report' / ' statement' / ' case' | noun='book'<br>tail: `ixth volume of the Philosophical Transactions; and also of a` |
| ' ' | ' hand' | 48 | ' hand' / ' f' / ' right' / ' left' / ' sh' | noun='umbrella'<br>tail: `ving the trooper a great poke between the shoulders with her` |
| ' ' | ' old' | 6 | ' old' / ' empty' / ' un' / ' in' / ' over' | noun='umbrella'<br>tail: `er quarter of the world—with nothing but a grey cloak and an` |
| ' boot' | ' face' | 273 | ' face' / ' hands' / ' eyes' / ' heart' / ' hair' | noun='boots'<br>tail: `kind and gentle, and as he stood before the fire warming his` |
| ' c' | ' world' | 51 | ' world' / ' present' / ' place' / ' d' / ' presence' | noun='candle'<br>tail: `athing lulls or his fixed eyes show any consciousness of the` |
| ' sh' | ' head' | 39 | ' head' / ' hair' / ' face' / ' figure' / ' hand' | noun='shawl'<br>tail: `stal upon the terrace, and a vase upon the pedestal, and her` |
| ' watch' | ' own' | 540 | ' own' / ' name' / ' friend' / ' father' / ' companion' | noun='watch'<br>tail: `l, Jarndyce,” returned his guest, who seemed to refer to his` |
| ' boot' | ' feet' | 2 | ' feet' / ' sh' / ' boot' / ' s' / ' le' | noun='boots'<br>tail: `Bucket thoughtfully came and warmed the soles of his` |
| ' watch' | ' friend' | 260 | ' friend' / ' bro' / ' w' / ' father' / ' companion' | noun='watch'<br>tail: `Tulkinghorn, muttering reproof to his` |
| ' pur' | ' father' | 578 | ' father' / ' w' / ' bro' / ' family' / ' mother' | noun='purse'<br>tail: `him that during the vacation and while things are slack, his` |
| ' clo' | ' look' | 198 | ' look' / ' few' / ' gl' / ' quick' / ' little' | noun='cloak'<br>tail: `o the hotel and wait until he joined me there, so he threw a` |
| ' watch' | ' w' | 2 | ' w' / ' face' / ' watch' / ' son' / ' da' | noun='watch'<br>tail: `“Now, little housewife,” said my guardian, looking at his` |
| ' stick' | ' f' | 81 | ' f' / ' s' / ' foot' / ' opponent' / ' head' | noun='stick'<br>tail: `Boythorn in a violent burst and stopping to strike his` |
| ' pur' | ' p' | 31 | ' p' / ' pen' / ' notebook' / ' watch' / ' old' | noun='purse'<br>tail: `eorge, my considerate friend,” returns Allan, taking out his` |

### C_plot   (0/50 hit; 50 miss)

| target | model top-1 | rank | top-5 | source / cue |
|--------|-------------|------|-------|--------------|
| 'Ded' | '<NAME>' | 18773 | '<NAME>' / '’' / '1' / '\n' / '2' | phrase='Lady '->'Dedlock'<br>tail: `impend over yourself.”  “Well, sir?”  “Well, Lady` |
| 'G' | '<NAME>' | 20109 | '<NAME>' / '\n' / '2' / '1' / '3' | phrase='Mr. '->'Guppy'<br>tail: `his companions are yet midway in theirs, that Mr.` |
| 'Ch' | '<NAME>' | 5899 | '<NAME>' / '1' / '\n' / '2' / '4' | phrase='Court of '->'Chancery'<br>tail: `ernoon some score of members of the High Court of` |
| 'T' | '<NAME>' | 17260 | '<NAME>' / '1' / '2' / '3' / '\n' | phrase='Mr. '->'Tulkinghorn'<br>tail: `eicester and his ancestors and his patrimony”—Mr.` |
| 'Bo' | '<NAME>' | 1785 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Mr. '->'Boythorn'<br>tail: `” suggested Richard.  “By my soul,” exclaimed Mr.` |
| 'Ro' | '<NAME>' | 4600 | '<NAME>' / '\n' / '1' / '’' / ' M' | phrase='Mrs. '->'Rouncewell'<br>tail: `there is any uncommon eye in the case, it is Mrs.` |
| 'T' | '<NAME>' | 32967 | '<NAME>' / '\n' / '1' / ' H' / ' M' | phrase='Mr. '->'Tulkinghorn'<br>tail: `ct involuntarily starts and falls back. It is Mr.` |
| 'T' | '<NAME>' | 26067 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Mr. '->'Tulkinghorn'<br>tail: `s and ends together than if she had meant it. Mr.` |
| 'Ro' | '<NAME>' | 269 | '<NAME>' / '?' / '\n' / '1' / '’' | phrase='Mrs. '->'Rouncewell'<br>tail: `s he got better.  “Where is your son George, Mrs.` |
| 'Ded' | '<NAME>' | 29483 | '<NAME>' / '1' / '\n' / '2' / '’' | phrase='Lady '->'Dedlock'<br>tail: `r by it, and reads, boldly written in each, “Lady` |
| 'Sn' | '<NAME>' | 5711 | '<NAME>' / '\n' / '2' / '1' / '3' | phrase='Mr. '->'Snagsby'<br>tail: `d to me, he wos!”  As he shuffles downstairs, Mr.` |
| 'G' | '<NAME>' | 19218 | '<NAME>' / '1' / '2' / '\n' / '3' | phrase='Mr. '->'Guppy'<br>tail: `rty! The property! Property!”  Mr. Weevle and Mr.` |
| 'Sn' | '<NAME>' | 254 | '<NAME>' / '\n' / '1' / '3' / '2' | phrase='Mr. '->'Snagsby'<br>tail: `ticularity express, but she knows that Jo was Mr.` |
| 'G' | '<NAME>' | 29070 | '<NAME>' / '\n' / ' and' / ' And' / '\xa0' | phrase='Mr. '->'Guppy'<br>tail: `ely strolling down a flat country to the sea. Mr.` |
| 'T' | '<NAME>' | 36904 | '<NAME>' / '\n' / '1' / '3' / '2' | phrase='Mr. '->'Tulkinghorn'<br>tail: `s ever. Yet it may be that my Lady fears this Mr.` |
| 'Ded' | '<NAME>' | 22640 | '<NAME>' / '’' / '1' / ' _' / '\n' | phrase='Lady '->'Dedlock'<br>tail: `aims one morning to the listening earth that Lady` |
| 'Ded' | '<NAME>' | 13696 | '<NAME>' / '’' / '1' / '2' / '3' | phrase='Lady '->'Dedlock'<br>tail: `h a foil in his old-fashioned rusty black to Lady` |
| 'G' | '<NAME>' | 6563 | '<NAME>' / '\n' / '1' / '2' / '\xa0' | phrase='Mr. '->'Guppy'<br>tail: `ou wouldn’t object to admit that, miss?” said Mr.` |
| 'W' | '’' | 20742 | '’' / '\n' / '1' / '2' / '<NAME>' | phrase='Chesney '->'Wold'<br>tail: `e. You have often carried my spare gun at Chesney` |
| 'Sn' | '<NAME>' | 1311 | '<NAME>' / ' and' / '1' / '\n' / '2' | phrase='Mr. '->'Snagsby'<br>tail: `and, my love?”  “At six,” says Mrs. Snagsby.  Mr.` |
| 'K' | '<NAME>' | 29870 | '<NAME>' / '\n' / '1' / '3' / '2' | phrase='Mr. '->'Krook'<br>tail: `is cough of mild persuasion behind his hand. “Mr.` |
| 'J' | '<NAME>' | 20410 | '<NAME>' / '1' / '\n' / '2' / '3' | phrase='Caddy '->'Jellyby'<br>tail: `ove with somebody and being ashamed of it.  Caddy` |
| 'Bucket' | '<NAME>' | 34131 | '<NAME>' / '1' / '0' / '2' / '4' | phrase='Inspector '->'Bucket'<br>tail: `hat I’m a-going to do. You know me. I’m Inspector` |
| 'K' | '<NAME>' | 443 | '<NAME>' / '\n' / '1' / '3' / '2' | phrase='Mr. '->'Krook'<br>tail: `Run, Flite, run! The nearest doctor! Run!” So Mr.` |
| 'Sn' | '<NAME>' | 5531 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Mr. '->'Snagsby'<br>tail: `h a nod.  “About a year and a half ago,” says Mr.` |
| 'Bo' | '<NAME>' | 2979 | '<NAME>' / '\n' / '1' / '3' / '2' | phrase='Mr. '->'Boythorn'<br>tail: `, who was so tame that he was brought down by Mr.` |
| 'W' | '’' | 18031 | '’' / '<NAME>' / '1' / '\n' / '2' | phrase='Chesney '->'Wold'<br>tail: `knows where to have her. To-day she is at Chesney` |
| 'Jar' | '<NAME>' | 846 | '<NAME>' / '’' / '1' / 'ride' / 'ate' | phrase='Jarndyce and '->'Jarndyce'<br>tail: `e liberty? Your servant, sir. I know Jarndyce and` |
| 'Bo' | '<NAME>' | 5290 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Mr. '->'Boythorn'<br>tail: `t actions against HIM for trespass,” returned Mr.` |
| 'Bo' | '<NAME>' | 280 | '<NAME>' / '\n' / '1' / '’' / '2' | phrase='Mr. '->'Boythorn'<br>tail: `room, alighted on his master’s head. To hear Mr.` |
| 'Bucket' | '<NAME>' | 31227 | '<NAME>' / '’' / '1' / '\n' / '0' | phrase='Inspector '->'Bucket'<br>tail: `unlikely that you may inquire of me why Inspector` |
| 'W' | '<NAME>' | 30666 | '<NAME>' / '’' / '1' / '\n' / ' are' | phrase='Chesney '->'Wold'<br>tail: `ster and the baronetcy, Sir Leicester and Chesney` |
| 'Jar' | '<NAME>' | 713 | '<NAME>' / '1' / '’' / ' was' / '\n' | phrase='Jarndyce and '->'Jarndyce'<br>tail: `as promised a new rocking-horse when Jarndyce and` |
| 'J' | '<NAME>' | 16624 | '<NAME>' / '’' / '\n' / '\xa0' / '1' | phrase='Mrs. '->'Jellyby'<br>tail: `n’t think she did. I therefore supposed that Mrs.` |
| 'Sum' | '<NAME>' | 26024 | '<NAME>' / '’' / '1' / 'ere' / '2' | phrase='Esther '->'Summerson'<br>tail: `nce. I was left in charge of a child named Esther` |
| 'Ded' | '<NAME>' | 30372 | '<NAME>' / '’' / 'ate' / '1' / '\n' | phrase='Sir Leicester '->'Dedlock'<br>tail: `re to-day.”  “The visitor, my dear! Sir Leicester` |
| 'Ded' | '<NAME>' | 31628 | '<NAME>' / '’' / '1' / '\n' / '2' | phrase='Sir Leicester '->'Dedlock'<br>tail: `o a relation of the family, a great Sir Leicester` |
| 'Sk' | '<NAME>' | 14515 | '<NAME>' / '’' / '1' / '4' / '2' | phrase='Harold '->'Skimpole'<br>tail: `y don’t stop, why should I? There you have Harold` |
| 'Ded' | '<NAME>' | 39127 | '<NAME>' / '1' / '’' / '2' / '3' | phrase='Sir Leicester '->'Dedlock'<br>tail: `shall peep in from the outerside.  “Sir Leicester` |
| 'W' | '’' | 41336 | '’' / 'ere' / '<NAME>' / '1' / '2' | phrase='Chesney '->'Wold'<br>tail: `d he might have the good fortune to be at Chesney` |
| 'J' | '<NAME>' | 629 | '<NAME>' / ' J' / '\n' / '\xa0' / '1' | phrase='Mrs. '->'Jellyby'<br>tail: `ompany.  We duly came back to breakfast, and Mrs.` |
| 'K' | '<NAME>' | 20989 | '<NAME>' / '\n' / '1' / '3' / '2' | phrase='Mr. '->'Krook'<br>tail: `r. He soon returns with the intelligence that Mr.` |
| 'K' | '<NAME>' | 315 | '<NAME>' / '\n' / '1' / '2' / '3' | phrase='Mr. '->'Krook'<br>tail: `How do you do, sir? You are looking charming, Mr.` |
| 'Ded' | '<NAME>' | 26805 | '<NAME>' / '’' / '1' / '2' / '\n' | phrase='Sir Leicester '->'Dedlock'<br>tail: `hundred.  “That is, I am deputed by Sir Leicester` |
| 'Ro' | '<NAME>' | 3115 | '<NAME>' / '1' / '\n' / '2' / '’' | phrase='Mrs. '->'Rouncewell'<br>tail: `e for ever and a day.  “He shall have,” says Mrs.` |
| 'Ro' | '<NAME>' | 99 | '<NAME>' / ' Ro' / '\n' / '’' / '1' | phrase='Mrs. '->'Rouncewell'<br>tail: `and with a profound curtsy.  “How do you do, Mrs.` |
| 'Ro' | '<NAME>' | 10436 | '<NAME>' / '’' / '1' / 'ate' / '2' | phrase='George '->'Rouncewell'<br>tail: `run glistening down her sun-brown face.  “George` |
| 'J' | '<NAME>' | 17220 | '<NAME>' / '\n' / '1' / ' and' / '\xa0' | phrase='Mrs. '->'Jellyby'<br>tail: `r that he called the matrimonial alliance of Mrs.` |
| 'J' | '<NAME>' | 2142 | '<NAME>' / '\n' / '1' / '\xa0' / ' J' | phrase='Mrs. '->'Jellyby'<br>tail: `quite well.  “Why, not quite, my dear,” said Mrs.` |
| 'Ro' | '<NAME>' | 11317 | '<NAME>' / '1' / '’' / '\n' / '2' | phrase='George '->'Rouncewell'<br>tail: `ed to remember you.”  “When I look at you, George` |

### D_code   (15/50 hit; 35 miss)

| target | model top-1 | rank | top-5 | source / cue |
|--------|-------------|------|-------|--------------|
| 'numpy' | '<NAME>' | 994 | '<NAME>' / '\n' / '1' / '3' / '2' | prefix: `'import '` |
| 'os' | '<NAME>' | 3525 | '<NAME>' / '\n' / '1' / '3' / '2' | prefix: `'import '` |
| 'default' | ' defaultdict' | 827 | ' defaultdict' / ' Counter' / ' name' / '\n' / ' deque' | prefix: `'from collections import '` |
| 'List' | '\n' | 2349 | '\n' / ' List' / ' Union' / ' Dict' / ' Any' | prefix: `'from typing import '` |
| '*' | ' name' | 23853 | ' name' / '<NAME>' / '\n' / ' first' / ' **' | prefix: `'def __init__(self, '` |
| 'range' | '1' | 132 | '1' / '0' / '2' / '3' / '5' | prefix: `'for i in '` |
| "'" | '’' | 1987 | '’' / ' encoding' / " '" / ' "' / ' mode' | prefix: `'with open(path, '` |
| 'None' | '1' | 10566 | '1' / '0' / '<NAME>' / '2' / '3' | prefix: `'return '` |
| 'ValueError' | '<NAME>' | 14097 | '<NAME>' / '1' / '2' / '\n' / '3' | prefix: `'raise '` |
| '_' | 'name' | 19115 | 'name' / '\n' / 'state' / ' The' / 'title' | prefix: `'self.'` |
| 'Logger' | 'logger' | 4 | 'logger' / '_' / 'L' / '(' / 'Logger' | prefix: `'logger = logging.get'` |
| 'Linear' | 'Conv' | 1 | 'Conv' / 'Linear' / 'functional' / 'Module' / 'init' | prefix: `'torch.nn.'` |
| 'array' | 'random' | 2 | 'random' / 'arange' / 'array' / 'nan' / 'where' | prefix: `'np.'` |
| 'run' | 'Popen' | 1 | 'Popen' / 'run' / 'call' / 'check' / 'py' | prefix: `'subprocess.'` |
| 'dumps' | 'loads' | 1 | 'loads' / 'dumps' / 'load' / 'dump' / 'parse' | prefix: `'json.'` |
| 'compile' | 'search' | 2 | 'search' / 'sub' / 'compile' / 'findall' / 'match' | prefix: `'re.'` |
| "'" | ' )' | 1325 | ' )' / '\n' / " '" / 'sep' / 'os' | prefix: `"'.split("` |
| 'name' | '<NAME>' | 8565 | '<NAME>' / '1' / '2' / ' __' / '3' | prefix: `'@property\ndef '` |
| 'x' | '1' | 34711 | '1' / '<NAME>' / '2' / '\n' / ' print' | prefix: `'try:\n    '` |
| 'e' | '<NAME>' | 12670 | '<NAME>' / '1' / '\n' / ' e' / '2' | prefix: `'except Exception as '` |
| 'x' | '<NAME>' | 41570 | '<NAME>' / '1' / '2' / '0' / '3' | prefix: `'assert '` |
| 'x' | '<NAME>' | 45593 | '<NAME>' / '1' / '2' / '3' / '5' | prefix: `'yield '` |
| 'main' | '<NAME>' | 11393 | '<NAME>' / '1' / '\n' / ' __' / '2' | prefix: `'async def '` |
| 'sleep' | 'gather' | 1 | 'gather' / 'sleep' / 'wait' / 'get' / 'create' | prefix: `'await asyncio.'` |
| 'Optional' | '\n' | 8367 | '\n' / ' The' / ' ' / ' It' / ' In' | prefix: `'typing.'` |
| 'chain' | 'groupby' | 2 | 'groupby' / 'per' / 'chain' / 'combin' / 'product' | prefix: `'itertools.'` |
| 'Ordered' | '\n' | 17 | '\n' / ' The' / 'Counter' / 'py' / 'append' | prefix: `'collections.'` |
| 'ascii' | 'format' | 21 | 'format' / '\n' / 'replace' / 'p' / 'find' | prefix: `'string.'` |
| 'pi' | 'stack' | 10 | 'stack' / 'h' / '\n' / 'org' / 'com' | prefix: `'math.'` |
| 'sleep' | '\n' | 5 | '\n' / ' The' / 'time' / ' ' / ' He' | prefix: `'time.'` |
| 'seed' | 'choice' | 2 | 'choice' / 'shuffle' / 'seed' / 'sample' / 'random' | prefix: `'random.'` |
| 'sha' | 'md' | 1 | 'md' / 'sha' / 'new' / 'pb' / '\n' | prefix: `'hashlib.'` |
| 'Flask' | 'py' | 18 | 'py' / 'request' / 'ext' / 'app' / 'json' | prefix: `'flask.'` |
| 'db' | 'contrib' | 1 | 'contrib' / 'db' / 'core' / 'utils' / 'conf' | prefix: `'django.'` |
| 'Linear' | ' ' | 132 | ' ' / 'nn' / ' The' / '\n' / ' In' | prefix: `'nn.'` |

