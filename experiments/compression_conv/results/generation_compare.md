# Generation comparison — baseline vs compressed (16x)

Aggregate coarsened-anchor accuracy across 8 prompts × 8 cycles = 64 predictions:
  - **baseline:** 15/64 = 0.234
  - **compressed:** 15/64 = 0.234

Architecture caveat: this model predicts every 16th token, not arbitrary next tokens. Side-by-side comparisons below are at the model's natural cadence.

## Prompt 0 (val position 10612, ctx 2048)

**Last 240 chars of prompt:**
```
 Democratic Party candidates , as for decades since the late nineteenth century , it was a one @-@ party state . 


 = = Demographics = = 


 The city 's growth has reflected the push and pull of many social and economic factors . The total population increased in each census from the city 's founding until 1970 , although varying from rates as high as 165 % to as
```

**Ground-truth next token:** `1877` = `' low'`

**Baseline top-5:**
```
  0.205    257  ' a'
  0.183    262  ' the'
  0.046    880  ' well'
  0.036    281  ' an'
  0.027    636  ' part'
```

**Compressed top-5:**
```
  0.230    257  ' a'
  0.216    262  ' the'
  0.044    880  ' well'
  0.034    281  ' an'
  0.024    366  ' "'
```

**Coarsened anchor predictions over 8 cycles** (each cycle predicts a token 16 positions ahead, using ground-truth tokens to fill between):

| cycle | ground truth | baseline (hit?) | compressed (hit?) |
|---|---|---|---|
| 0 | `' low'` | `' a'` (✗) | `' a'` (✗) |
| 1 | `' ,'` | `' ,'` (✓) | `' ,'` (✓) |
| 2 | `' since'` | `' in'` (✗) | `' to'` (✗) |
| 3 | `' declined'` | `' of'` (✗) | `' was'` (✗) |
| 4 | `'s'` | `'s'` (✓) | `'s'` (✓) |
| 5 | `'85'` | `' @'` (✗) | `' @'` (✗) |
| 6 | `' km'` | `' h'` (✗) | `' )'` (✗) |
| 7 | `' .'` | `' of'` (✗) | `' ,'` (✗) |

Baseline hits: 2/8  |  Compressed hits: 2/8

---

## Prompt 1 (val position 67873, ctx 2048)

**Last 240 chars of prompt:**
```
 Ichiki 's assault was repulsed with devastating losses for the attackers in what became known as the Battle of the Tenaru : all but 128 of the 917 men of the First Element ( including Ichiki himself ) were killed in the battle . The survivors returned to Taivu Point , notified 17th Army headquarters of their defeat in the battle and awaited further reinforcements and orders from Rabaul
```

**Ground-truth next token:** `764` = `' .'`

**Baseline top-5:**
```
  0.102    837  ' ,'
  0.087    764  ' .'
  0.048    290  ' and'
  0.031   1267  ' )'
  0.026    373  ' was'
```

**Compressed top-5:**
```
  0.089    764  ' .'
  0.058    837  ' ,'
  0.036    290  ' and'
  0.018    357  ' ('
  0.012    705  " '"
```

**Coarsened anchor predictions over 8 cycles** (each cycle predicts a token 16 positions ahead, using ground-truth tokens to fill between):

| cycle | ground truth | baseline (hit?) | compressed (hit?) |
|---|---|---|---|
| 0 | `' .'` | `' ,'` (✗) | `' .'` (✓) |
| 1 | `'k'` | `'a'` (✗) | `'n'` (✗) |
| 2 | `'adal'` | `'b'` (✗) | `' @'` (✗) |
| 3 | `' troop'` | `' forces'` (✗) | `' @'` (✗) |
| 4 | `' to'` | `' be'` (✗) | `' be'` (✗) |
| 5 | `' ship'` | `' of'` (✗) | `' .'` (✗) |
| 6 | `' Rab'` | `' the'` (✗) | `' the'` (✗) |
| 7 | `' men'` | `' ,'` (✗) | `' first'` (✗) |

Baseline hits: 0/8  |  Compressed hits: 1/8

---

## Prompt 2 (val position 100989, ctx 2048)

**Last 240 chars of prompt:**
```
 legal documents , and the testimony of former members , concluded that the organization was " antithetical to a democratic state " . Federal ministries and state governments were asked to use all legal means at their disposal to check the activities of Scientology . 

 Government publications on the dangers of sects increased between 1996 and 1998 , and a significant number of them dealt with the Church of Scientology . The German courts had approved such
```

**Ground-truth next token:** `16125` = `' publications'`

**Baseline top-5:**
```
  0.645    355  ' as'
  0.029    257  ' a'
  0.012    262  ' the'
  0.008    281  ' an'
  0.007    837  ' ,'
```

**Compressed top-5:**
```
  0.754    355  ' as'
  0.028    287  ' in'
  0.017    319  ' on'
  0.017    351  ' with'
  0.012    284  ' to'
```

**Coarsened anchor predictions over 8 cycles** (each cycle predicts a token 16 positions ahead, using ground-truth tokens to fill between):

| cycle | ground truth | baseline (hit?) | compressed (hit?) |
|---|---|---|---|
| 0 | `' publications'` | `' as'` (✗) | `' as'` (✗) |
| 1 | `' the'` | `' the'` (✓) | `' the'` (✓) |
| 2 | `' 1996'` | `' the'` (✗) | `' the'` (✗) |
| 3 | `' to'` | `' ,'` (✗) | `' ,'` (✗) |
| 4 | `' .'` | `' ,'` (✗) | `' .'` (✓) |
| 5 | `' new'` | `' the'` (✗) | `' the'` (✗) |
| 6 | `' An'` | `'\n'` (✗) | `'\n'` (✗) |
| 7 | `' of'` | `' .'` (✗) | `' .'` (✗) |

Baseline hits: 1/8  |  Compressed hits: 2/8

---

## Prompt 3 (val position 110250, ctx 2048)

**Last 240 chars of prompt:**
```
 Japanese guns replacing her original British @-@ made guns . The same year , she was assigned to the 5th Division of the 3rd Fleet . In 1918 , Asahi became flagship of her division and participated in the Japanese intervention in the Russian Civil War . She escorted troop convoys to the Russian Far East and was guard ship at Kamchatka from January to August 1918 . Asahi was re
```

**Ground-truth next token:** `31691` = `'classified'`

**Baseline top-5:**
```
  0.508   2488  ' @'
  0.001    400  'th'
  0.001  10860  ' tons'
  0.001    764  ' .'
  0.001    358  'nd'
```

**Compressed top-5:**
```
  0.841   2488  ' @'
  0.004   4064  ' %'
  0.003   1510  ' million'
  0.003  10571  ' km'
  0.002    400  'th'
```

**Coarsened anchor predictions over 8 cycles** (each cycle predicts a token 16 positions ahead, using ground-truth tokens to fill between):

| cycle | ground truth | baseline (hit?) | compressed (hit?) |
|---|---|---|---|
| 0 | `'classified'` | `' @'` (✗) | `' @'` (✗) |
| 1 | `' and'` | `' and'` (✓) | `' the'` (✗) |
| 2 | `' the'` | `' the'` (✓) | `' the'` (✓) |
| 3 | `' on'` | `' was'` (✗) | `' ,'` (✗) |
| 4 | `' .'` | `' ,'` (✗) | `' ,'` (✗) |
| 5 | `','` | `'-'` (✗) | `'.'` (✗) |
| 6 | `' speed'` | `' .'` (✗) | `' new'` (✗) |
| 7 | `' '` | `' '` (✓) | `' The'` (✗) |

Baseline hits: 3/8  |  Compressed hits: 1/8

---

## Prompt 4 (val position 134027, ctx 2048)

**Last 240 chars of prompt:**
```
entials of the Thousand Character Classic in Six Scripts ( Qianwen Liushu Tongyao , 千文六書統要 ) ( 1663 ) , which Hu compiled with the aid of his calligraphy teacher , Li Deng . It was published after Li 's death , partly in homage to him . 

 The three Hu brothers worked together
```

**Ground-truth next token:** `284` = `' to'`

**Baseline top-5:**
```
  0.152    351  ' with'
  0.088    764  ' .'
  0.085    837  ' ,'
  0.068    287  ' in'
  0.048    290  ' and'
```

**Compressed top-5:**
```
  0.146    287  ' in'
  0.133    284  ' to'
  0.109    351  ' with'
  0.099    416  ' by'
  0.072    355  ' as'
```

**Coarsened anchor predictions over 8 cycles** (each cycle predicts a token 16 positions ahead, using ground-truth tokens to fill between):

| cycle | ground truth | baseline (hit?) | compressed (hit?) |
|---|---|---|---|
| 0 | `' to'` | `' with'` (✗) | `' in'` (✗) |
| 1 | `' ,'` | `' ,'` (✓) | `' ,'` (✓) |
| 2 | `'�'` | `' ,'` (✗) | `' )'` (✗) |
| 3 | `' studio'` | `' same'` (✗) | `' first'` (✗) |
| 4 | `'ix'` | `'j'` (✗) | `' )'` (✗) |
| 5 | `'�'` | `'a'` (✗) | `' )'` (✗) |
| 6 | `' a'` | `' a'` (✓) | `' a'` (✓) |
| 7 | `' himself'` | `' ,'` (✗) | `' ,'` (✗) |

Baseline hits: 2/8  |  Compressed hits: 2/8

---

## Prompt 5 (val position 198693, ctx 2048)

**Last 240 chars of prompt:**
```
 book sparked much debate in leftist circles and inspired more aggressive tactics within the anti @-@ globalization movement in the following few years . 

 Agents of Repression ( 1988 ) , co @-@ authored by Jim Vander Wall , describes what the authors claim was a secret war against the Black Panther Party and American Indian Movement carried out during the late 1960s and ' 70s by the FBI under the
```

**Ground-truth next token:** `7375` = `' CO'`

**Baseline top-5:**
```
  0.008    366  ' "'
  0.008    717  ' first'
  0.007    976  ' same'
  0.005    749  ' most'
  0.005    640  ' time'
```

**Compressed top-5:**
```
  0.021    717  ' first'
  0.013    366  ' "'
  0.011   2646  ' film'
  0.010   3496  ' song'
  0.010    983  ' game'
```

**Coarsened anchor predictions over 8 cycles** (each cycle predicts a token 16 positions ahead, using ground-truth tokens to fill between):

| cycle | ground truth | baseline (hit?) | compressed (hit?) |
|---|---|---|---|
| 0 | `' CO'` | `' "'` (✗) | `' first'` (✗) |
| 1 | `'issued'` | `' @'` (✗) | `' @'` (✗) |
| 2 | `' a'` | `' ,'` (✗) | `' ,'` (✗) |
| 3 | `' groups'` | `' ,'` (✗) | `' ,'` (✗) |
| 4 | `' concerned'` | `' ,'` (✗) | `' .'` (✗) |
| 5 | `' of'` | `' ,'` (✗) | `' ,'` (✗) |
| 6 | `' the'` | `' the'` (✓) | `' the'` (✓) |
| 7 | `' as'` | `' ,'` (✗) | `' ,'` (✗) |

Baseline hits: 1/8  |  Compressed hits: 1/8

---

## Prompt 6 (val position 221360, ctx 2048)

**Last 240 chars of prompt:**
```
 confrontation with the police was exaggerated ; the group were hoping to get shut down by the authorities in order to dramatize the music video , but the police continually gave them extensions for shooting the video . In the background of the video is a sign for The Million Dollar Hotel , which was rebuilt to create some interest , in case no one showed up at the film shoot . Although the video is of a
```

**Ground-truth next token:** `2107` = `' live'`

**Baseline top-5:**
```
  0.028    366  ' "'
  0.017   2060  ' single'
  0.015   1271  ' number'
  0.010    649  ' new'
  0.008   1178  ' few'
```

**Compressed top-5:**
```
  0.023    366  ' "'
  0.013    717  ' first'
  0.008    898  ' own'
  0.007    649  ' new'
  0.006    640  ' time'
```

**Coarsened anchor predictions over 8 cycles** (each cycle predicts a token 16 positions ahead, using ground-truth tokens to fill between):

| cycle | ground truth | baseline (hit?) | compressed (hit?) |
|---|---|---|---|
| 0 | `' live'` | `' "'` (✗) | `' "'` (✗) |
| 1 | `' the'` | `' the'` (✓) | `' the'` (✓) |
| 2 | `' 1989'` | `' song'` (✗) | `' first'` (✗) |
| 3 | `' ='` | `' ='` (✓) | `' ='` (✓) |
| 4 | `'-'` | `'-'` (✓) | `'-'` (✓) |
| 5 | `' from'` | `' by'` (✗) | `' the'` (✗) |
| 6 | `' as'` | `' ,'` (✗) | `" '"` (✗) |
| 7 | `' a'` | `' .'` (✗) | `' The'` (✗) |

Baseline hits: 3/8  |  Compressed hits: 3/8

---

## Prompt 7 (val position 233372, ctx 2048)

**Last 240 chars of prompt:**
```
 , vulgar and juvenile " . Chang echoed the sentiments of The Village Voice in lamenting that the film failed to pursue the premise to " darker , more daring territory " and faulted it for falling back on " over @-@ the @-@ top comic exaggeration " . 


 = = = Accolades = = = 


 The film received several award nominations , including a Satellite
```

**Ground-truth next token:** `11289` = `' Award'`

**Baseline top-5:**
```
  0.086    366  ' "'
  0.075    837  ' ,'
  0.046    764  ' .'
  0.030    290  ' and'
  0.027   2488  ' @'
```

**Compressed top-5:**
```
  0.037   2488  ' @'
  0.011     82  's'
  0.011    837  ' ,'
  0.010    286  ' of'
  0.010    290  ' and'
```

**Coarsened anchor predictions over 8 cycles** (each cycle predicts a token 16 positions ahead, using ground-truth tokens to fill between):

| cycle | ground truth | baseline (hit?) | compressed (hit?) |
|---|---|---|---|
| 0 | `' Award'` | `' "'` (✗) | `' @'` (✗) |
| 1 | `' Awards'` | `' "'` (✗) | `' .'` (✗) |
| 2 | `' best'` | `' "'` (✗) | `' the'` (✗) |
| 3 | `'@'` | `'@'` (✓) | `'@'` (✓) |
| 4 | `' award'` | `' film'` (✗) | `' "'` (✗) |
| 5 | `' '` | `' '` (✓) | `' '` (✓) |
| 6 | `' 26'` | `' ,'` (✗) | `' 18'` (✗) |
| 7 | `' '` | `' '` (✓) | `' '` (✓) |

Baseline hits: 3/8  |  Compressed hits: 3/8

---
