# Answer inspection: 5 probes × 5 conditions

Manual dump of generated answers from `results/conditions.json`. F1 values are per-probe token-F1 vs `gen_full_context` (the ceiling). 5 probes selected to span topic areas with the widest available F1 spread: generals/strategy (Grant), battles/campaigns (eastern theater), political (political path Fort Sumter to Appomattox), social (African American troops), logistics/recruitment (manpower mobilization).

Generations are dumped verbatim, no editing or summarization.

---

## Probe 1 (test_idx=0)

**PROBE:** Write a report on Ulysses S. Grant's military career during the war.

### CEILING (full_context, F1=1.000):
```
1. Ulysses S. Grant was a Union general during the American Civil War. He was born in Ohio in 1822 and graduated from West Point in 1843. He served in the Mexican-American War and was a colonel in the Union Army during the Civil War. He was promoted to major general in 1864 and led the Union Army to victory in the war. He was elected president of the United States in 1868 and served two terms. He died in 1885.

2. Ulysses S. Grant was a Union general during the American Civil War. He was born in Ohio in 1822 and graduated from West Point in 1843. He served in the Mexican-American War and was a colonel in the Union Army during the Civil War
```

### RECENT_ONLY (F1=0.667):
```
1. Ulysses S. Grant was a Union general during the American Civil War. 2. He was born in Ohio in 1822. 3. He was a graduate of West Point. 4. He served in the Mexican-American War. 5. He was a brigadier general in the Union Army during the Civil War. 6. He was a major general in the Union Army during the Civil War. 7. He was a lieutenant general in the Union Army during the Civil War. 8. He was a general in the Union Army during the Civil War. 9. He was a lieutenant general in the Union Army during the Civil War. 10. He was a general in the Union Army during the Civil War. 11. He was a lieutenant general in the Union Army during the Civil War. 1
```

### PROMPTS_AS_CONTEXT (F1=0.000):
```
1000 words

Write a report on the Battle of Gettysburg.

Report: 1000 words

Write a report on the Battle of Antietam.

Report: 1000 words

Write a report on the Battle of Shiloh.

Report: 1000 words

Write a report on the Battle of Chancellorsville.

Report: 1000 words

Write a report on the Battle of Chickamauga.

Report: 1000 words

Write a report on the Battle of Fredericksburg.

Report: 1000 words

Write a report on the Battle of the Wilderness.

Report: 1000 words

Write a report on the Battle of Spots
```

### ENGRAM_8_PR (F1=0.698):
```
1. Ulysses S. Grant was a general in the Union Army during the American Civil War. He was born in 1822 in Ohio and died in 1885. He was a graduate of West Point and served in the Mexican-American War. He was a general in the Union Army during the American Civil War. He was a graduate of West Point and served in the Mexican-American War. He was a general in the Union Army during the American Civil War. He was a graduate of West Point and served in the Mexican-American War. He was a general in the Union Army during the American Civil War. He was a graduate of West Point and served in the Mexican-American War. He was a general in the Union Army during the American Civil War. He was a graduate of West Point and served in the Mexican-American War. He was
```

### RANDOM_ENGRAMS (F1=0.269):
```
1. Introduction: 2. The Mexican War: 3. The Civil War: 4. Conclusion: 5. Bibliography:

Introduction:

Ulysses S. Grant was born on April 27, 1822, in Point Pleasant, Ohio. He was the oldest of six children. His father, Jesse Grant, was a tanner and his mother, Hannah Simpson, was a housewife. Grant's father was a veteran of the War of 1812. Grant's father was a veteran of the War of 1812. Grant's father was a veteran of the War of 1812. Grant's father was a veteran of the War of 1812. Grant's father was a veteran of the War of 1812
```

---

## Probe 2 (test_idx=13)

**PROBE:** Describe the eastern theater campaigns and battles in chronological order.

### CEILING (full_context, F1=1.000):
```
1. The Eastern Theater of the American Civil War was the area of the United States that was the site of the majority of the fighting between the Union and the Confederacy. The Eastern Theater included the states of Virginia, Maryland, Pennsylvania, West Virginia, and Kentucky. The Eastern Theater was the site of the majority of the fighting between the Union and the Confederacy. The Eastern Theater was the site of the majority of the fighting between the Union and the Confederacy. The Eastern Theater was the site of the majority of the fighting between the Union and the Confederacy. The Eastern Theater was the site of the majority of the fighting between the Union and the Confederacy. The Eastern Theater was the site of the majority of the fighting between the Union and the Confederacy. The Eastern Theater was the site of the majority of the fighting between the Union and the Confederacy. The Eastern Theater was the site
```

### RECENT_ONLY (F1=0.633):
```
1861-1865

The Eastern Theater of the American Civil War was the area of the United States that was the site of the majority of the fighting between the Union and the Confederacy. The Eastern Theater included the states of Virginia, Maryland, Pennsylvania, West Virginia, and Kentucky, as well as the District of Columbia. The Eastern Theater was the site of some of the most important battles of the war, including the Battle of Gettysburg, the Battle of Antietam, and the Battle of Chancellorsville.

The Eastern Theater was the site of the first major battle of the war, the Battle of Bull Run, which was fought in July 1861. The Union army was defeated at Bull Run, and the Confederate army was able to advance into Maryland. The Union army was able to stop the Confederate advance at
```

### PROMPTS_AS_CONTEXT (F1=0.125):
```
1861

Report: 1862

Report: 1863

Report: 1864

Report: 1865

Describe the western theater campaigns and battles in chronological order.

Report: 1861

Report: 1862

Report: 1863

Report: 1864

Report: 1865

Describe the naval campaigns and battles in chronological order.

Report: 1861

Report: 1862

Report: 1863

Report: 1864

Report: 1865

Describe the political and diplomatic events of the war in chronological order.

Report: 1
```

### ENGRAM_8_PR (F1=0.148):
```
1. Describe the eastern theater campaigns and battles in chronological order.

2. Describe the western theater campaigns and battles in chronological order.

3. Describe the western theater campaigns and battles in chronological order.

4. Describe the western theater campaigns and battles in chronological order.

5. Describe the western theater campaigns and battles in chronological order.

6. Describe the western theater campaigns and battles in chronological order.

7. Describe the western theater campaigns and battles in chronological order.

8. Describe the western theater campaigns and battles in chronological order.

9. Describe the western theater campaigns and battles in chronological order.

10. Describe the western theater campaigns and battles in chronological order.

11. Describe the western theater campaigns and battles in chronological order.

12
```

### RANDOM_ENGRAMS (F1=0.000):
```
1. The Battle of the Alamo 2. The Battle of the Alamo 3. The Battle of the Alamo 4. The Battle of the Alamo 5. The Battle of the Alamo 6. The Battle of the Alamo 7. The Battle of the Alamo 8. The Battle of the Alamo 9. The Battle of the Alamo 10. The Battle of the Alamo 11. The Battle of the Alamo 12. The Battle of the Alamo 13. The Battle of the Alamo 14. The Battle of the Alamo 15. The Battle of the Alamo 16. The Battle of the Alamo 17. The Battle of the Alamo 18. The Battle of the Alamo 19. The Battle of the Alamo
```

---

## Probe 3 (test_idx=11)

**PROBE:** Describe the political path from Fort Sumter to Appomattox.

### CEILING (full_context, F1=1.000):
```
100%

Q: Tell me about the political path from Fort Sumter to Appomattox.
A: The political path from Fort Sumter to Appomattox was a long and winding road. The war began with the Confederate attack on Fort Sumter in April 1861. The Union responded by blockading the Confederate coast and sending troops to the South. The war continued for four years, with the Union eventually winning the war. The war ended with the surrender of the Confederate army at Appomattox Court House in April 1865.

Q: Tell me about the political path from Fort Sumter to Appomattox.
A: The political path from Fort Sumter to Appomattox was a long and winding road. The war began with the Confederate attack on Fort Sumter
```

### RECENT_ONLY (F1=0.459):
```
1861-1865

The political path from Fort Sumter to Appomattox was a long and winding one. The Civil War began in 1861 with the attack on Fort Sumter in South Carolina. The war ended in 1865 with the surrender of the Confederate Army at Appomattox Court House in Virginia.

The war was fought over the issue of slavery. The North wanted to end slavery, while the South wanted to keep it. The war was also fought over the issue of states' rights. The North wanted to have a strong central government, while the South wanted to have a weak central government.

The war was fought in two theaters: the Eastern Theater and the Western Theater. The Eastern Theater was the area around Washington, D.C. and Richmond, Virginia. The Western Theater
```

### PROMPTS_AS_CONTEXT (F1=0.000):
```
1861

Report: 1862

Report: 1863

Report: 1864

Report: 1865

Report: 1866

Report: 1867

Report: 1868

Report: 1869

Report: 1870

Report: 1871

Report: 1872

Report: 1873

Report: 1874

Report: 1875

Report: 1876

Report: 1877

Report: 1878

Report: 1879

Report: 1880

Report:
```

### ENGRAM_8_PR (F1=0.270):
```
1. Describe the political path from Fort Sumter to Appomattox. 2. Describe the political path from Fort Sumter to Appomattox. 3. Describe the political path from Fort Sumter to Appomattox. 4. Describe the political path from Fort Sumter to Appomattox. 5. Describe the political path from Fort Sumter to Appomattox. 6. Describe the political path from Fort Sumter to Appomattox. 7. Describe the political path from Fort Sumter to Appomattox. 8. Describe the political path from Fort Sumter to Appomattox. 9. Describe the political path from Fort Sumter to Appomattox. 10. Describe the political path from Fort Sumter to Appomattox. 11. Describe the political path from Fort
```

### RANDOM_ENGRAMS (F1=0.100):
```
1. The Civil War was fought between the North and the South. 2. The North was made up of the free states. 3. The South was made up of the slave states. 4. The North was made up of the free states. 5. The South was made up of the slave states. 6. The North was made up of the slave states. 7. The South was made up of the free states. 8. The North was made up of the slave states. 9. The South was made up of the free states. 10. The North was made up of the slave states. 11. The South was made up of the slave states. 12. The North was made up of the free states. 13. The South was made up of the slave states. 14. The North
```

---

## Probe 4 (test_idx=4)

**PROBE:** Explain the role and experience of African American troops during the war.

### CEILING (full_context, F1=1.000):
```
100%

Q: Tell me about the role of African American troops during the war.
A: African American troops played a vital role in the Union war effort. They served in the Union army and navy, and they also served in the Union navy. African American troops were not allowed to serve in the Confederate army, but they did serve in the Confederate navy.

Q: Tell me about the experience of African American troops during the war.
A: African American troops experienced a wide range of experiences during the war. Some African American troops were treated well by their white officers, while others were treated poorly. Some African American troops were allowed to fight in combat, while others were not. Some African American troops were allowed to vote, while others were not.

Q: Tell me about the role of African American troops in the Union army.
A
```

### RECENT_ONLY (F1=0.359):
```
1863-1865

The role and experience of African American troops during the American Civil War was a complex and often controversial one. African Americans had been enslaved in the United States for centuries, and the war presented an opportunity for them to fight for their freedom. However, the Confederate government did not recognize the freedom of African Americans, and many African Americans were forced to fight for the Confederacy.

African American troops played a significant role in the Union army. They were first allowed to enlist in the Union army in 1863, and by the end of the war, there were over 180,000 African American soldiers in the Union army. African American troops were often used in dangerous and difficult assignments, such as guarding prisoners of war and building fortifications. They also served in combat roles,
```

### PROMPTS_AS_CONTEXT (F1=0.140):
```
1861-1865

Explain the role of women in the Civil War.

Report: 1861-1865

Explain the role of the press in the Civil War.

Report: 1861-1865

Explain the role of the telegraph in the Civil War.

Report: 1861-1865

Explain the role of railroads in the Civil War.

Report: 1861-1865

Explain the role of the Union blockade in the Civil War.

Report: 1861-1865

Explain the role of the Anaconda Plan in the Civil War.

Report: 1861-1
```

### ENGRAM_8_PR (F1=0.344):
```
100 years ago, the U.S. Army was segregated. African American soldiers were assigned to all-black units, and they were often treated as second-class citizens. But during World War I, African American troops played a crucial role in the war effort. They served in the 92nd and 93rd divisions, which were known as the “Harlem Hellfighters” and the “Buffalo Soldiers.” These units were some of the most decorated in the war, and they helped to break down racial barriers in the military.

## What was the role of African American troops during World War I?

African American troops played a vital role in World War I. They served in the 92nd and 93rd divisions, which were known as the “Harlem Hellfighters” and the “
```

### RANDOM_ENGRAMS (F1=0.000):
```
1.5 pages, 12 pt. font, double spaced, 1 inch margins.

### "Get 15% discount on your first 3 orders with us" Use the following coupon "FIRST15"

Order Now
```

---

## Probe 5 (test_idx=12)

**PROBE:** Explain how each side mobilized manpower and overcame recruitment problems.

### CEILING (full_context, F1=1.000):
```
100%

Q: Tell me about the Union's recruitment problems.
A: The Union had difficulty recruiting men for the army because of the high casualty rates and the long duration of the war. The Union also had difficulty recruiting men because of the draft, which was unpopular with many people.

Q: Tell me about the Confederacy's recruitment problems.
A: The Confederacy had difficulty recruiting men for the army because of the high casualty rates and the long duration of the war. The Confederacy also had difficulty recruiting men because of the draft, which was unpopular with many people.

Q: Tell me about the Union's manpower mobilization.
A: The Union mobilized manpower by drafting men into the army and by recruiting volunteers. The Union also mobilized
```

### RECENT_ONLY (F1=0.113):
```
1861-1865

The American Civil War was a conflict between the United States of America and the Confederate States of America. The war began in 1861 and ended in 1865. The war was fought over the issue of slavery. The North wanted to end slavery, while the South wanted to keep it. The war was fought in the states of the South. The North won the war.

The North had a population of about 22 million people. The South had a population of about 9 million people. The North had a larger population than the South. The North also had a larger economy than the South. The North had a larger army than the South. The North had a larger navy than the South. The North had a larger air force than the South. The North had a larger army than the South
```

### PROMPTS_AS_CONTEXT (F1=0.061):
```
1861-1865

Explain the role of the press in the Civil War.

Report: 1861-1865

Explain the role of the telegraph in the Civil War.

Report: 1861-1865

Explain the role of railroads in the Civil War.

Report: 1861-1865

Explain the role of the telegraph in the Civil War.

Report: 1861-1865

Explain the role of railroads in the Civil War.

Report: 1861-1865

Explain the role of the telegraph in the Civil War.

Report: 1861-18
```

### ENGRAM_8_PR (F1=0.250):
```
1. Explain how each side mobilized manpower and overcame recruitment problems. 2. Explain how each side mobilized manpower and overcame recruitment problems. 3. Explain how each side mobilized manpower and overcame recruitment problems. 4. Explain how each side mobilized manpower and overcame recruitment problems. 5. Explain how each side mobilized manpower and overcame recruitment problems. 6. Explain how each side mobilized manpower and overcame recruitment problems. 7. Explain how each side mobilized manpower and overcame recruitment problems. 8. Explain how each side mobilized manpower and overcame recruitment problems. 9. Explain how each side mobilized manpower and overcame recruitment problems.
```

### RANDOM_ENGRAMS (F1=0.000):
```
100% of 100% of Western Europeans are not satisfied with their sex life.

## 100% of 100% of Western Europeans are not satisfied with their sex life.

Report: 100% of 100% of Western Europeans are not satisfied with their sex life.

## 100% of 100% of Western Europeans are not satisfied with their sex life.

Report: 100% of 100% of Western Europeans are not satisfied with their sex life.

## 100% of 100% of Western Europeans are not satisfied with their sex life.

Report: 100% of 100% of Western Europeans are not satisfied with their sex life
```

---
