"""100 distinct U.S. Civil War subtopics, organized by category.

Each topic is a short title that will be expanded by Mistral-7B into a
prompt-response pair. Categorization is used for ground-truth relevance
annotation (a probe like "role of railroads" maps to all logistics/
technology/movement turns).
"""

# 25 specific battles
BATTLES = [
    "First Battle of Bull Run (Manassas)",
    "Battle of Antietam",
    "Battle of Gettysburg",
    "Battle of Chancellorsville",
    "Siege of Vicksburg",
    "Battle of Chattanooga",
    "Battle of Atlanta",
    "Sherman's March to the Sea",
    "Battle of the Wilderness",
    "Battle of Spotsylvania Court House",
    "Battle of Cold Harbor",
    "Siege of Petersburg",
    "Battle of Appomattox Court House",
    "Battle of Shiloh",
    "Battle of Stones River (Murfreesboro)",
    "Battle of Chickamauga",
    "Battle of Fredericksburg",
    "Second Battle of Bull Run",
    "Seven Days Battles",
    "Battle of Hampton Roads (Monitor vs. Virginia)",
    "Capture of New Orleans",
    "Attack on Fort Sumter",
    "Battle of Fort Donelson",
    "Battle of Mobile Bay",
    "Battle of Pea Ridge",
]

# 20 generals and commanders
GENERALS = [
    "Robert E. Lee",
    "Ulysses S. Grant",
    "William Tecumseh Sherman",
    "Stonewall Jackson",
    "James Longstreet",
    "Joseph E. Johnston",
    "George B. McClellan",
    "George Meade",
    "Philip Sheridan",
    "J.E.B. Stuart",
    "Nathan Bedford Forrest",
    "Braxton Bragg",
    "Albert Sidney Johnston",
    "P.G.T. Beauregard",
    "John Bell Hood",
    "Ambrose Burnside",
    "Joseph Hooker",
    "Winfield Scott",
    "George Pickett",
    "George H. Thomas",
]

# 15 political and leadership figures + decisions
POLITICAL = [
    "Abraham Lincoln's wartime leadership",
    "Jefferson Davis as Confederate president",
    "Edwin Stanton as Secretary of War",
    "William Seward as Secretary of State",
    "the Emancipation Proclamation",
    "the 13th Amendment",
    "Lincoln's suspension of habeas corpus",
    "the Confederate Conscription Act of 1862",
    "the Border States and their alignment",
    "Confederate diplomacy with Britain and France",
    "the U.S. Election of 1864",
    "the Trent Affair",
    "Salmon P. Chase and Union war finance",
    "Andrew Johnson before the assassination",
    "Alexander Stephens as Confederate Vice President",
]

# 15 logistics, technology, and military innovations
LOGISTICS = [
    "the role of railroads in the Civil War",
    "the role of the telegraph",
    "ironclad warships",
    "repeating rifles and the Spencer carbine",
    "the H.L. Hunley submarine",
    "Mathew Brady and Civil War photography",
    "Thaddeus Lowe and military balloons",
    "the Anaconda Plan",
    "the Union naval blockade",
    "cavalry doctrine and tactics",
    "artillery developments and Napoleon guns",
    "Civil War military medicine",
    "Union vs. Confederate logistics capacity",
    "early trench warfare at Petersburg",
    "Union quartermaster operations",
]

# 15 social impact, civilian, and aftermath
SOCIAL = [
    "slavery and abolition during the war",
    "United States Colored Troops (USCT)",
    "women's contributions to the war effort",
    "wartime refugees in the South",
    "civilian casualties and impact",
    "the New York City Draft Riots of 1863",
    "Andersonville Confederate prison",
    "Union prisoner-of-war camps",
    "disease as the leading cause of military death",
    "wartime economic effects on the North",
    "wartime economic effects on the South",
    "early Reconstruction policy",
    "Civil War veterans and pensions",
    "religion among soldiers",
    "guerilla warfare in border regions",
]

# 10 specific named campaigns
CAMPAIGNS = [
    "the Peninsula Campaign of 1862",
    "the Maryland Campaign of 1862",
    "the Vicksburg Campaign",
    "the Atlanta Campaign of 1864",
    "the Overland Campaign of 1864",
    "Sherman's Carolinas Campaign of 1865",
    "the Red River Campaign of 1864",
    "the Tullahoma Campaign",
    "the Knoxville Campaign",
    "the Mobile Campaign of 1865",
]

ALL_TOPICS = (
    [(t, "battle")    for t in BATTLES] +
    [(t, "general")   for t in GENERALS] +
    [(t, "political") for t in POLITICAL] +
    [(t, "logistics") for t in LOGISTICS] +
    [(t, "social")    for t in SOCIAL] +
    [(t, "campaign")  for t in CAMPAIGNS]
)
assert len(ALL_TOPICS) == 100, f"got {len(ALL_TOPICS)} topics, expected 100"
