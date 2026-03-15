"""MPAR Prompt Pairs v2 — redesigned for better retrieval signal.

Design principles:
- Storage prompts are uniformly 12-25 words with rich domain context
- Retrieval prompts carry enough semantic signal to disambiguate (no bare "what was the reading?")
- No ultra-short prompts that collapse to generic mean-pooled attractors
- Each pair has unique domain vocabulary that appears on both sides
"""

PROMPT_PAIRS_V2 = [
    # Financial / shopping
    (
        "I spent $4.96 on groceries at the Safeway supermarket on the afternoon of August 2 2025.",
        "I need to check how much money I paid for the grocery shopping trip at Safeway in August.",
        "$4.96",
    ),
    (
        "The dinner bill at Osteria Francescana restaurant came to $78.50 after tax and tip were added.",
        "Can you remind me of the total restaurant bill from the Osteria Francescana dinner including tip?",
        "$78.50",
    ),
    (
        "The monthly rent for the apartment on Maple Street is $2,350 and it is due on the first of each month.",
        "I need to know the monthly apartment rent amount that I owe for the Maple Street place.",
        "$2,350",
    ),
    (
        "The mechanic at Henderson Auto Shop quoted $1,247.89 for the transmission repair on the Honda Civic.",
        "How much did the Henderson Auto Shop estimate for fixing the Honda Civic transmission?",
        "$1,247.89",
    ),
    (
        "I moved $500 from checking into the high-yield savings account at Ally Bank on Monday morning.",
        "What was the amount of the transfer I made to the Ally Bank savings account on Monday?",
        "$500",
    ),
    # Medical
    (
        "At the morning checkup on Tuesday, the nurse recorded the patient's blood pressure as 142 over 91.",
        "I need to look up the blood pressure numbers from the Tuesday morning nursing assessment.",
        "142/91",
    ),
    (
        "During the cardiology appointment, her resting heart rate was measured at 72 beats per minute on the ECG monitor.",
        "What heart rate did the cardiologist record on the ECG during the resting measurement?",
        "72",
    ),
    (
        "The blood panel from Quest Diagnostics laboratory showed the total cholesterol level was 218 mg/dL.",
        "I need to find the cholesterol number from the Quest Diagnostics blood panel lab results.",
        "218",
    ),
    (
        "After the overnight fasting period, the glucose meter showed a blood sugar reading of 103 mg/dL.",
        "What was the fasting blood sugar number that showed up on the glucose meter test?",
        "103",
    ),
    (
        "Dr. Nakamura prescribed 20mg of lisinopril to be taken orally once daily for hypertension management.",
        "What dosage of lisinopril did Dr. Nakamura prescribe for the hypertension treatment plan?",
        "20mg",
    ),
    # Travel / codes
    (
        "The United Airlines flight to Chicago has confirmation code XK447B and departs from gate B22.",
        "I need to pull up the booking confirmation code for the United Airlines Chicago flight.",
        "XK447B",
    ),
    (
        "The Hilton Garden Inn reservation for the Portland trip is booked under reference number HR-90215.",
        "What is the Hilton Garden Inn reservation reference number for the Portland hotel stay?",
        "HR-90215",
    ),
    (
        "The Hertz rental car counter is located at terminal 3 gate C in the arrivals area of the airport.",
        "Where exactly in the airport terminal do I go to pick up the Hertz rental car?",
        "terminal 3 gate C",
    ),
    (
        "The Amtrak Northeast Regional train to Boston departs from platform 7 at Penn Station.",
        "Which Penn Station platform should I go to for the Amtrak Northeast Regional Boston train?",
        "platform 7",
    ),
    (
        "The lockbox combination for the Airbnb cottage on Lakeview Drive is the four digits 4829.",
        "What is the four-digit lockbox entry code for the Lakeview Drive Airbnb cottage?",
        "4829",
    ),
    # Locations / offices
    (
        "Sarah Connors has her private office on the third floor of the McKinley building in room 312.",
        "Which room and floor is Sarah Connors' office in at the McKinley building?",
        "room 312 third floor",
    ),
    (
        "The color laser printer for the marketing department is in the copy room on the second floor near the elevator.",
        "Where is the marketing department's color laser printer located in the building?",
        "copy room second floor",
    ),
    (
        "The main entrance to the underground parking garage is accessible from the Oak Street side of the building.",
        "Which street has the entrance I should use to get into the underground parking garage?",
        "Oak Street",
    ),
    (
        "The corporate IT help desk and technical support office is located in building B room 105 on the ground floor.",
        "Where do I find the corporate IT help desk if I need in-person technical support?",
        "building B room 105",
    ),
    (
        "In case of fire, the nearest emergency exit is through the stairwell door at the east end of the third-floor hallway.",
        "Which stairwell and hallway direction leads to the closest fire emergency exit on the third floor?",
        "east end stairwell",
    ),
    # Dates / times
    (
        "The quarterly budget review meeting with the finance team starts at 2:15pm on Thursday March 19 in conference room A.",
        "When exactly does the quarterly budget review meeting with the finance team begin on the March Thursday?",
        "2:15pm Thursday March 19",
    ),
    (
        "The final submission deadline for the Horizon research project deliverables is April 30 2025.",
        "By what date do all the Horizon research project deliverables need to be submitted?",
        "April 30 2025",
    ),
    (
        "The dental cleaning and checkup appointment at Dr. Yamamoto's office is scheduled for Tuesday at 10:30am.",
        "When is the dental cleaning appointment at Dr. Yamamoto's office scheduled for this week?",
        "Tuesday 10:30am",
    ),
    (
        "The Radiohead concert at Madison Square Garden is on Saturday June 7 and doors open at 8pm.",
        "What date and time is the Radiohead show at Madison Square Garden happening?",
        "Saturday June 7 8pm",
    ),
    (
        "The residential lease agreement for the Maple Street apartment expires on September 1 2026.",
        "When is the expiration date on the residential lease for the Maple Street apartment?",
        "September 1 2026",
    ),
    # People / contacts
    (
        "The senior project manager overseeing the Atlas migration is David Chen and his direct extension is 4471.",
        "Who is the project manager running the Atlas migration and what is their phone extension?",
        "David Chen ext 4471",
    ),
    (
        "The property landlord for the Oakwood Terrace apartment complex is Margaret Thompson who handles all lease issues.",
        "Who is the landlord responsible for lease matters at the Oakwood Terrace apartment complex?",
        "Margaret Thompson",
    ),
    (
        "In a medical emergency, the designated on-call contact is Dr. Patel who can be reached at 555-0193.",
        "Who is the on-call doctor I should contact in a medical emergency and what is their phone number?",
        "Dr. Patel 555-0193",
    ),
    (
        "The evening babysitter for the kids is Jessica Rodriguez and her cell phone number is 555-8824.",
        "What is the name and phone number of the babysitter who watches the kids in the evening?",
        "Jessica Rodriguez 555-8824",
    ),
    (
        "The CPA handling the annual tax filing and business audit is Robert Liu at Deloitte.",
        "Who is the accountant from Deloitte responsible for the annual tax filing and business audit?",
        "Robert Liu",
    ),
    # Technical / passwords / settings
    (
        "The guest WiFi network at the downtown coworking space uses the password BlueSky2024 with no spaces.",
        "What password do I type to connect to the guest WiFi at the downtown coworking space?",
        "BlueSky2024",
    ),
    (
        "The production PostgreSQL database server is hosted at the static IP address 192.168.1.42 on the internal network.",
        "What is the IP address of the production PostgreSQL database server on the internal network?",
        "192.168.1.42",
    ),
    (
        "The REST API gateway enforces a rate limit of 1000 requests per minute for authenticated clients.",
        "How many requests per minute does the REST API gateway allow for authenticated clients?",
        "1000 per minute",
    ),
    (
        "To connect via SSH to the staging environment server, use port 2222 instead of the default port 22.",
        "What non-standard port number is configured for SSH access to the staging environment server?",
        "2222",
    ),
    (
        "The WordPress admin panel credentials are username admin and temporary password TempPass99 until the reset.",
        "What are the current login credentials for accessing the WordPress admin panel?",
        "admin TempPass99",
    ),
    # Measurements / quantities
    (
        "The living room in the new house on Elm Street measures 14 feet wide by 18 feet long according to the floor plan.",
        "What are the width and length dimensions of the living room in the Elm Street house floor plan?",
        "14 by 18 feet",
    ),
    (
        "The FedEx shipping package containing the prototype electronics weighs exactly 3.7 kilograms on the scale.",
        "How much does the FedEx package with the prototype electronics weigh according to the scale?",
        "3.7 kilograms",
    ),
    (
        "The raised vegetable garden plot in the backyard measures 6 meters long by 4 meters wide.",
        "What are the length and width measurements of the raised vegetable garden plot in the backyard?",
        "6 by 4 meters",
    ),
    (
        "The ceiling height in the converted loft bedroom upstairs is 9 feet 6 inches at the peak.",
        "How much vertical clearance is there at the peak of the converted loft bedroom ceiling upstairs?",
        "9 feet 6 inches",
    ),
    (
        "The industrial water storage tank in the basement utility room has a total capacity of 55 gallons.",
        "What is the maximum volume capacity of the industrial water storage tank in the basement utility room?",
        "55 gallons",
    ),
    # Recipes / cooking
    (
        "Grandma's chocolate layer cake recipe requires exactly 2 and a quarter cups of all-purpose flour sifted twice.",
        "How many cups of flour does grandma's chocolate layer cake recipe call for after sifting?",
        "2.25 cups",
    ),
    (
        "The chicken and broccoli casserole should bake in the oven at 375 degrees Fahrenheit for 45 minutes until golden.",
        "What oven temperature and baking time does the chicken and broccoli casserole need to cook until golden?",
        "375 degrees 45 minutes",
    ),
    (
        "The teriyaki chicken marinade recipe calls for 3 tablespoons of low-sodium soy sauce mixed with ginger.",
        "How much soy sauce goes into the teriyaki chicken marinade along with the ginger?",
        "3 tablespoons",
    ),
    (
        "After kneading the sourdough bread dough, let it proof and rise for 90 minutes in a warm draft-free spot.",
        "How long should the sourdough bread dough proof and rise after kneading before shaping?",
        "90 minutes",
    ),
    (
        "The minestrone soup recipe from the Italian cookbook yields enough food for approximately 8 adult servings.",
        "How many adult-sized servings does the Italian cookbook minestrone soup recipe produce?",
        "8",
    ),
    # Scores / statistics
    (
        "The Monday night football game ended with a final score of 24 to 17 in favor of the home team Eagles.",
        "What was the final score of the Monday night football game between the Eagles home team and visitors?",
        "24 to 17",
    ),
    (
        "After completing all coursework this semester, the undergraduate student's cumulative GPA stands at 3.74.",
        "What cumulative GPA did the undergraduate student earn after finishing this semester's coursework?",
        "3.74",
    ),
    (
        "The employee satisfaction survey distributed company-wide received a response rate of 43 percent overall.",
        "What percentage of employees responded to the company-wide satisfaction survey that was distributed?",
        "43 percent",
    ),
    (
        "According to the latest status report, the Phoenix infrastructure project has reached 87 percent completion.",
        "How far along toward completion is the Phoenix infrastructure project based on the latest status report?",
        "87 percent",
    ),
    (
        "The starting shortstop finished the regular baseball season with a batting average of .312 over 140 games.",
        "What batting average did the starting shortstop post across the full regular baseball season?",
        ".312",
    ),
]
