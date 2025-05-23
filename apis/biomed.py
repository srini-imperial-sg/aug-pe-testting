import random



def get_prompt(prompt, ds_name):
    if ds_name == 'psytar':
        return get_psytar_prompt(prompt)
    elif ds_name == 'hallmarks_of_cancer':
        return get_hoc_prompt(prompt)
    elif ds_name == 'mimic':
        return get_mimic_prompt(prompt)
    elif ds_name == 'n2c2_2008':
        return get_n2c2_2008_prompt(prompt)
    elif ds_name == 'danielml':
        return get_danielml_prompt_from_sentiment_only(prompt)
    elif ds_name == 'luckycat37':
        return get_luckycat37_prompt_from_sentiment_only(prompt)
    raise NotImplementedError()

def get_psytar_prompt(prompt):
    labels = prompt.split("|")
    styles = """Chronic patient
New parent
Senior reviewer
Allergy sufferer
Prescription user
Healthcare professional
Migraine patient
Fitness enthusiast
Mental health patient
Insomnia sufferer""".splitlines()

    label_map = {
        "ADR": "Adverse Drug Reaction",
        "DI": "Drug Indications",
        "EF": "Drug Effectiveness",
        "INF": "Drug Ineffeciveness",
        "Others": "Others",
        "SSI": "Sign/Symptoms/Illness",
        "WD": "Withdrowal Symptoms"
    }
    style = random.choice(styles)
    return f"Suppose you're a {style}. Write a one-sentence medication review that mentions the following adverse drug reactions: {', '.join(label_map.get(l, 'No reactions.') for l in labels)}"


def get_danielml_prompt_from_sentiment_only(example):
    sentiment = example

    personas = [
        "a financial journalist at a business news outlet",
        "a market analyst covering emerging economies",
        "a macroeconomics researcher focusing on Southeast Asia",
        "a policy advisor specializing in financial inclusion",
        "a senior executive at a regional microfinance institution",
        "a portfolio manager assessing risk in underserved markets",
        "a development economist studying urban poverty",
        "a fintech strategist working on digital banking for the unbanked",
        "a regulatory affairs officer at a central bank",
        "an impact investor evaluating social return on investment"
    ]

    persona = random.choice(personas)

    sentiment_prompts = {
        "positive": f"As {persona}, write a financial summary that highlights growth, opportunity, or favorable outcomes.",
        "negative": f"As {persona}, write a financial commentary that focuses on risks, setbacks, or structural challenges.",
        "neutral":  f"As {persona}, write a factual and objective financial statement with a professional tone."
    }

    return sentiment_prompts.get(sentiment, "As a financial expert, provide a contextual analysis.")



def get_luckycat37_prompt_from_sentiment_only(sentiment):
    publishers = ["Bloomberg", "Reuters", "CNBC", "Benzinga", "TechCrunch"]
    topics = {
        "positive": [
            "Company X reports record profits in Q2 earnings",
            "Markets rally as inflation slows and hiring surges",
            "Tech stocks lead gains amid upbeat consumer demand",
            "Breakthrough in AI technology boosts investor confidence",
            "New product launch sends stock prices soaring"
        ],
        "negative": [
            "Markets fall amid recession fears and weak earnings",
            "Major tech firm hit by massive data breach",
            "Oil prices slump for fourth straight day",
            "Layoffs surge as economic uncertainty intensifies",
            "Regulators investigate antitrust violations in merger deal"
        ],
        "neutral": [
            "Analysts expect steady growth in telecom sector",
            "Federal Reserve holds interest rates steady",
            "Survey shows mixed consumer sentiment",
            "New study explores trends in remote work adoption",
            "Quarterly forecast outlines stable economic outlook"
        ]
    }

    # Randomly choose a sentiment label
    title = random.choice(topics[sentiment])

    # Generate article body with slight variation
    body_templates = {
        "positive": f"As a reporter for {random.choice(publishers)}, write an article about -- {title}. Analysts note optimism among investors as indicators remain strong. Tech and energy sectors are driving the rally.",
        "negative": f"As a reporter for {random.choice(publishers)}, write an article about -- {title}. Experts warn of potential long-term impact. Financials and industrials led the losses in the market today.",
        "neutral": f"As a reporter for {random.choice(publishers)}, write an article about -- {title}. The report provides insight into economic dynamics but does not indicate a clear market direction."
    }

    prompt = body_templates.get(sentiment, "As a financial expert, provide a contextual analysis.")
    
    return prompt




def get_danielml_diverse_prompt(sentiment: str, fewshot: bool = True) -> str:
    """
    Generate a diverse, multi-template prompt for a given sentiment.
    - Randomly picks a persona.
    - Randomly selects one of several base templates.
    - Randomly injects 1 extra feature instruction.
    - Optionally prepends 1 few-shot example.
    """
    # 1. Persona list
    PERSONAS = [
        "a financial journalist at a business news outlet",
        "a market analyst covering emerging economies",
        "a macroeconomics researcher focusing on Southeast Asia",
        "a policy advisor specializing in financial inclusion",
        "a senior executive at a regional microfinance institution",
        "a portfolio manager assessing risk in underserved markets",
        "a development economist studying urban poverty",
        "a fintech strategist working on digital banking for the unbanked",
        "a regulatory affairs officer at a central bank",
        "an impact investor evaluating social return on investment"
    ]

    # 2. Prompt templates for each sentiment
    BASE_TEMPLATES = {
        "positive": [
            "As {persona}, write a financial summary that highlights growth, opportunity, or favorable outcomes.",
            "You are {persona}. Draft a positive financial outlook emphasizing new gains and strong indicators.",
            "Acting as {persona}, compose a news-style report focusing on upward trends and successes."
        ],
        "negative": [
            "As {persona}, write a financial commentary that focuses on risks, setbacks, or structural challenges.",
            "You are {persona}. Draft a cautionary financial analysis highlighting downturns and vulnerabilities.",
            "Acting as {persona}, compose a critical report underlining potential obstacles and losses."
        ],
        "neutral": [
            "As {persona}, write a factual and objective financial statement with a professional tone.",
            "You are {persona}. Draft a neutral summary presenting facts without bias or emotional language.",
            "Acting as {persona}, compose an impartial report summarizing key figures and events."
        ]
    }

    # 3. Extra random features to inject
    FEATURE_INJECTIONS = [
        "Also include a rhetorical question about future outlook.",
        "Use a bullet-point list for key highlights.",
        "Begin with a headline-style title.",
        "Include a brief statistic to illustrate your point.",
        "End with a concise call to action for investors."
    ]

    # 4. Optional few-shot seeds per sentiment
    FEWSHOT_EXAMPLES = {
        "positive": [
            "Example: 'Company XYZ saw revenues jump 20% amid strong consumer demand.'",
            "Example: 'Asset prices rallied, underpinned by robust economic indicators.'"
        ],
        "negative": [
            "Example: 'Supply chain disruptions pushed costs up, squeezing profit margins.'",
            "Example: 'Uncertainty looms as key metrics fall short of analyst estimates.'"
        ],
        "neutral": [
            "Example: 'Q2 earnings were flat compared to Q1, with revenues at $1.2B.'",
            "Example: 'The central bank held rates steady, citing stable inflation.'"
        ]
    }


    persona = random.choice(PERSONAS)
    
    # 1. Pick base template
    base_tmpl = random.choice(BASE_TEMPLATES[sentiment])
    prompt = base_tmpl.format(persona=persona)
    
    # 2. Inject a random feature
    inject = random.choice(FEATURE_INJECTIONS)
    prompt = f"{prompt} {inject}"
    
    # 3. Optionally prepend a few-shot example
    if fewshot:
        seed = random.choice(FEWSHOT_EXAMPLES[sentiment])
        prompt = f"{seed}\n\n{prompt}"
    
    return prompt




def get_hoc_prompt(prompt):
    labels = prompt.split("|")
    styles = """Cell Biologist
Immunologist
Molecular Geneticist
Metabolic Scientist
Vascular Biologist
Evolutionary Biologist
Systems Biologist
Epigeneticist
Tissue Engineer
Biochemist""".splitlines()

    label_map = [
        "activating invasion and metastasis",
        "avoiding immune destruction",
        "cellular energetics",
        "enabling replicative immortality",
        "evading growth suppressors",
        "genomic instability and mutation",
        "inducing angiogenesis",
        "resisting cell death",
        "sustaining proliferative signaling",
        "tumor promoting inflammation"
    ]
    label_map = {l:l for l in label_map}
    style = random.choice(styles)
    return f"Suppose you're a {style} writing a scientific paper about hallmarks of cancer. Write one sentence from your papers' abstract, that mentions the following hallmark(s): {', '.join(label_map.get(l, 'No hallmark.') for l in labels)}"


def get_mimic_prompt(prompt):
    labels = prompt.split("|")
    styles = """Critical Care Physician
Respiratory Therapist
Clinical Pharmacist
Infectious Disease Specialist
ICU Nurse
Nephrologist
Clinical Dietitian
Physical Therapist
Social Worker
Palliative Care Specialist""".splitlines()

    label_map = [ 
            ((1, 139), "Infectious And Parasitic Diseases"),
            ((140, 239), "Neoplasms"),
            ((240, 279), "Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders"),
            ((280, 289), "Diseases Of The Blood And Blood-Forming Organs"),
            ((290, 319), "Mental Disorders"),
            ((320, 389), "Diseases Of The Nervous System And Sense Organs"),
            ((390, 459), "Diseases Of The Circulatory System"),
            ((460, 519), "Diseases Of The Respiratory System"),
            ((520, 579), "Diseases Of The Digestive System"),
            ((580, 629), "Diseases Of The Genitourinary System"),
            ((630, 679), "Complications Of Pregnancy, Childbirth, And The Puerperium"),
            ((680, 709), "Diseases Of The Skin And Subcutaneous Tissue"),
            ((710, 739), "Diseases Of The Musculoskeletal System And Connective Tissue"),
            ((740, 759), "Congenital Anomalies"),
            ((760, 779), "Certain Conditions Originating In The Perinatal Period"),
            ((780, 799), "Symptoms, Signs, And Ill-Defined Conditions"),
            ((800, 999), "Injury And Poisoning")
        ]
    label_map = {l:l for _, l in label_map}
    style = random.choice(styles)
    return f"Suppose you're a {style} writing a brief hospital history section as part of a discharge summary of a patient admitted to ICU. Mentions the following disesases: {', '.join(label_map.get(l, 'None.') for l in labels)}"


ALL_MIMIC_TONES = [
    "Chronological Narrative: Sequential storytelling with full sentences",
"Problem-Based: Listed by medical issues",
"Bullet-Point Style: Short points with dates",
"System-Based: Organized by organ systems",
"Heavy Abbreviation Style: Maximum use of medical acronyms",
"Intervention-Focused: Listed by medical procedures",
"Milestone-Based: Major events and timepoints",
"Table Format: Data arranged in grid",
"Assessment/Plan Style: Summary with ongoing tasks",
"Outcome-Oriented: Problems with their resolutions"
]


def get_n2c2_2008_prompt(prompt):
    labels = prompt.split("|")
    styles = """Hospitalist
Internist
General Surgeon
Cardiologist
Neurologist
Pulmonologist
Orthopedic Surgeon
Psychiatrist
Intensivist
Oncologist""".splitlines()

    label_map = [
            'Asthma',
            'CAD',
            'CHF',
            'Depression',
            'Diabetes',
            'GERD',
            'Gallstones',
            'Gout',
            'Hypercholesterolemia',
            'Hypertension',
            'Hypertriglyceridemia',
            'OA',
            'OSA',
            'Obesity',
            'PVD',
            'Venous Insufficiency'
    ]
    label_map = {l:l for l in label_map}
    style = random.choice(styles)
    return f"Suppose you're a {style} writing a patient discharge summary of a patient. Mention the following conditions and co-morbidities: {', '.join(label_map.get(l, 'None.') for l in labels)}"