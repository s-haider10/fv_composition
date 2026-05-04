"""
Task definitions with DISJOINT extraction and evaluation pools.

Design principles (Neel-style):
  - Each atom has 30+ pairs, split into:
    - extraction_pool (first 20): used for FV extraction (sample N_ICL ICL examples
      from this pool, with multiple permutations)
    - eval_pool (last 10+): used for evaluation, NEVER seen during FV extraction
  - All answers are single-token-friendly (verified via audit_tokenization)
  - Numbers chosen to avoid tokenization quirks (e.g., 'Vienna' truncation)
"""

# ============= ATOMIC TASKS =============
# Each task has >= 30 pairs. First 20 = extraction. Last 10+ = eval.

TASKS = {
    # --- STRING / RETRIEVAL ATOMS ---
    "country_capital": [
        # Extraction pool (20)
        ("Japan","Tokyo"),("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
        ("Spain","Madrid"),("Russia","Moscow"),("Egypt","Cairo"),("Greece","Athens"),
        ("Poland","Warsaw"),("Sweden","Stockholm"),("Norway","Oslo"),("Finland","Helsinki"),
        ("Portugal","Lisbon"),("Hungary","Budapest"),("Ireland","Dublin"),("Belgium","Brussels"),
        ("Denmark","Copenhagen"),("Thailand","Bangkok"),("Vietnam","Hanoi"),("Turkey","Ankara"),
        # Eval pool (10) -- held out
        ("Iran","Tehran"),("Iraq","Baghdad"),("Cuba","Havana"),("Peru","Lima"),
        ("Chile","Santiago"),("Kenya","Nairobi"),("Morocco","Rabat"),("Canada","Ottawa"),
        ("Argentina","Buenos"),("Australia","Canberra"),
    ],
    "country_currency": [
        # Extraction
        ("Japan","yen"),("USA","dollar"),("UK","pound"),("Germany","euro"),
        ("Russia","ruble"),("India","rupee"),("China","yuan"),("Mexico","peso"),
        ("Switzerland","franc"),("Sweden","krona"),("Poland","zloty"),("Brazil","real"),
        ("Turkey","lira"),("Israel","shekel"),("Vietnam","dong"),("Thailand","baht"),
        ("Korea","won"),("Denmark","krone"),("Hungary","forint"),("France","euro"),
        # Eval
        ("Iran","rial"),("Iraq","dinar"),("Cuba","peso"),("Peru","sol"),
        ("Chile","peso"),("Kenya","shilling"),("Morocco","dirham"),("Canada","dollar"),
        ("Argentina","peso"),("Australia","dollar"),
    ],
    "uppercase": [
        # Extraction
        ("japan","JAPAN"),("hello","HELLO"),("paris","PARIS"),("apple","APPLE"),
        ("table","TABLE"),("water","WATER"),("music","MUSIC"),("paper","PAPER"),
        ("light","LIGHT"),("stone","STONE"),("river","RIVER"),("happy","HAPPY"),
        ("dance","DANCE"),("smile","SMILE"),("plant","PLANT"),("ocean","OCEAN"),
        ("cloud","CLOUD"),("dream","DREAM"),("brave","BRAVE"),("quiet","QUIET"),
        # Eval
        ("zebra","ZEBRA"),("flame","FLAME"),("globe","GLOBE"),("honey","HONEY"),
        ("ivory","IVORY"),("juice","JUICE"),("knife","KNIFE"),("lemon","LEMON"),
        ("magic","MAGIC"),("noble","NOBLE"),
    ],
    "first_letter": [
        # Extraction
        ("japan","j"),("hello","h"),("paris","p"),("apple","a"),("table","t"),
        ("water","w"),("music","m"),("paper","p"),("light","l"),("stone","s"),
        ("river","r"),("happy","h"),("dance","d"),("smile","s"),("plant","p"),
        ("ocean","o"),("cloud","c"),("dream","d"),("brave","b"),("quiet","q"),
        # Eval
        ("zebra","z"),("flame","f"),("globe","g"),("honey","h"),("ivory","i"),
        ("juice","j"),("knife","k"),("lemon","l"),("magic","m"),("noble","n"),
    ],

    # --- ARITHMETIC ATOMS ---
    "successor": [
        # Extraction (20) -- numbers 3-50
        ("3","4"),("7","8"),("12","13"),("25","26"),("41","42"),
        ("9","10"),("17","18"),("33","34"),("4","5"),("11","12"),
        ("19","20"),("28","29"),("36","37"),("44","45"),("6","7"),
        ("14","15"),("21","22"),("38","39"),("47","48"),("5","6"),
        # Eval (10) -- disjoint range 50-99
        ("50","51"),("57","58"),("63","64"),("72","73"),("88","89"),
        ("66","67"),("75","76"),("82","83"),("91","92"),("99","100"),
    ],
    "double": [
        # Extraction
        ("3","6"),("7","14"),("12","24"),("25","50"),("4","8"),
        ("9","18"),("11","22"),("15","30"),("20","40"),("6","12"),
        ("8","16"),("13","26"),("17","34"),("21","42"),("5","10"),
        ("14","28"),("16","32"),("18","36"),("22","44"),("19","38"),
        # Eval (using same number range as successor eval for fair comparison)
        ("50","100"),("57","114"),("63","126"),("72","144"),("88","176"),
        ("66","132"),("75","150"),("82","164"),("91","182"),("99","198"),
    ],
    "triple": [
        # Extraction
        (str(x), str(3*x)) for x in
        [3,7,12,25,4,9,11,15,20,6,8,13,17,21,5,14,16,18,22,19]
    ] + [
        # Eval
        (str(x), str(3*x)) for x in [50,57,63,72,88,66,75,82,91,99]
    ],
}


# ============= COMPOSITIONS =============
# Each composition derives from atom pairs. Same disjoint extraction/eval split.

def make_composition(name, f, g, atom_tasks, fn):
    """fn(query) -> expected composed answer string."""
    pairs = []
    # Use the union of extraction pools from both atoms? No -- pick atom whose
    # input space matches. For our cases, g_op's input space is the right one.
    source = atom_tasks[g]
    pairs = [(x, fn(x)) for x, _ in source]
    return {"f": f, "g": g, "pairs": pairs}


def build_compositions():
    capital = {x: y for x, y in TASKS["country_capital"]}
    currency = {x: y for x, y in TASKS["country_currency"]}
    return {
        "uppercase_of_capital": {
            "f": "uppercase", "g": "country_capital",
            "pairs": [(c, cap.upper()) for c, cap in TASKS["country_capital"]],
        },
        "first_letter_of_capital": {
            "f": "first_letter", "g": "country_capital",
            "pairs": [(c, cap[0].lower()) for c, cap in TASKS["country_capital"]],
        },
        "uppercase_of_currency": {
            "f": "uppercase", "g": "country_currency",
            "pairs": [(c, cur.upper()) for c, cur in TASKS["country_currency"]],
        },
        "first_letter_of_currency": {
            "f": "first_letter", "g": "country_currency",
            "pairs": [(c, cur[0].lower()) for c, cur in TASKS["country_currency"]],
        },
        "successor_of_double": {
            "f": "successor", "g": "double",
            "pairs": [(x, str(2*int(x)+1)) for x, _ in TASKS["double"]],
        },
        "double_of_successor": {
            "f": "double", "g": "successor",
            "pairs": [(x, str(2*(int(x)+1))) for x, _ in TASKS["successor"]],
        },
        "successor_of_triple": {
            "f": "successor", "g": "triple",
            "pairs": [(x, str(3*int(x)+1)) for x, _ in TASKS["triple"]],
        },
        "triple_of_successor": {
            "f": "triple", "g": "successor",
            "pairs": [(x, str(3*(int(x)+1))) for x, _ in TASKS["successor"]],
        },
        "successor_of_successor": {
            "f": "successor", "g": "successor",
            "pairs": [(x, str(int(x)+2)) for x, _ in TASKS["successor"]],
        },
    }


COMPOSITIONS = build_compositions()


# ============= POOL SPLITS =============
EXTRACTION_SIZE = 20
EVAL_SIZE = 10


def get_extraction_pool(task_pairs):
    return task_pairs[:EXTRACTION_SIZE]


def get_eval_pool(task_pairs):
    return task_pairs[EXTRACTION_SIZE:EXTRACTION_SIZE + EVAL_SIZE]


# Categorize task types
STRING_ATOMS = ["country_capital", "country_currency", "uppercase", "first_letter"]
ARITH_ATOMS = ["successor", "double", "triple"]
STRING_COMPS = ["uppercase_of_capital", "first_letter_of_capital",
                "uppercase_of_currency", "first_letter_of_currency"]
ARITH_COMPS = ["successor_of_double", "double_of_successor",
               "successor_of_triple", "triple_of_successor",
               "successor_of_successor"]
