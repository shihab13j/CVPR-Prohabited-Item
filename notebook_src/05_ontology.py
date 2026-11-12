# ==============================================================================
# 4. UNIFIED ONTOLOGY (three evaluation tracks, never mixed):
#      EXACT        : same canonical class name only
#      HARMONIZED   : + documented hyponym aliases (e.g. folding_knife IS-A knife)
#      HIERARCHICAL : threat superclass (sharp / firearm / blunt_tool / ...)
#    An object is UNKNOWN for an experiment iff it is not known under the HARMONIZED track.
# ==============================================================================
CANON = {
    "gun": "gun", "pistol": "gun", "knife": "knife", "knives": "knife", "wrench": "wrench", "pliers": "pliers",
    "plier": "pliers", "scissors": "scissors", "scissor": "scissors", "hammer": "hammer",
    "handcuffs": "handcuffs", "handcuff": "handcuffs", "baton": "baton", "sprayer": "sprayer", "powerbank": "powerbank",
    "lighter": "lighter", "bullet": "bullet",
    "foldingknife": "folding_knife", "straightknife": "straight_knife", "utilityknife": "utility_knife",
    "multitoolknife": "multitool_knife", "multiknife": "multitool_knife",
    "portablecharger1": "portable_charger_1", "portablecharger2": "portable_charger_2", "mobilephone": "mobile_phone",
    "laptop": "laptop", "tablet": "tablet", "cosmetic": "cosmetic", "water": "water",
    "nonmetalliclighter": "nonmetallic_lighter",
    "blade": "blade", "dagger": "dagger", "swissarmyknife": "swiss_army_knife", "cans": "cans",
    "cartondrinks": "carton_drinks", "glassbottle": "glass_bottle", "plasticbottle": "plastic_bottle",
    "vacuumcup": "vacuum_cup", "spraycans": "spray_cans", "tin": "tin",
}


def canon(name):
    k = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return CANON.get(k, re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_"))


# Explicit hyponym aliases (child IS-A parent per the dataset papers' class definitions). Reviewed:
#  * OPIXray knife sub-types and CLCXray dagger / Swiss-army knife are knives;
#  * HiXray "Portable_Charger_1/2" are power banks (PIDray "powerbank");
#  * HiXray "Nonmetallic_Lighter" is a lighter.
#  REMOVED after review: blade -> knife (CLCXray blade = razor blade) and spray_cans -> sprayer
#  (aerosol cans vs. PIDray sprayer); these are left UNKNOWN / superclass-only.
SEM_ALIAS = {"folding_knife": "knife", "straight_knife": "knife", "utility_knife": "knife", "multitool_knife": "knife",
             "swiss_army_knife": "knife", "dagger": "knife",
             "portable_charger_1": "powerbank", "portable_charger_2": "powerbank",
             "nonmetallic_lighter": "lighter"}

SUPER = {  # threat superclass (level 3 of the hierarchy)
    "firearm": ["gun", "bullet"],
    "sharp": ["knife", "folding_knife", "straight_knife", "utility_knife", "multitool_knife", "swiss_army_knife",
              "dagger", "blade", "scissors"],
    "blunt_tool": ["wrench", "pliers", "hammer", "baton"],
    "restraint": ["handcuffs"],
    "pressurized": ["sprayer", "spray_cans"],
    "fire_hazard": ["lighter", "nonmetallic_lighter"],
    "battery_electronic": ["powerbank", "portable_charger_1", "portable_charger_2", "mobile_phone", "laptop", "tablet"],
    "liquid_container": ["water", "cosmetic", "cans", "carton_drinks", "glass_bottle", "plastic_bottle",
                         "vacuum_cup", "tin"],
}
SUPER_OF = {c: s for s, cs in SUPER.items() for c in cs}

ATTR_GROUPS = OrderedDict(
    shape=["elongated", "pointed_edge", "jointed", "rectangular", "cylindrical", "curved", "compact", "ring_shaped"],
    material=["metallic", "dense", "plastic", "liquid", "circuitry", "battery_cell", "glass"],
    function=["cutting", "gripping", "striking", "projectile", "ignition", "restraint", "pressurized",
              "power_storage", "containing", "communication"],
)
ATTRS = [a for g in ATTR_GROUPS.values() for a in g]
_knife = {"elongated", "pointed_edge", "metallic", "cutting"}
CLASS_ATTR = {
    "gun": {"jointed", "curved", "metallic", "dense", "projectile"},
    "bullet": {"cylindrical", "compact", "metallic", "dense", "projectile"},
    "knife": _knife, "straight_knife": _knife, "dagger": _knife | {"dense"}, "blade": {"rectangular", "compact", "metallic", "cutting"},
    "folding_knife": _knife | {"jointed"}, "utility_knife": _knife | {"rectangular", "plastic"},
    "multitool_knife": _knife | {"jointed", "gripping"}, "swiss_army_knife": _knife | {"jointed", "compact"},
    "scissors": {"elongated", "pointed_edge", "jointed", "ring_shaped", "metallic", "cutting"},
    "wrench": {"elongated", "metallic", "dense", "gripping", "striking"},
    "pliers": {"elongated", "jointed", "metallic", "gripping"},
    "hammer": {"elongated", "metallic", "dense", "striking"},
    "baton": {"elongated", "cylindrical", "dense", "striking"},
    "handcuffs": {"ring_shaped", "jointed", "metallic", "restraint"},
    "sprayer": {"cylindrical", "metallic", "liquid", "pressurized", "containing"},
    "spray_cans": {"cylindrical", "metallic", "liquid", "pressurized", "containing"},
    "lighter": {"rectangular", "compact", "metallic", "liquid", "ignition"},
    "nonmetallic_lighter": {"rectangular", "compact", "plastic", "liquid", "ignition"},
    "powerbank": {"rectangular", "dense", "circuitry", "battery_cell", "power_storage"},
    "portable_charger_1": {"rectangular", "dense", "circuitry", "battery_cell", "power_storage"},
    "portable_charger_2": {"rectangular", "compact", "circuitry", "battery_cell", "power_storage"},
    "mobile_phone": {"rectangular", "compact", "circuitry", "battery_cell", "communication"},
    "laptop": {"rectangular", "dense", "circuitry", "battery_cell", "communication"},
    "tablet": {"rectangular", "circuitry", "battery_cell", "communication"},
    "water": {"cylindrical", "plastic", "liquid", "containing"},
    "cosmetic": {"cylindrical", "compact", "plastic", "liquid", "containing"},
    "cans": {"cylindrical", "metallic", "liquid", "containing"}, "tin": {"cylindrical", "metallic", "containing"},
    "carton_drinks": {"rectangular", "liquid", "containing"},
    "glass_bottle": {"cylindrical", "glass", "liquid", "containing"},
    "plastic_bottle": {"cylindrical", "plastic", "liquid", "containing"},
    "vacuum_cup": {"cylindrical", "metallic", "liquid", "containing"},
}
TEXT_NAME = {c: c.replace("_", " ") for c in list(CLASS_ATTR)}
TEXT_NAME.update(powerbank="power bank", portable_charger_1="portable charger", portable_charger_2="portable charger",
                 nonmetallic_lighter="non-metallic lighter", multitool_knife="multi-tool knife")
# global order used to number classes in every experiment
GLOBAL_ORDER = ["gun", "knife", "wrench", "pliers", "scissors", "hammer", "handcuffs", "baton", "sprayer", "powerbank",
                "lighter", "bullet", "folding_knife", "straight_knife", "utility_knife", "multitool_knife",
                "portable_charger_1", "portable_charger_2", "mobile_phone", "laptop", "tablet", "cosmetic", "water",
                "nonmetallic_lighter", "blade", "dagger", "swiss_army_knife", "cans", "carton_drinks", "glass_bottle",
                "plastic_bottle", "vacuum_cup", "spray_cans", "tin"]


def attr_vector(c):
    return np.array([1.0 if a in CLASS_ATTR.get(c, set()) else 0.0 for a in ATTRS], dtype=np.float32)


def superclass(c):
    return SUPER_OF.get(c, SUPER_OF.get(SEM_ALIAS.get(c, ""), "other"))


def resolve_known(c, known, track="harmonized"):
    """track='exact': same name only. track='harmonized': + SEM_ALIAS. Returns the known class or None."""
    if c in known:
        return c
    if track == "harmonized" and C.USE_SEMANTIC_ALIASES and SEM_ALIAS.get(c) in known:
        return SEM_ALIAS[c]
    return None


def order_classes(cs):
    cs = sorted(set(cs), key=lambda c: (GLOBAL_ORDER.index(c) if c in GLOBAL_ORDER else 999, c))
    return cs


ONTOLOGY_VERSION = sha(dict(canon=CANON, alias=SEM_ALIAS if C.USE_SEMANTIC_ALIASES else {}, super=SUPER,
                            attrs={k: sorted(v) for k, v in CLASS_ATTR.items()}))
print("ontology version", ONTOLOGY_VERSION)
