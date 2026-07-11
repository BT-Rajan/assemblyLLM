# prepare_data.py
import json
import random
import os

# 1. CORE SYSTEM DOMAIN METADATA DEFINITIONS
PARTS_REGISTRY = {
    "P001": {"name": "Plunger Cap", "material": "ABS Plastic", "color": "Translucent Black", "stroke_distance": "4.5mm"},
    "P002": {"name": "Upper Cam Thruster", "material": "Nylon-66", "tooth_count": "4", "ramp_angle": "45 degrees"},
    "P003": {"name": "Lower Cam Follower", "material": "POM Polyacetal", "tracks": "Linear sliding slots", "radial_step_depth": "2.0mm"},
    "P004": {"name": "Return Spring", "material": "Music Wire Steel", "spring_rate": "0.38 N/mm", "free_length": "22.0mm", "wire_diameter": "0.45mm", "active_coils": "14"},
    "P005": {"name": "Ink Reservoir Tube", "material": "Polypropylene", "outer_diameter": "3.0mm", "inner_volume": "1.2ml", "ink_type": "Oil-based Gel", "ink_color": "Blue"},
    "P006": {"name": "Tip Housing", "material": "Brass", "target_fit": "Press-fit to P005", "tip_size": "0.7mm Fine"},
    "P007": {"name": "Ball Bearing", "material": "Tungsten Carbide", "diameter": "0.698mm", "sphericity_tolerance": "0.001mm"},
    "P008": {"name": "Upper Barrel", "material": "Polycarbonate", "thread_specification": "M8 x 0.75 Internal", "grip_zone_profile": "None"},
    "P009": {"name": "Lower Barrel", "material": "Polycarbonate", "thread_specification": "M8 x 0.75 External", "ergonomic_sleeve": "Molded Rubber TPE", "tip_opening_clearance": "2.4mm"},
}

MATRIX_RELATIONS = [
    {"source": "P001", "dest": "P002", "code": "F:AXIAL_PUSH", "logic": "Axial downward force translation during thumb actuation"},
    {"source": "P001", "dest": "P008", "code": "C:SLEEVED_BY", "logic": "P001 passes inside P008 restricting lateral wobble"},
    {"source": "P002", "dest": "P003", "code": "F:ROTATIONAL_CAM", "logic": "Tooth geometry forces P003 to rotate and step between channels"},
    {"source": "P003", "dest": "P005", "code": "F:AXIAL_PUSH", "logic": "P003 directly drives the rear shoulder of the ink cartridge down"},
    {"source": "P004", "dest": "P005", "code": "F:COMPRESSION", "logic": "Spring anchors onto an extruded shoulder on P005 counteracting downward force"},
    {"source": "P004", "dest": "P009", "code": "M:BUTT_JOINT", "logic": "Forward edge of spring rests rigidly inside the internal tip step of P009"},
    {"source": "P005", "dest": "P006", "code": "M:PRESS_FIT", "logic": "Interference fit sealing ink passage via friction lock"},
    {"source": "P006", "dest": "P007", "code": "C:CRIMPED_HOLD", "logic": "The brass casing is rolled slightly over the equator of the ball to lock it in place"},
    {"source": "P008", "dest": "P009", "code": "M:THREADS", "logic": "Screwed together securely via matching internal/external threads"},
]

# 2. SEED TEMPLATE VARIATIONS FOR SYNTHETIC AUGMENTATION
TEMPLATES_LOOKUP = [
    "What is the {prop} profile assigned to component {part}?",
    "Get me the exact structural {prop} specs for part {part}.",
    "Extract telemetry values for property {prop} belonging to item {part}.",
    "System diagnostics inquiry: check {prop} on item {part}.",
    "Show the physical {prop} configuration metric for {part}.",
    "What's the {prop} of {part}?",
    "Can you tell me the {prop} for part {part}?",
    "{part} - what is its {prop}?",
    "I need the {prop} value for component {part}.",
    "Does {part} have a specific {prop}? What is it?",
]

TEMPLATES_TRACE = [
    "If part {part} fails or breaks down, what downstream units lose operational integrity?",
    "Trace the cascading structural system impact if component {part} is compromised.",
    "Run a downstream dependency sweep starting from component {part}.",
    "What happens across the assembly loop if {part} experiences physical deformation?",
    "Analyze downstream faults triggered by an isolated collapse of component {part}.",
    "If {part} breaks, what else stops working?",
    "What relies on {part}?",
    "What's downstream of {part}?",
    "Which parts would be affected if {part} failed?",
]

TEMPLATES_MATRIX = [
    "How does component {src} transfer or pass loads down to {dst}?",
    "Identify the operational joint boundary interface code between {src} and {dst}.",
    "Check the codified design matrix connection mapping from {src} to {dst}.",
    "What mechanical or structural constraint links part {src} directly to {dst}?",
    "How are {src} and {dst} connected?",
    "Is there a direct connection from {src} to {dst}?",
    "What's the interface between {src} and {dst}?",
]

# Real users don't type snake_case field names - they ask about "thread specification",
# not "thread_specification". Every lookup template gets a chance to use this natural form
# instead of the raw dict key, so the model learns both instead of only the raw key it's
# never actually going to see typed by an actual person.
def natural_property_name(prop_key):
    return prop_key.replace("_", " ")

# 2b. NEGATIVE / OUT-OF-SCOPE EXAMPLES
# Nothing above teaches the model what to do with a question outside these three action
# types - broad "what's in this thing" questions, or genuinely unrelated questions. Without
# examples like these, an out-of-scope prompt just gets best-effort character prediction
# with no learned "this isn't something I handle" behavior at all.
LISTING_PROMPTS = [
    "What are the components of a ballpoint pen?",
    "What parts make up this assembly?",
    "List all the parts in this pen.",
    "What is a ballpoint pen made of?",
    "Give me a breakdown of every component in this device.",
    "What pieces does this assembly consist of?",
    "Show me the full parts list.",
    "What are all the components of an ink pen?",
    "How many parts does this pen have, and what are they?",
    "What's inside this pen?",
]

UNSUPPORTED_PROMPTS = [
    "What's the weather like today?",
    "Who invented the ballpoint pen?",
    "How much does this pen cost?",
    "Can you write me a poem about pens?",
    "What's your favorite color?",
    "How do I file my taxes?",
    "What time is it?",
    "Tell me a joke.",
    "What's the capital of France?",
    "Can you recommend a good pen brand?",
]

def perturb(prompt):
    """Cheap, honest lexical variations - not padding the count with literal duplicates."""
    variants = {prompt}
    if prompt.endswith("?"):
        variants.add(prompt[:-1] + ".")
    if prompt[0].isupper():
        variants.add(prompt[0].lower() + prompt[1:])
    variants.add("Quick question - " + prompt[0].lower() + prompt[1:])
    return variants

# 3. GENERATION PIPELINE ENGINE
def generate_synthetic_dataset(output_size=2500, eval_holdout_frac=0.15, seed=42):
    rng = random.Random(seed)

    # Each "family" below is a group of (prompt, completion) pairs that all share the same
    # underlying entity (a specific part+property, a specific trace target, or a specific
    # edge). Holding out entire families for eval - rather than randomly splitting
    # individual prompt/completion pairs - is what makes the eval set actually measure
    # generalization instead of just re-testing phrasings of answers already seen in
    # training.
    families = []

    for part_id, attrs in PARTS_REGISTRY.items():
        for prop_key in attrs.keys():
            if prop_key == "name":
                continue
            target_json = json.dumps({"action": "lookup", "target": part_id, "property": prop_key})
            prompts = set()
            for template in TEMPLATES_LOOKUP:
                natural = natural_property_name(prop_key)
                prop_variant = natural if rng.random() < 0.5 else prop_key
                prompts |= perturb(template.format(prop=prop_variant, part=part_id))
            families.append([{"prompt": p, "completion": target_json} for p in prompts])

    for part_id in PARTS_REGISTRY.keys():
        target_json = json.dumps({"action": "trace_downstream", "target": part_id})
        prompts = set()
        for template in TEMPLATES_TRACE:
            prompts |= perturb(template.format(part=part_id))
        families.append([{"prompt": p, "completion": target_json} for p in prompts])

    for edge in MATRIX_RELATIONS:
        target_json = json.dumps({"action": "matrix_check", "source": edge["source"], "destination": edge["dest"]})
        prompts = set()
        for template in TEMPLATES_MATRIX:
            prompts |= perturb(template.format(src=edge["source"], dst=edge["dest"]))
        families.append([{"prompt": p, "completion": target_json} for p in prompts])

    # Negative examples are their own single-prompt "families" so they can still be held
    # out for eval like everything else.
    for p in LISTING_PROMPTS:
        families.append([{"prompt": v, "completion": json.dumps({"action": "list_all_parts"})} for v in perturb(p)])
    for p in UNSUPPORTED_PROMPTS:
        families.append([{"prompt": v, "completion": json.dumps({"action": "unsupported"})} for v in perturb(p)])

    rng.shuffle(families)
    holdout_count = max(1, int(len(families) * eval_holdout_frac))
    eval_families = families[:holdout_count]
    train_families = families[holdout_count:]

    train_data = [row for fam in train_families for row in fam]
    eval_data = [row for fam in eval_families for row in fam]
    rng.shuffle(train_data)
    rng.shuffle(eval_data)

    total_available = len(train_data) + len(eval_data)
    if total_available < output_size:
        print(f"Note: requested output_size={output_size}, but genuine template+entity diversity "
              f"only supports {total_available} unique examples. Using all {total_available} rather "
              f"than padding with literal duplicates, which would inflate the count without adding "
              f"any real signal. Add more templates/entities to raise this ceiling.")
    else:
        # Only trim - never fabricate duplicate rows to hit a target above the real ceiling.
        train_data = train_data[:int(output_size * (1 - eval_holdout_frac))]
        eval_data = eval_data[:int(output_size * eval_holdout_frac)]

    # Serialize down to disk outputs
    os.makedirs("data", exist_ok=True)

    with open("data/train.jsonl", "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open("data/eval.jsonl", "w") as f:
        for item in eval_data:
            f.write(json.dumps(item) + "\n")

    print(f"Data Generation complete. Saved {len(train_data)} training items & {len(eval_data)} "
          f"verification items to data/ folder ({len(train_families)} train / {len(eval_families)} "
          f"eval entity families - eval families never share an entity with train, so eval loss "
          f"actually reflects generalization).")

if __name__ == "__main__":
    generate_synthetic_dataset(output_size=2500)