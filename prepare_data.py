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
]

TEMPLATES_TRACE = [
    "If part {part} fails or breaks down, what downstream units lose operational integrity?",
    "Trace the cascading structural system impact if component {part} is compromised.",
    "Run a downstream dependency sweep starting from component {part}.",
    "What happens across the assembly loop if {part} experiences physical deformation?",
    "Analyze downstream faults triggered by an isolated collapse of component {part}.",
]

TEMPLATES_MATRIX = [
    "How does component {src} transfer or pass loads down to {dst}?",
    "Identify the operational joint boundary interface code between {src} and {dst}.",
    "Check the codified design matrix connection mapping from {src} to {dst}.",
    "What mechanical or structural constraint links part {src} directly to {dst}?",
]

# 3. GENERATION PIPELINE ENGINE
def generate_synthetic_dataset(output_size=3000):
    dataset = []
    
    # Task Category 1: Flexible Lookup Generating
    for part_id, attrs in PARTS_REGISTRY.items():
        for prop_key in attrs.keys():
            if prop_key == "name": continue
            for template in TEMPLATES_LOOKUP:
                prompt = template.format(prop=prop_key, part=part_id)
                target_json = {"action": "lookup", "target": part_id, "property": prop_key}
                dataset.append({"prompt": prompt, "completion": json.dumps(target_json)})
                
    # Task Category 2: Trace Cascading Graph Loops
    for part_id in PARTS_REGISTRY.keys():
        for template in TEMPLATES_TRACE:
            prompt = template.format(part=part_id)
            target_json = {"action": "trace_downstream", "target": part_id}
            dataset.append({"prompt": prompt, "completion": json.dumps(target_json)})
            
    # Task Category 3: Interface Interconnection Matrices
    for edge in MATRIX_RELATIONS:
        for template in TEMPLATES_MATRIX:
            prompt = template.format(src=edge["source"], dst=edge["dest"])
            target_json = {"action": "matrix_check", "source": edge["source"], "destination": edge["dest"]}
            dataset.append({"prompt": prompt, "completion": json.dumps(target_json)})

    # Augment and balance by shuffling or introducing slight natural language noise
    random.seed(42)
    random.shuffle(dataset)
    
    # Trim or loop to exact request size boundaries
    dataset = dataset[:output_size] if len(dataset) >= output_size else dataset
    
    # Split into a clean 90/10 train and validation division
    split_idx = int(len(dataset) * 0.9)
    train_data = dataset[:split_idx]
    eval_data = dataset[split_idx:]
    
    # Serialize down to disk outputs
    os.makedirs("data", exist_ok=True)
    
    with open("data/train.jsonl", "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
            
    with open("data/eval.jsonl", "w") as f:
        for item in eval_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Data Generation complete. Saved {len(train_data)} training items & {len(eval_data)} verification items to data/ folder.")

if __name__ == "__main__":
    generate_synthetic_dataset(output_size=2500)