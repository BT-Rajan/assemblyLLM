# inference.py
import torch
import json
import networkx as nn_graph
import os
from config import AssemblyRouterConfig
from model import AssemblyRouterLM
from dataset import AssemblyTokenizer

# ==========================================
# 1. LIVE HARDWARE & ENGINE ASSEMBLY RETRIEVAL
# ==========================================
def load_trained_router(checkpoint_path, vocab_path):
    tokenizer = AssemblyTokenizer()
    
    # Overwrite basic character maps if a dynamic tokenizer vocab file was built
    if os.path.exists(vocab_path):
        with open(vocab_path, "r") as f:
            v_data = json.load(f)
            tokenizer.stoi = v_data["stoi"]
            tokenizer.itos = {int(k): v for k, v in v_data["itos"].items()}

    cfg = AssemblyRouterConfig(vocab_size=len(tokenizer.stoi))
    model = AssemblyRouterLM(cfg)
    
    # Force load maps onto CPU safely for local inference tracking
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer

# ==========================================
# 2. PROGRAMMATIC CODFIED PEN GRAPH ASSEMBLY
# ==========================================
def get_assembly_graph_matrix():
    g = nn_graph.DiGraph()
    
    # Re-map our precise physical parts and their variable characteristics
    parts = {
        "P001": {"name": "Plunger Cap", "material": "ABS Plastic", "stroke_distance": "4.5mm"},
        "P002": {"name": "Upper Cam Thruster", "material": "Nylon-66", "tooth_count": "4", "ramp_angle": "45 degrees"},
        "P003": {"name": "Lower Cam Follower", "material": "POM Polyacetal", "tracks": "Linear sliding slots"},
        "P004": {"name": "Return Spring", "material": "Music Wire Steel", "spring_rate": "0.38 N/mm", "free_length": "22.0mm"},
        "P005": {"name": "Ink Reservoir Tube", "material": "Polypropylene", "inner_volume": "1.2ml", "ink_color": "Blue"},
        "P006": {"name": "Tip Housing", "material": "Brass", "tip_size": "0.7mm Fine"},
        "P007": {"name": "Ball Bearing", "material": "Tungsten Carbide", "diameter": "0.698mm"},
        "P008": {"name": "Upper Barrel", "material": "Polycarbonate", "thread_specification": "M8 x 0.75 Internal"},
        "P009": {"name": "Lower Barrel", "material": "Polycarbonate", "thread_specification": "M8 x 0.75 External"},
    }
    for p_id, attrs in parts.items():
        g.add_node(p_id, **attrs)
        
    # Inject codified link matrices
    relations = [
        ("P001", "P002", "F:AXIAL_PUSH", "Axial downward force translation during thumb actuation"),
        ("P001", "P008", "C:SLEEVED_BY", "P001 passes inside P008 restricting lateral wobble"),
        ("P002", "P003", "F:ROTATIONAL_CAM", "Tooth geometry forces P003 to rotate and step between channels"),
        ("P003", "P005", "F:AXIAL_PUSH", "P003 directly drives the rear shoulder of the ink cartridge down"),
        ("P004", "P005", "F:COMPRESSION", "Spring anchors onto an extruded shoulder on P005 counteracting force"),
        ("P004", "P009", "M:BUTT_JOINT", "Forward edge of spring rests rigidly inside the internal tip step of P009"),
        ("P005", "P006", "M:PRESS_FIT", "Interference fit sealing ink passage via friction lock"),
        ("P006", "P007", "C:CRIMPED_HOLD", "The brass casing is rolled slightly over the equator of the ball"),
        ("P008", "P009", "M:THREADS", "Screwed together securely via matching internal/external threads"),
    ]
    for src, dst, code, logic in relations:
        g.add_edge(src, dst, interface_code=code, connection_logic=logic)
    return g

# ==========================================
# 3. AUTOREGRESSIVE GENERATION LOOP
# ==========================================
def generate_router_token(model, tokenizer, prompt_text, max_len=128):
    full_prompt = f"User: {prompt_text}\nBot: "
    tokens = tokenizer.encode(full_prompt)
    
    # Strip trailing closing markers to give the model generation freedom
    if tokens[-1] == tokenizer.stoi["</s>"]: tokens.pop()
    
    input_tensor = torch.tensor([tokens], dtype=torch.long)
    
    with torch.no_grad():
        for _ in range(max_len):
            logits, _ = model(input_tensor)
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            
            if next_token == tokenizer.stoi["</s>"]: break
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]])], dim=1)
            
    generated_sequence = input_tensor[0].tolist()[len(tokens):]
    return tokenizer.decode(generated_sequence)

# ==========================================
# 4. ACTIVE KNOWLEDGE QUERY EXECUTION INTERCEPTOR
# ==========================================
def execute_system_query(user_input, model, tokenizer, graph):
    # Step A: The LLM reads natural language and isolates structural intent tokens
    raw_json_token = generate_router_token(model, tokenizer, user_input)

    try:
        # The model occasionally emits a little leading noise before the actual JSON
        # object. These route objects are always flat (no nested braces), so the last
        # '{' through the last '}' reliably isolates the real object.
        start, end = raw_json_token.rfind("{"), raw_json_token.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object boundaries found")
        command = json.loads(raw_json_token[start:end + 1])
        action = command.get("action")
        target = command.get("target")
        
        # Branch A: Precision database dictionary lookup
        if action == "lookup":
            prop = command.get("property")
            if graph.has_node(target):
                val = graph.nodes[target].get(prop, "Property profile invalid")
                part_name = graph.nodes[target].get("name", "Unknown Part")
                return f"[DATABASE RESULT] {target} ({part_name}) -> '{prop}' value is: {val}"
        
        # Branch B: Downstream mathematical network tracking
        elif action == "trace_downstream":
            if graph.has_node(target):
                descendants = list(nn_graph.descendants(graph, target))
                descendants.sort()
                return f"[GRAPH ANALYSIS RESULT] Component '{target}' structural dependency failure risks impacting downstream components: {descendants}"
                
        # Branch C: Matrix structural check interface lookup
        elif action == "matrix_check":
            src, dst = command.get("source"), command.get("destination")
            if graph.has_edge(src, dst):
                edge_data = graph.get_edge_data(src, dst)
                return f"[MATRIX JOINT CONFIRMED] Connection {src} -> {dst} | Code: {edge_data['interface_code']} | Logic: {edge_data['connection_logic']}"
                
    except Exception:
        pass
        
    return f"[PARSING ANOMALY] Neural network generation loop returned corrupted route signature: {raw_json_token}"

# ==========================================
# 5. USER CONSOLE RUNNER
# ==========================================
if __name__ == "__main__":
    print("\n--- Initializing Independent Assembly LLM Router & Matrix Graph ---")
    pen_graph = get_assembly_graph_matrix()
    
    try:
        router_model, sys_tokenizer = load_trained_router("checkpoints/best_model.pt", "data/tokenizer_vocab.json")
        print("System fully active. Enter an assembly inquiry below.")
        
        # Test Sample Interrogations
        test_queries = [
            "What is the thread_specification profile assigned to component P008?",
            "If part P001 breaks down, what downstream units lose operational integrity?",
            "Check the codified design matrix connection mapping from P005 to P006."
        ]
        
        for idx, q in enumerate(test_queries, 1):
            print(f"\nInquiry #{idx}: {q}")
            response = execute_system_query(q, router_model, sys_tokenizer, pen_graph)
            print(response)
            
    except FileNotFoundError:
        print("\nError: Could not locate 'checkpoints/best_model.pt'. Please make sure the train.py execution pipeline finishes saving the optimized system weights first.")