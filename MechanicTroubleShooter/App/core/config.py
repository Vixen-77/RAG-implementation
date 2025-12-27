
import os

DATA_FOLDER = "Data"
CHROMA_PERSIST_DIR = "./chroma_parent_child" 
                                            
CHROMA_DB_DIR = "./chroma_db"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

DEMO_QUERIES = {
    "engine_smoke": {
        "query": "I see smoke coming from under the hood, what should I do?",
        "description": "Fault: Smoke from engine compartment"
    },
    "won't_start": {
        "query": "I turn the key and the engine doesn't start, but the lights work. What could be wrong?",
        "description": "Fault: Engine fails to start (starter motor works)"
    },
    "rapid_blinking": {
        "query": "My turn signal is blinking much faster than usual. What does that mean?",
        "description": "Fault: Indicator light flashing frequency"
    },
    "steering_vibration": {
        "query": "The steering wheel shakes when I drive fast. Is something broken?",
        "description": "Fault: Vibration while driving"
    },
    "brake_pedal_soft": {
        "query": "My brake pedal feels spongy and goes all the way to the floor. Is this safe?",
        "description": "Fault: Loss of braking pressure"
    },
    "oil_pressure_light": {
        "query": "The oil pressure warning light just came on while I was driving. Should I stop?",
        "description": "Fault: Oil pressure warning"
    },
    "white_smoke_exhaust": {
        "query": "There is white smoke coming from the exhaust but the car runs fine. Is this normal?",
        "description": "Fault: Exhaust smoke (DPF regeneration)"
    },
    "coolant_boiling": {
        "query": "The coolant in the reservoir is boiling. What is the cause?",
        "description": "Fault: Cooling system malfunction"
    },
    "dpf_blocked": {
        "query": "My DPF warning light is on and the engine feels sluggish. What should I do?",
        "description": "Fault: Diesel Particulate Filter blockage"
    },
    "adblue_fault": {
        "query": "I have an AdBlue system fault warning. How do I check and refill AdBlue?",
        "description": "Fault: AdBlue system malfunction"
    },
    "fuse_location": {
        "query": "Where is the fuse for the cooling fan located and what is its amperage?",
        "description": "Info: Fuse box layout query"
    }
}
