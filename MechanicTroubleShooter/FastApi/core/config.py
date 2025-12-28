import os

DATA_FOLDER = "Data"
CHROMA_PERSIST_DIR = "./chroma_parent_child" 
CHROMA_DB_DIR = "./chroma_db"

OLLAMA_URL = "http://localhost:11434/api/generate"
TEXT_MODEL = "llama3.1"
VISION_MODEL = "llava-phi3"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

DEMO_QUERIES = {
    "engine_smoke": {
        "query": "I see smoke coming from under the hood, what should I do?"
    },
    "won't_start": {
        "query": "I turn the key and the engine doesn't start, but the lights work. What could be wrong?"
    },
    "rapid_blinking": {
        "query": "My turn signal is blinking much faster than usual. What does that mean?"
    },
    "steering_vibration": {
        "query": "The steering wheel shakes when I drive fast. Is something broken?"
    },
    "brake_pedal_soft": {
        "query": "My brake pedal feels spongy and goes all the way to the floor. Is this safe?"
    },
    "oil_pressure_light": {
        "query": "The oil pressure warning light just came on while I was driving. Should I stop?"
    },
    "white_smoke_exhaust": {
        "query": "There is white smoke coming from the exhaust but the car runs fine. Is this normal?"
    },
    "coolant_boiling": {
        "query": "The coolant in the reservoir is boiling. What is the cause?"
    },
    "dpf_blocked": {
        "query": "My DPF warning light is on and the engine feels sluggish. What should I do?"
    },
    "adblue_fault": {
        "query": "I have an AdBlue system fault warning. How do I check and refill AdBlue?"
    },
    "fuse_location": {
        "query": "Where is the fuse for the cooling fan located and what is its amperage?"
    }
}
