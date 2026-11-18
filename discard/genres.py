from dataclasses import dataclass

GENRES = ["country", "rock", "pop", "hip-hop", "electronic"]

@dataclass
class Preset:
    genre: str
    style: str
    bpm_lo: int
    bpm_hi: int
    keyscales: list
    lm: str
    inference_steps: int
    guidance_scale: float
    dia: str  # "adlib" | "bridge" | "none"

_K_MAJ = ["G major", "C major", "D major"]
_K_MIN = ["A minor", "E minor", "B minor"]

PRESETS = {
    "country": Preset("country",
        "heartfelt country, warm acoustic guitar, brushed drums, mellow sincere male vocal, storytelling",
        80, 100, _K_MAJ, "acestep-5Hz-lm-1.7B", 24, 3.0, "bridge"),
    "rock": Preset("rock",
        "driving rock, distorted electric guitars, punchy drums, energetic male vocal, anthemic",
        110, 140, _K_MIN, "acestep-5Hz-lm-1.7B", 24, 4.0, "adlib"),
    "pop": Preset("pop",
        "bright modern pop, catchy synth hooks, tight drums, polished vocal, danceable",
        100, 125, _K_MAJ, "acestep-5Hz-lm-1.7B", 24, 3.0, "none"),
    "hip-hop": Preset("hip-hop",
        "boom-bap trap hip-hop, hard 808s, crisp hats, confident rap-sung vocal",
        80, 100, _K_MIN, "acestep-5Hz-lm-1.7B", 24, 3.5, "adlib"),
    "electronic": Preset("electronic",
        "melodic electronic EDM, lush synths, four-on-the-floor, airy vocal, festival energy",
        120, 130, _K_MIN, "acestep-5Hz-lm-1.7B", 24, 3.0, "none"),
}
