from pathlib import Path

DOMAINS = ("Art", "Clipart", "Product", "Real World")
DOMAIN_TO_ID = {domain: index for index, domain in enumerate(DOMAINS)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EXPECTED_CLASS_COUNT = 65
DEFAULT_MODEL = "vit_base_patch16_224.augreg_in21k"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
