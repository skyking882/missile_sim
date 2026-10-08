"""Expose primitive ECS scalar values through registered, typed converters."""
from pathlib import Path
import sys
path = Path(sys.argv[1]) / "python/ecs/ECSTypes.cpp"
source = path.read_text()
if "  REGISTER_TYPE(bool);" not in source:
    source = source.replace("  REGISTER_TYPE(uint8_t);", "  REGISTER_TYPE(bool);\n  REGISTER_TYPE(float);\n  REGISTER_TYPE(uint8_t);", 1)
    path.write_text(source)
    print("Registered primitive ECS scalar bindings")
elif "  REGISTER_TYPE(double);\n" in source:
    path.write_text(source.replace("  REGISTER_TYPE(double);\n", ""))
