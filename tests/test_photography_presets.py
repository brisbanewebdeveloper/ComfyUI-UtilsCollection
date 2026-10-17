import pathlib
import sys
import types


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_photography_preset_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_photography_preset_test.nodes import preset_nodes
from utils_collection_photography_preset_test import presets_collection


FAMILIES = (
    ("SYSTEM_MESSAGE_STYLE_", preset_nodes.UC_SystemMessagePresets),
    ("INSTRUCT_PROMPT_STYLE_", preset_nodes.UC_InstructPromptPresets),
    ("BONUS_PROMPT_STYLE_", preset_nodes.UC_BonusPromptPresets),
)

RENAMED_EXISTING_SUFFIXES = (
    "PHOTOGRAPHY_PHOTOREALISM",
    "PHOTOGRAPHY_ANALOG_FILM",
    "PHOTOGRAPHY_EARLY_2000S_ANALOG",
    "PHOTOGRAPHY_POLAROID",
    "PHOTOGRAPHY_KODACHROME",
)

NEW_SUFFIXES = (
    "PHOTOGRAPHY_FULL_FRAME_DIGITAL",
    "PHOTOGRAPHY_MEDIUM_FORMAT",
    "PHOTOGRAPHY_CCD_COMPACT_DIGITAL",
    "PHOTOGRAPHY_SMARTPHONE",
    "PHOTOGRAPHY_STUDIO_STROBE",
    "PHOTOGRAPHY_DIRECT_FLASH",
    "PHOTOGRAPHY_AVAILABLE_LIGHT",
    "PHOTOGRAPHY_FINE_ART_BLACK_AND_WHITE",
)

REMOVED_PRESET_IDS = {
    "STYLE_PHOTOREALISM",
    "STYLE_ANALOG_FILM",
    "STYLE_EARLY_2000S_ANALOG",
    "STYLE_POLAROID",
    "STYLE_KODACHROME",
    "STYLE_CONTEMPORARY_DIGITAL_PHOTOGRAPHY",
    "STYLE_DOCUMENTARY_PHOTOGRAPHY",
    "STYLE_EDITORIAL_PHOTOGRAPHY",
    "STYLE_STUDIO_PHOTOGRAPHY",
    "STYLE_DIRECT_FLASH_PHOTOGRAPHY",
    "STYLE_SMARTPHONE_PHOTOGRAPHY",
    "STYLE_COMPACT_DIGITAL_CAMERA",
    "STYLE_MEDIUM_FORMAT_PHOTOGRAPHY",
    "STYLE_MACRO_PHOTOGRAPHY",
    "STYLE_BLACK_AND_WHITE_PHOTOGRAPHY",
}


def _value(prefix, suffix):
    return getattr(presets_collection, f"{prefix}{suffix}")


def test_photography_choices_are_grouped_and_shared():
    expected = {
        f"STYLE_{suffix}"
        for suffix in RENAMED_EXISTING_SUFFIXES + NEW_SUFFIXES
    }
    shared = set(preset_nodes.UC_UnifiedPresets.get_shared_presets())

    assert expected <= shared
    for _, node in FAMILIES:
        choices = set(node.get_presets())
        assert expected <= choices
        assert choices.isdisjoint(REMOVED_PRESET_IDS)


def test_preset_execution_returns_exact_stored_text():
    for suffix in RENAMED_EXISTING_SUFFIXES + NEW_SUFFIXES:
        preset_id = f"STYLE_{suffix}"
        for prefix, node in FAMILIES:
            stored = _value(prefix, suffix)
            assert isinstance(stored, str)
            assert node.execute(preset_id).args[0] == stored
