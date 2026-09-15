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


VISIBLE_PHOTOGRAPHIC_SIGNATURES = {
    "PHOTOGRAPHY_ANALOG_FILM": (
        "color-negative",
        "grain",
        "highlight",
        "halation",
        "shadows",
        "color separation",
        "microcontrast",
        "focus falloff",
    ),
    "PHOTOGRAPHY_EARLY_2000S_ANALOG": (
        "point-and-shoot",
        "minilab",
        "midtones",
        "grain",
        "compact-lens",
        "automatic-exposure",
        "highlight latitude",
    ),
    "PHOTOGRAPHY_FULL_FRAME_DIGITAL": (
        "aperture",
        "depth of field",
        "exposure",
        "highlight",
        "shadow",
        "low iso",
        "texture",
        "light",
    ),
    "PHOTOGRAPHY_MEDIUM_FORMAT": (
        "wide aperture",
        "shallow depth of field",
        "bokeh",
        "tonal separation",
        "highlight",
        "color gradation",
        "low iso",
        "texture",
    ),
    "PHOTOGRAPHY_CCD_COMPACT_DIGITAL": (
        "short focal length",
        "small aperture",
        "deep depth of field",
        "automatic exposure",
        "direct color",
        "local contrast",
        "clipped",
        "chroma noise",
        "edge",
    ),
    "PHOTOGRAPHY_SMARTPHONE": (
        "wide angle",
        "deep depth of field",
        "automatic exposure",
        "lifted shadow",
        "controlled highlight",
        "edge sharpening",
        "low noise",
        "natural color",
    ),
    "PHOTOGRAPHY_STUDIO_STROBE": (
        "key light",
        "fill",
        "specular",
        "shadow",
        "white balance",
        "moderate aperture",
        "sharp",
        "low iso",
        "ambient",
    ),
    "PHOTOGRAPHY_DIRECT_FLASH": (
        "frontal flash",
        "hard edged shadows",
        "falloff",
        "near plane",
        "darker",
        "specular",
        "moderate aperture",
        "frozen motion",
        "contrast",
    ),
    "PHOTOGRAPHY_AVAILABLE_LIGHT": (
        "existing",
        "color temperature",
        "wide aperture",
        "shallow depth of field",
        "exposure",
        "shadow",
        "highlight rolloff",
        "high iso",
        "grain",
    ),
    "PHOTOGRAPHY_FINE_ART_BLACK_AND_WHITE": (
        "black-and-white",
        "grayscale",
        "no residual color",
        "tonal separation",
        "midtones",
        "blacks",
        "highlight rolloff",
        "monochrome grain",
        "focus",
    ),
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






def test_photographic_signatures_are_visible_and_distinct():
    for suffix, required_terms in VISIBLE_PHOTOGRAPHIC_SIGNATURES.items():
        components = [_value(prefix, suffix) for prefix, _ in FAMILIES]
        combined = " ".join(components).lower()
        assert all(term in combined for term in required_terms), suffix
        assert len(set(components)) == len(components)


def test_component_roles_are_complementary():
    for suffix in VISIBLE_PHOTOGRAPHIC_SIGNATURES:
        system = _value("SYSTEM_MESSAGE_STYLE_", suffix)
        instruct = _value("INSTRUCT_PROMPT_STYLE_", suffix)
        bonus = _value("BONUS_PROMPT_STYLE_", suffix)

        assert system.startswith("You are an AI that specializes in writing image-editing descriptions")
        assert "one concise structured instruction" in system
        assert instruct.startswith("Render the reference as")
        assert "Preserve the composition" in instruct or "preserve the composition" in instruct
        assert "Preserve the" not in bonus
        assert len(system) > len(instruct) > len(bonus)


def test_photography_presets_do_not_reframe_or_invent_content():
    banned = (
        "close-up",
        "wide shot",
        "camera angle",
        "magnif",
        "reframe",
        "crop the",
        "add a",
        "add an",
        "subject",
        "person",
        "animal",
        "portrait",
        "product",
    )
    for suffix in VISIBLE_PHOTOGRAPHIC_SIGNATURES:
        for prefix, _ in FAMILIES:
            normalized = _value(prefix, suffix).lower()
            assert all(term not in normalized for term in banned), (suffix, prefix)


def test_smartphone_preset_does_not_name_devices_or_interfaces():
    banned = (
        "smartphone",
        "mobile",
        "camera",
        "screen",
        "interface",
        "viewfinder",
        "overlay",
    )
    for prefix, _ in FAMILIES:
        normalized = _value(prefix, "PHOTOGRAPHY_SMARTPHONE").lower()
        assert all(term not in normalized for term in banned), prefix


def test_analog_presets_do_not_stack_distressed_media_artifacts():
    banned = (
        "heavy grain",
        "light leak",
        "scratch",
        "dust",
        "faded",
        "damage",
        "grainy and raw",
        "imperfection",
    )
    for suffix in ("PHOTOGRAPHY_ANALOG_FILM", "PHOTOGRAPHY_EARLY_2000S_ANALOG"):
        combined = " ".join(_value(prefix, suffix) for prefix, _ in FAMILIES).lower()
        assert all(term not in combined for term in banned)


def test_preset_execution_returns_exact_stored_text():
    for suffix in RENAMED_EXISTING_SUFFIXES + NEW_SUFFIXES:
        preset_id = f"STYLE_{suffix}"
        for prefix, node in FAMILIES:
            stored = _value(prefix, suffix)
            assert isinstance(stored, str)
            assert node.execute(preset_id).args[0] == stored
