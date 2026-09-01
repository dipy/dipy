import pytest

from dipy.utils.optpkg import optional_package

_, has_fury, _ = optional_package("fury", min_version="2.0.0")
if not has_fury:
    pytest.skip("Requires fury>=2.0.0", allow_module_level=True)
else:
    from PIL import Image

    from dipy.viz.skyline.UI import theme


@pytest.mark.parametrize(
    "hex_color,expected",
    [
        ("#000000", (0.0, 0.0, 0.0, 1.0)),
        ("#FFFFFF", (1.0, 1.0, 1.0, 1.0)),
        ("#FF0000", (1.0, 0.0, 0.0, 1.0)),
        ("#00FF00", (0.0, 1.0, 0.0, 1.0)),
        ("#0000FF", (0.0, 0.0, 1.0, 1.0)),
    ],
)
def test_hex_to_rgba_primary_colors(hex_color, expected):
    assert theme.hex_to_rgba(hex_color) == expected


def test_hex_to_rgba_accepts_hex_without_the_leading_hash():
    assert theme.hex_to_rgba("EE942E") == theme.hex_to_rgba("#EE942E")


def test_hex_to_rgba_scales_each_channel_by_255():
    assert theme.hex_to_rgba("#804020") == pytest.approx(
        (128 / 255.0, 64 / 255.0, 32 / 255.0, 1.0)
    )


def test_hex_to_rgba_carries_the_alpha_through():
    assert theme.hex_to_rgba("#838383", alpha=0.12)[3] == 0.12


def test_theme_entries_are_rgba_tuples_in_unit_range():
    for name, color in theme.THEME.items():
        assert len(color) == 4, name
        assert all(0.0 <= channel <= 1.0 for channel in color), name


def test_theme_matches_the_documented_hex_palette():
    assert theme.THEME["background"] == theme.hex_to_rgba("#191919")
    assert theme.THEME["primary"] == theme.hex_to_rgba("#EE942E")
    assert theme.THEME["secondary"] == theme.hex_to_rgba("#FFFFFF")
    assert theme.THEME["text"] == theme.hex_to_rgba("#838383")
    assert theme.THEME["text_highlight"] == theme.THEME["primary"]
    assert theme.THEME["shadow"] == theme.hex_to_rgba("#838383", alpha=0.12)


@pytest.mark.parametrize(
    "palette,keys",
    [
        (
            "SLIDER_THEME",
            [
                "track_color",
                "track_bg",
                "thumb_color",
                "track_covered_color",
                "label_color",
                "value_color",
                "shadow_color",
            ],
        ),
        (
            "WINDOW_THEME",
            [
                "title_color",
                "title_active_color",
                "background_color",
                "collapse_color",
            ],
        ),
        (
            "SWITCH_THEME",
            [
                "background_color",
                "active_color",
                "inactive_text_color",
                "active_text_color",
                "border_color",
            ],
        ),
        (
            "DROPDOWN_THEME",
            [
                "background_color",
                "border_color",
                "hover_color",
                "selected_color",
                "arrow_color",
            ],
        ),
    ],
)
def test_component_palettes_reuse_the_base_theme(palette, keys):
    values = getattr(theme, palette)

    assert sorted(values) == sorted(keys)
    for color in values.values():
        assert color in theme.THEME.values()


def test_asset_directories_exist():
    assert theme.SKYLINE_HOME.is_dir()
    assert theme.ASSETS.is_dir()
    assert theme.FONTS.is_dir()
    assert theme.IMAGES.is_dir()
    assert theme.ASSETS.parent == theme.SKYLINE_HOME
    assert theme.FONTS.parent == theme.ASSETS
    assert theme.IMAGES.parent == theme.ASSETS


def test_fonts_are_downloaded_and_non_empty():
    assert theme.FONT.is_file()
    assert theme.FONT_AWESOME.is_file()
    assert theme.FONT.stat().st_size > 0
    assert theme.FONT_AWESOME.stat().st_size > 0


def test_logos_are_downloaded_and_resized():
    assert theme.LOGO.is_file()
    assert theme.LOGO_SMALL.is_file()
    with Image.open(theme.LOGO) as logo:
        assert logo.size == (48, 48)
    with Image.open(theme.LOGO_SMALL) as logo_small:
        assert logo_small.size == (32, 32)
