"""
LED Controller

Manages all LED updates for Launchpad.
Centralizes LED logic and prevents duplication.
"""

import logging
import typing as t
from lumiblox.common.enums import AppState, ButtonType
from lumiblox.devices.launchpad import LaunchpadMK2
from lumiblox.controller.background_animator import BackgroundAnimator
from lumiblox.controller.sequence_controller import PlaybackState
from lumiblox.common.config import get_config

logger = logging.getLogger(__name__)


class LEDController:
    """
    Manages all LED displays on the Launchpad.
    
    Responsibilities:
    - Scene button LEDs
    - Sequence button LEDs
    - Control button LEDs
    - Background animations
    """
    
    def __init__(self, launchpad: LaunchpadMK2, animator: BackgroundAnimator):
        """Initialize LED controller."""
        self.launchpad = launchpad
        self.animator = animator
        self.config = get_config()
    
    def update_scene_led(self, scene: t.Tuple[int, int], active: bool, page: int = 0) -> None:
        """Update LED for a scene button."""
        if not self.launchpad.is_connected:
            return
        
        color = self._get_scene_color(scene, active, page)
        self.launchpad.set_button_led(
            ButtonType.SCENE,
            [scene[0], scene[1]],
            color
        )
    
    def update_scene_led_other_page(self, scene: t.Tuple[int, int], other_page: int, dim_factor: float = 0.25) -> None:
        """Show a dimmed other-page color to hint a scene is active on another page."""
        if not self.launchpad.is_connected:
            return
        
        color = self._get_scene_color(scene, True, other_page)
        if isinstance(color, str):
            from lumiblox.common.utils import hex_to_rgb
            color = hex_to_rgb(color)
        dimmed = [c * dim_factor for c in color]
        self.launchpad.set_button_led(
            ButtonType.SCENE,
            [scene[0], scene[1]],
            dimmed
        )
    
    def update_sequence_led(self, index: t.Tuple[int, int], active: bool, is_multi_step: bool = False) -> None:
        """Update LED for a sequence button."""
        if not self.launchpad.is_connected:
            return
        
        if active:
            color = self.config.data["colors"]["preset_on"]
        else:
            color = self.config.data["colors"]["off"]
        
        self.launchpad.set_button_led(
            ButtonType.SEQUENCE,
            [index[0], index[1]],
            color
        )
    
    def update_control_led(self, coordinates: t.Tuple[int, int], color_key: str) -> None:
        """Update LED for a control button."""
        if not self.launchpad.is_connected:
            return
        
        color = self.config.data["colors"].get(color_key, [0, 0, 0])
        
        self.launchpad.set_button_led(
            ButtonType.CONTROL,
            [coordinates[0], coordinates[1]],
            color
        )
    
    def update_background(self, animation_type: str, app_state) -> bool:
        """Update background animation on inactive LEDs."""
        if not self.launchpad.is_connected:
            return False

        # Get complete background buffer with animation, zone colors, and brightness
        background_buffer = self.animator.get_background(
            animation_type=animation_type, app_state=app_state
        )

        # Apply background to inactive LEDs (those with no foreground color set)
        for x in range(9):
            for y in range(9):
                if not self.launchpad.pixel_buffer_output[x, y, :].any():
                    color = background_buffer[x, y, :].tolist()
                    self.launchpad.set_led(x, y, color)

        return True
    
    def clear_sequence_leds(self) -> None:
        """Clear all sequence button LEDs."""
        if not self.launchpad.is_connected:
            return
        
        for x in range(8):
            for y in range(3):
                self.launchpad.set_button_led(
                    ButtonType.SEQUENCE,
                    [x, y],
                    self.config.data["colors"]["off"]
                )
    
    def update_sequence_leds_for_save_mode(self, save_mode_type: str, existing_indices: t.Set[t.Tuple[int, int]]) -> None:
        """Update all sequence LEDs for save mode."""
        if not self.launchpad.is_connected:
            return
        
        for x in range(8):
            for y in range(3):
                index = (x, y)
                has_sequence = index in existing_indices
                
                if has_sequence:
                    if save_mode_type == "shift":
                        color = self.config.data["colors"]["preset_save_shift_mode"]
                    else:
                        color = self.config.data["colors"]["preset_save_mode"]
                else:
                    if save_mode_type == "normal":
                        # Show empty slots in save mode
                        from lumiblox.common.utils import hex_to_rgb
                        base_color = hex_to_rgb(
                            self.config.data["colors"]["save_mode_preset_background"]
                        )
                        brightness = self.config.data["brightness_background"]
                        color = [c * brightness for c in base_color]
                    else:
                        color = self.config.data["colors"]["off"]
                
                self.launchpad.set_button_led(
                    ButtonType.SEQUENCE,
                    [x, y],
                    color
                )

    def display_pilot_selection(self, pilot_count: int, active_index: t.Optional[int]) -> None:
        """Show pilot selection state across sequence buttons."""
        if not self.launchpad.is_connected:
            return

        pilot_count = max(0, min(pilot_count, 24))

        active_color = self.config.data["colors"]["preset_on"]
        available_color = self.config.data["colors"].get(
            "presets_background",
            self.config.data["colors"]["off"],
        )
        off_color = self.config.data["colors"]["off"]

        for slot in range(24):
            x = slot % 8
            y = slot // 8
            if slot < pilot_count:
                color = active_color if active_index == slot else available_color
            else:
                color = off_color

            self.launchpad.set_button_led(
                ButtonType.SEQUENCE,
                [x, y],
                color,
            )
    
    def flash_success(self, index: t.Tuple[int, int]) -> None:
        """Flash a button green to indicate success."""
        if not self.launchpad.is_connected:
            return
        
        import time
        self.launchpad.set_button_led(
            ButtonType.SEQUENCE,
            [index[0], index[1]],
            self.config.data["colors"]["success_flash"]
        )
        self.launchpad.flush(force=True)
        time.sleep(0.2)
    
    def render_status_frame(
        self,
        background_type: str,
        app_state: "AppState",
        playback_state: "PlaybackState",
        active_sequence_index: t.Optional[t.Tuple[int, int]],
        sequence_steps: t.Optional[list],
        has_active_scenes: bool,
        pilot_running: bool,
        active_page: int,
        key_bindings: dict,
    ) -> None:
        """Render one frame of status LEDs (background, playback, controls)."""
        if not self.launchpad.is_connected:
            return

        # Background
        self.update_background(background_type, app_state)

        # Playback toggle LED
        playback_coords = tuple(key_bindings["playback_toggle_button"]["coordinates"])
        playback_color = "playback_playing" if playback_state == PlaybackState.PLAYING else "playback_paused"
        self.update_control_led(playback_coords, playback_color)

        # Next-step LED
        next_step_coords = tuple(key_bindings["next_step_button"]["coordinates"])
        can_advance = (
            playback_state == PlaybackState.PAUSED
            and sequence_steps is not None
            and len(sequence_steps) > 1
        )
        next_color = "next_step" if can_advance else "off"
        self.update_control_led(next_step_coords, next_color)

        # Pilot toggle LED
        pilot_coords = tuple(key_bindings["pilot_toggle_button"]["coordinates"])
        pilot_color = "pilot_toggle_on" if pilot_running else "pilot_toggle_off"
        self.update_control_led(pilot_coords, pilot_color)

        # Clear button
        clear_coords = tuple(key_bindings["clear_button"]["coordinates"])
        clear_color = "success_flash" if has_active_scenes else "off"
        self.update_control_led(clear_coords, clear_color)

        # Align-to-beat button: always glow pink to make it easy to find
        if "align_to_beat_button" in key_bindings:
            align_coords = tuple(key_bindings["align_to_beat_button"]["coordinates"])
            # Use the preset_on color (pink) as a constant glow
            self.update_control_led(align_coords, "preset_on")

        # Page buttons
        page_buttons = ["page_1_button", "page_2_button"]
        for page_idx, page_key in enumerate(page_buttons):
            if page_key in key_bindings:
                page_coords = tuple(key_bindings[page_key]["coordinates"])
                page_color = "page_active" if page_idx == active_page else "off"
                self.update_control_led(page_coords, page_color)

    def _get_scene_color(self, scene: t.Tuple[int, int], active: bool, page: int = 0) -> t.List[float]:
        """Get color for a scene LED based on the page it belongs to."""
        if not active:
            return self.config.data["colors"]["off"]
        
        # Use column color if configured, selecting the page-specific palette
        if self.config.data.get("scene_on_color_from_column", False):
            colors_key = "column_colors" if page == 0 else "column_colors_page_2"
            column_color = self.config.data["colors"].get(colors_key, {}).get(
                str(scene[0])
            )
            if column_color:
                return column_color
        
        return self.config.data["colors"]["scene_on"]
    
