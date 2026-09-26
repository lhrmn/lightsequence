"""
Light Controller - Slim Orchestrator

Coordinates all subsystems:
- SequenceController: Manages sequences and playback
- SceneController: Handles scene activation
- InputHandler: Processes button events
- LEDController: Updates LEDs
- DeviceManager: Tracks device states
- DeviceMonitor: Auto-reconnects devices
"""

import logging
import time
import typing as t
from typing import TYPE_CHECKING

from lumiblox.controller.sequence_controller import (
    SequenceController,
    SequenceStep,
    PlaybackState,
)
from lumiblox.controller.scene_controller import SceneController
from lumiblox.common.enums import ButtonType
from lumiblox.controller.input_handler import InputHandler, ButtonEvent
from lumiblox.controller.led_controller import LEDController
from lumiblox.controller.device_monitor import DeviceMonitor
from lumiblox.controller.background_animator import BackgroundAnimator, BackgroundManager
from lumiblox.controller.command_queue import CommandQueue, CommandType, ControllerCommand
from lumiblox.controller.app_state_manager import AppStateManager
from lumiblox.devices.launchpad import LaunchpadMK2
from lumiblox.midi.light_software import LightSoftware
from lumiblox.midi.light_software_sim import LightSoftwareSim
from lumiblox.midi.light_software_protocol import LightSoftwareProtocol
from lumiblox.common.device_state import DeviceManager, DeviceType
from lumiblox.common.config import get_config
from lumiblox.common.constants import ROWS_PER_PAGE, NUM_SCENE_PAGES, SCENE_COLUMNS
from lumiblox.common.enums import AppState
from lumiblox.common.project_data_repository import ProjectDataRepository
from lumiblox.midi.midi_manager import midi_manager

if TYPE_CHECKING:
    from lumiblox.pilot.pilot_controller import PilotController

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class LightController:
    """
    Main orchestrator - coordinates all subsystems.
    
    """
    
    def __init__(self, simulation: bool = False):
        """Initialize the light controller."""
        # Load config first (needed by other components)
        self.config = get_config()
        
        # Device management
        self.device_manager = DeviceManager()
        self.device_monitor = DeviceMonitor(self.device_manager)
        
        # Initialize project data repository
        self.project_repo = ProjectDataRepository()
        
        # Core controllers
        self.sequence_ctrl = SequenceController(self.project_repo)
        self.scene_ctrl = SceneController()
        self.input_handler = InputHandler()
        
        # Hardware
        self.light_software: LightSoftwareProtocol
        if simulation:
            logger.info("Using simulated lighting software")
            self.light_software = LightSoftwareSim(config=self.config)
        else:
            self.light_software = LightSoftware(
                device_manager=self.device_manager, config=self.config
            )
        
        self.launchpad = LaunchpadMK2(device_manager=self.device_manager)
        self._animator = BackgroundAnimator()
        self.led_ctrl = LEDController(self.launchpad, self._animator)
        self.background_mgr = BackgroundManager()
        
        # Command queue for thread-safe GUI -> controller communication
        self.command_queue = CommandQueue()

        # App state
        self.active_sequence: t.Optional[t.Tuple[int, int]] = None
        self._last_sequence_scenes: t.Set[t.Tuple[int, int]] = set()
        self.active_page: int = 0
        self._blink_phase: bool = False
        self._last_blink_toggle: float = 0.0
        self._dual_active_positions: t.Set[t.Tuple[int, int]] = set()
        self._other_page_only_positions: t.Set[t.Tuple[int, int]] = set()
        self._BLINK_INTERVAL: float = 0.35  # seconds between color alternation
        self.pilot_controller: t.Optional["PilotController"] = None

        self.last_manual_sequence_time: float = 0.0

        # App state manager (save modes, pilot select)
        self.app_state_mgr = AppStateManager(
            led_ctrl=self.led_ctrl,
            sequence_ctrl=self.sequence_ctrl,
            project_repo=self.project_repo,
            switch_pilot_fn=self.switch_pilot,
            set_sequence_led_fn=self._set_sequence_led_state,
        )

        # Apply background animation from config (set once at startup)
        self.background_mgr.set_background(
            self.config.data.get("background_animation", "default")
        )

        
        # External callbacks (for GUI)
        self.on_sequence_changed: t.Optional[t.Callable[[t.Optional[t.Tuple[int, int]]], None]] = None
        self.on_sequence_saved: t.Optional[t.Callable[[], None]] = None
        
        # Wire up callbacks
        self._setup_callbacks()
    
    def _setup_callbacks(self) -> None:
        """Connect all the controllers together."""
        # Scene controller -> Light software & LED updates
        self.scene_ctrl.on_scene_activate = self._handle_scene_activate
        self.scene_ctrl.on_scene_deactivate = self._handle_scene_deactivate
        
        # Sequence controller -> Scene controller
        self.sequence_ctrl.on_step_change = self._handle_step_change
        self.sequence_ctrl.on_sequence_complete = self._handle_sequence_complete
        
        # Input handler -> Action handlers
        self.input_handler.on_scene_button = self._handle_scene_button
        self.input_handler.on_sequence_button = self._handle_sequence_button
        self.input_handler.on_control_button = self._handle_control_button
        
        # Register control buttons from config
        self._register_control_buttons()
        
        # Device monitor
        self.device_monitor.register_reconnect_callback(
            DeviceType.LAUNCHPAD, self.launchpad.connect
        )
        self.device_monitor.register_reconnect_callback(
            DeviceType.LIGHT_SOFTWARE, self.light_software.connect_midi
        )
        self.device_manager.register_state_change_callback(
            self._on_device_state_changed
        )
    
    def _register_control_buttons(self) -> None:
        """Register control buttons from configuration."""
        bindings = self.config.data.get("key_bindings", {})
        allowed_controls = {
            "save_button",
            "save_shift_button",
            "playback_toggle_button",
            "next_step_button",
            "clear_button",
            "pilot_select_button",
            "align_to_beat_button",
            "pilot_toggle_button",
            "page_1_button",
            "page_2_button",
        }
        for name, binding in bindings.items():
            if name not in allowed_controls:
                logger.debug("Skipping unsupported control binding: %s", name)
                continue
            coords = tuple(binding["coordinates"])
            self.input_handler.register_control_button(name, coords)
    
    # ============================================================================
    # LIFECYCLE
    # ============================================================================
    
    def initialize(self) -> bool:
        """Initialize hardware connections (non-blocking)."""
        # Try to connect light software
        self.light_software.connect_midi()
        
        # Start device monitor
        self.device_monitor.start()
        
        # Sync initial state from light software
        self._sync_initial_scenes()
        
        # Show scene LEDs for the initial page
        self._refresh_scene_leds_for_page()
        
        logger.info("Light controller initialized")
        return True
    
    def run_main_loop(self) -> None:
        """Main control loop."""
        self.initialize()
        
        logger.info("Light controller running. Press Ctrl+C to exit.")
        
        try:
            while True:
                # Process queued commands from GUI
                self._process_commands()

                # Process inputs
                self._process_launchpad_input()
                self._process_midi_feedback()
                
                # Update outputs
                self._update_leds()
                
                time.sleep(0.02)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.device_monitor.stop()
        self.sequence_ctrl.cleanup()
        self.launchpad.close()
        self.light_software.close()
        # Final MIDI shutdown — close all tracked ports at application exit
        midi_manager.shutdown()
        logger.info("Cleanup complete")
    
    def switch_pilot(self, pilot_index: int) -> bool:
        """
        Switch to a different pilot and reload its sequences.
        
        Args:
            pilot_index: Index of the pilot to switch to
            
        Returns:
            True if successful
        """
        if self.project_repo.set_active_pilot(pilot_index):
            # Stop any active sequence playback
            self.sequence_ctrl.stop_playback()
            
            # Reload sequences from the new active pilot
            self.sequence_ctrl.load_from_repository()
            
            logger.info(f"Switched to pilot at index {pilot_index}")
            return True
        return False
    
    # ============================================================================
    # INPUT PROCESSING
    # ============================================================================
    
    def _process_launchpad_input(self) -> None:
        """Process button events from launchpad."""
        if not self.launchpad.is_connected:
            return
        
        button_data = self.launchpad.get_button_events()
        if button_data:
            # Convert to generic ButtonEvent
            event = self._convert_launchpad_event(button_data)
            if event:
                self.input_handler.handle_button_event(event)
    
    def _convert_launchpad_event(self, data: t.Dict) -> t.Optional[ButtonEvent]:
        """Convert launchpad event to generic ButtonEvent."""
        lp_type = data["type"]
        coords = tuple(data["index"])
        pressed = data["active"]
        
        # Map launchpad button type to generic type
        if lp_type == ButtonType.SCENE:
            btn_type = ButtonType.SCENE
            # Apply page offset to get absolute scene coordinates
            coords = (coords[0], coords[1] + self.active_page * ROWS_PER_PAGE)
        elif lp_type == ButtonType.SEQUENCE:
            btn_type = ButtonType.SEQUENCE
        elif lp_type == ButtonType.CONTROL:
            btn_type = ButtonType.CONTROL
        else:
            btn_type = ButtonType.UNKNOWN
        
        return ButtonEvent(btn_type, coords, pressed, source="launchpad")
    
    def _process_midi_feedback(self) -> None:
        """Process MIDI feedback from light software."""
        if not self.light_software.connection_good:
            return
        
        changes = self.light_software.process_feedback()
        for scene, is_active in changes.items():
            if scene:
                guarded = self.scene_ctrl.get_sequence_guard_scenes()
                lp_scene = self._scene_to_launchpad_scene(scene)
                # For guarded scenes, controller is source of truth; only allow offs
                if scene in guarded:
                    if is_active:
                        continue
                    self.scene_ctrl.mark_scene_active(scene, False)
                    if lp_scene is not None:
                        self.led_ctrl.update_scene_led(lp_scene, False, page=self.active_page)
                    continue

                self.scene_ctrl.mark_scene_active(scene, is_active)
                if lp_scene is not None:
                    self.led_ctrl.update_scene_led(lp_scene, is_active, page=self.active_page)
    
    def _sync_initial_scenes(self) -> None:
        """Sync initial active scenes from light software."""
        if hasattr(self.light_software, 'process_feedback'):
            changes = self.light_software.process_feedback()
            for scene, is_active in changes.items():
                if is_active and scene:
                    self.scene_ctrl.mark_scene_active(scene, True)
    
    # ============================================================================
    # EVENT HANDLERS
    # ============================================================================
    
    def _handle_scene_button(self, scene: t.Tuple[int, int], pressed: bool) -> None:
        """Handle scene button press."""
        if pressed:
            is_active = self.scene_ctrl.toggle_scene(scene)
            logger.debug(f"Scene {scene} toggled: {is_active}")
    
    def _handle_sequence_button(self, index: t.Tuple[int, int], pressed: bool) -> None:
        """Handle sequence button press (based on app state)."""
        if pressed:
            self.last_manual_sequence_time = time.time()
            
        state = self.app_state_mgr.state
        if state == AppState.SAVE_MODE:
            self._save_sequence(index)
        elif state == AppState.SAVE_SHIFT_MODE:
            self._add_step_to_sequence(index)
        elif state == AppState.PILOT_SELECT_MODE:
            if pressed:
                self.app_state_mgr.handle_pilot_selection_button(index)
        else:
            self._activate_deactivate_sequence(index)
    
    def _handle_control_button(self, name: str, pressed: bool) -> None:
        """Handle control button press."""
        if not pressed:
            return
        
        handlers = {
            "save_button": self.app_state_mgr.toggle_save_mode,
            "save_shift_button": self.app_state_mgr.toggle_save_shift_mode,
            "playback_toggle_button": self._toggle_playback,
            "next_step_button": self._next_step,
            "clear_button": self._clear_all_scenes,
            "pilot_select_button": self.app_state_mgr.toggle_pilot_select_mode,
            "align_to_beat_button": self._align_to_beat,
            "pilot_toggle_button": self._toggle_pilot_automation,
            "page_1_button": lambda: self._switch_page(0),
            "page_2_button": lambda: self._switch_page(1),
        }
        
        handler = handlers.get(name)
        if handler:
            handler()
    
    # ============================================================================
    # SCENE CALLBACKS
    # ============================================================================
    
    def _handle_scene_activate(self, scene: t.Tuple[int, int]) -> None:
        """Handle scene activation."""
        logger.debug("scene_activate scene=%s", scene)
        # Use explicit state setting for deterministic control
        self.light_software.set_scene_state(scene, True)
        lp_scene = self._scene_to_launchpad_scene(scene)
        if lp_scene is not None:
            self.led_ctrl.update_scene_led(lp_scene, True, page=self.active_page)
    
    def _handle_scene_deactivate(self, scene: t.Tuple[int, int]) -> None:
        """Handle scene deactivation."""
        logger.debug("scene_deactivate scene=%s", scene)
        # Send explicit velocity 0 so offs are deterministic even if a toggle is missed.
        self.light_software.set_scene_state(scene, False)
        lp_scene = self._scene_to_launchpad_scene(scene)
        if lp_scene is not None:
            self.led_ctrl.update_scene_led(lp_scene, False, page=self.active_page)
    
    # ============================================================================
    # SEQUENCE CALLBACKS
    # ============================================================================
    
    def _handle_step_change(self, scenes: t.List[t.Tuple[int, int]]) -> None:
        """Handle sequence step change."""
        logger.debug(
            "step_change scenes=%s active_sequence=%s", scenes, self.sequence_ctrl.active_sequence
        )
        current_sequence = self.sequence_ctrl.active_sequence
        if current_sequence is None:
            logger.debug("Ignoring step change with no active sequence")
            return
        self._last_sequence_scenes = set(scenes)
        if current_sequence != self.active_sequence:
            if self.active_sequence:
                self._set_sequence_led_state(self.active_sequence, False)
            self.active_sequence = current_sequence
            if self.active_sequence:
                self._set_sequence_led_state(self.active_sequence, True)
            if self.on_sequence_changed:
                self.on_sequence_changed(self.active_sequence)
        self.scene_ctrl.activate_scenes(scenes, controlled=True)

    
    def _handle_sequence_complete(self) -> None:
        """Handle sequence completion."""
        logger.info("Sequence completed")
    
    # ============================================================================
    # SEQUENCE ACTIONS
    # ============================================================================
    
    def _activate_deactivate_sequence(self, index: t.Tuple[int, int], force_activate: bool = False) -> None:
        """Activate or deactivate a sequence."""
        if self.active_sequence == index and not force_activate:
            # Deactivate
            last_scenes = self._last_sequence_scenes.copy()
            self.sequence_ctrl.clear()
            self.scene_ctrl.clear_controlled()
            if last_scenes:
                self.scene_ctrl.force_deactivate_scenes(last_scenes)
            self._last_sequence_scenes.clear()
            self.active_sequence = None
            self._set_sequence_led_state(index, False)
            if self.on_sequence_changed:
                self.on_sequence_changed(None)
        else:
            # Deactivate old sequence first
            if self.active_sequence:
                self._set_sequence_led_state(self.active_sequence, False)
                last_scenes = self._last_sequence_scenes.copy()
                next_sequence = self.sequence_ctrl.get_sequence(index)
                next_first_scenes = (
                    set(next_sequence[0].scenes) if next_sequence else set()
                )
                scenes_to_force_off = last_scenes - next_first_scenes
                if scenes_to_force_off:
                    self.scene_ctrl.force_deactivate_scenes(scenes_to_force_off)
            
            # Activate new sequence
            self.active_sequence = index
            self.sequence_ctrl.activate_sequence(index)
            self._set_sequence_led_state(index, True)
            if self.on_sequence_changed:
                self.on_sequence_changed(index)
    
    def _save_sequence(self, index: t.Tuple[int, int]) -> None:
        """Save current scenes as a sequence."""
        scenes = list(self.scene_ctrl.get_active_scenes())
        step = SequenceStep(scenes=scenes, duration=1.0, name="Step 1")
        self.sequence_ctrl.save_sequence(index, [step], loop=True)
        
        logger.info(f"Saved {len(scenes)} scenes to sequence {index}")
        if self.on_sequence_saved:
            self.on_sequence_saved()
        
        self.app_state_mgr.exit_save_mode()
        self._activate_deactivate_sequence(index)
    
    def _add_step_to_sequence(self, index: t.Tuple[int, int]) -> None:
        """Add current scenes as new step to sequence."""
        scenes = list(self.scene_ctrl.get_active_scenes())
        existing = self.sequence_ctrl.get_sequence(index)
        
        if existing:
            new_step = SequenceStep(scenes=scenes, duration=1.0, name=f"Step {len(existing)+1}")
            existing.append(new_step)
            self.sequence_ctrl.save_sequence(index, existing, loop=True)
        else:
            step = SequenceStep(scenes=scenes, duration=1.0, name="Step 1")
            self.sequence_ctrl.save_sequence(index, [step], loop=True)
        
        self.led_ctrl.flash_success(index)
        if self.on_sequence_saved:
            self.on_sequence_saved()
        
        self.app_state_mgr.exit_save_mode()
        self._activate_deactivate_sequence(index)
    
    def _set_sequence_led_state(self, index: t.Tuple[int, int], active: bool) -> None:
        """Update a sequence LED unless overridden by pilot mode."""
        if self.app_state_mgr.state == AppState.PILOT_SELECT_MODE:
            return
        self.led_ctrl.update_sequence_led(index, active)
    
    # ============================================================================
    # PAGE MANAGEMENT
    # ============================================================================
    
    def _scene_to_launchpad_scene(self, scene: t.Tuple[int, int]) -> t.Optional[t.Tuple[int, int]]:
        """Convert absolute scene coords to Launchpad scene coords.
        
        Returns None if the scene is not on the currently active page.
        """
        x, y = scene
        page_start = self.active_page * ROWS_PER_PAGE
        page_end = page_start + ROWS_PER_PAGE
        if page_start <= y < page_end:
            return (x, y - page_start)
        return None
    
    def _switch_page(self, page: int) -> None:
        """Switch to a different scene page on the Launchpad."""
        if page == self.active_page or page < 0 or page >= NUM_SCENE_PAGES:
            return
        
        logger.info("Switching scene page %d -> %d", self.active_page + 1, page + 1)
        self.active_page = page
        self._refresh_scene_leds_for_page()
    
    def _refresh_scene_leds_for_page(self) -> None:
        """Refresh all scene LEDs on the Launchpad for the current page."""
        active_scenes = self.scene_ctrl.get_active_scenes()
        other_page = 1 - self.active_page
        self._dual_active_positions = set()
        self._other_page_only_positions = set()
        for lp_y in range(ROWS_PER_PAGE):
            for lp_x in range(SCENE_COLUMNS):
                page0_scene = (lp_x, lp_y)
                page1_scene = (lp_x, lp_y + ROWS_PER_PAGE)
                p0_active = page0_scene in active_scenes
                p1_active = page1_scene in active_scenes
                current_active = (lp_x, lp_y + self.active_page * ROWS_PER_PAGE) in active_scenes
                other_active = (lp_x, lp_y + other_page * ROWS_PER_PAGE) in active_scenes
                if p0_active and p1_active:
                    # Both pages active — register for blink
                    self._dual_active_positions.add((lp_x, lp_y))
                    self.led_ctrl.update_scene_led((lp_x, lp_y), True, page=self.active_page)
                elif other_active and not current_active:
                    # Only the other page is active — show dim hint
                    self._other_page_only_positions.add((lp_x, lp_y))
                    self.led_ctrl.update_scene_led_other_page((lp_x, lp_y), other_page)
                else:
                    self.led_ctrl.update_scene_led((lp_x, lp_y), current_active, page=self.active_page)
    
    def _update_blinking_scene_leds(self) -> None:
        """Alternate colors for Launchpad positions active on both pages.
        
        Also maintains dim hints for positions only active on the other page.
        """
        active_scenes = self.scene_ctrl.get_active_scenes()
        other_page = 1 - self.active_page
        new_dual_active: t.Set[t.Tuple[int, int]] = set()
        new_other_only: t.Set[t.Tuple[int, int]] = set()

        for lp_y in range(ROWS_PER_PAGE):
            for lp_x in range(SCENE_COLUMNS):
                page0_scene = (lp_x, lp_y)
                page1_scene = (lp_x, lp_y + ROWS_PER_PAGE)
                p0_active = page0_scene in active_scenes
                p1_active = page1_scene in active_scenes
                current_active = (lp_x, lp_y + self.active_page * ROWS_PER_PAGE) in active_scenes
                other_active = (lp_x, lp_y + other_page * ROWS_PER_PAGE) in active_scenes

                if p0_active and p1_active:
                    new_dual_active.add((lp_x, lp_y))
                    show_page = 0 if self._blink_phase else 1
                    self.led_ctrl.update_scene_led((lp_x, lp_y), True, page=show_page)
                elif other_active and not current_active:
                    new_other_only.add((lp_x, lp_y))
                    # Only push dim color when position was not already tracked
                    if (lp_x, lp_y) not in self._other_page_only_positions:
                        self.led_ctrl.update_scene_led_other_page((lp_x, lp_y), other_page)

        # Positions that stopped being dual-active or other-only: restore
        changed = (self._dual_active_positions | self._other_page_only_positions) - (new_dual_active | new_other_only)
        for lp_x, lp_y in changed:
            current_scene = (lp_x, lp_y + self.active_page * ROWS_PER_PAGE)
            is_active = current_scene in active_scenes
            self.led_ctrl.update_scene_led(
                (lp_x, lp_y), is_active, page=self.active_page
            )

        self._dual_active_positions = new_dual_active
        self._other_page_only_positions = new_other_only

    # ============================================================================
    # OTHER CONTROLS
    # ============================================================================
    
    def _toggle_playback(self) -> None:
        """Toggle sequence playback."""
        if self.sequence_ctrl.playback_state == PlaybackState.PLAYING:
            self.sequence_ctrl.pause()
        else:
            self.sequence_ctrl.play()
    
    def _next_step(self) -> None:
        """Advance to next step."""
        self.sequence_ctrl.next_step()
    
    def _clear_all_scenes(self) -> None:
        """Clear all active scenes."""
        if self.active_sequence:
            self.sequence_ctrl.stop_playback()
            self._set_sequence_led_state(self.active_sequence, False)
            self.active_sequence = None
        self._last_sequence_scenes.clear()
        
        self.scene_ctrl.clear_all()

    def _toggle_pilot_automation(self) -> None:
        """Toggle pilot automation execution on/off and persist configuration (leaves tracking running)."""
        if not self.pilot_controller:
            logger.warning("Pilot controller not attached; cannot toggle automation")
            return
            
        current_paused = self.pilot_controller.automation_paused
        new_paused = not current_paused
        
        self.pilot_controller.automation_paused = new_paused
        self.config.set_pilot_automation_paused(new_paused)
        
        # If unpausing and the engine isn't fully running, try to start it
        if not new_paused and not self.pilot_controller.is_running():
            try:
                self.pilot_controller.ensure_running()
            except Exception as exc:
                logger.warning("Failed to start pilot controller from automation toggle: %s", exc)
        
        state_str = "paused" if new_paused else "armed"
        logger.info(f"Pilot automation {state_str} via hardware button")

    def _align_to_beat(self) -> None:
        """Trigger manual alignment to beat via the PilotController."""
        if not self.pilot_controller:
            logger.warning("Pilot controller not attached; cannot align to beat")
            return
        try:
            if not self.pilot_controller.ensure_running():
                logger.warning("Pilot controller could not be started for alignment")
                return
            self.pilot_controller.align_to_beat()
            logger.info("Pilot align-to-beat triggered")
        except Exception as exc:
            logger.warning("Failed to trigger align-to-beat: %s", exc)
    
    def _update_leds(self) -> None:
        """Update all LED displays."""
        # Gather read-only state for LED rendering
        pilot_running = False
        if self.pilot_controller:
            try:
                pilot_running = self.pilot_controller.is_running() and not self.pilot_controller.automation_paused
            except Exception:
                pass
        else:
            pilot_running = bool(self.config.data.get("pilot", {}).get("enabled", False)) and not self.config.pilot_automation_paused

        active_index = self.sequence_ctrl.active_sequence
        active_steps = self.sequence_ctrl.get_sequence(active_index) if active_index else None

        self.led_ctrl.render_status_frame(
            background_type=self.background_mgr.get_current_background(),
            app_state=self.app_state_mgr.state,
            playback_state=self.sequence_ctrl.playback_state,
            active_sequence_index=active_index,
            sequence_steps=active_steps,
            has_active_scenes=self.scene_ctrl.has_active_scenes(),
            pilot_running=pilot_running,
            active_page=self.active_page,
            key_bindings=self.config.data.get("key_bindings", {}),
        )

        # Blink dual-active scene LEDs (active on both pages)
        if self.launchpad.is_connected:
            now = time.time()
            if now - self._last_blink_toggle >= self._BLINK_INTERVAL:
                self._blink_phase = not self._blink_phase
                self._last_blink_toggle = now
                self._update_blinking_scene_leds()

        # Send all LED changes of this frame to the device in one batch
        self.launchpad.flush()
    
    def _on_device_state_changed(self, device_type, new_state) -> None:
        """Handle device state changes."""
        logger.debug(f"Device {device_type.value} -> {new_state.value}")
    
    # ============================================================================
    # EXTERNAL INTERFACE (for GUI)
    # ============================================================================
    
    def process_button_event_from_external(self, event: t.Dict) -> None:
        """Process button event from external source (GUI)."""
        # Convert dict to ButtonEvent if needed
        if isinstance(event, dict):
            evt = ButtonEvent(
                button_type=ButtonType(event.get("type", "unknown")),
                coordinates=tuple(event.get("index", (0, 0))),
                pressed=event.get("active", False),
                source="external"
            )
            self.input_handler.handle_button_event(evt)
    
    def process_midi_feedback_from_external(self) -> None:
        """Process MIDI feedback (for GUI compatibility)."""
        self._process_midi_feedback()

    def set_pilot_controller(self, pilot_controller: "PilotController") -> None:
        """Attach a pilot controller for preset synchronization."""
        self.pilot_controller = pilot_controller
        self.app_state_mgr.pilot_controller = pilot_controller
        # Sync automation paused state from config
        self.pilot_controller.automation_paused = self.config.pilot_automation_paused

    # ============================================================================
    # COMMAND QUEUE (thread-safe GUI -> controller communication)
    # ============================================================================

    def _process_commands(self) -> None:
        """Drain and handle all pending commands from the queue."""
        self.command_queue.process_all(self._handle_command)

    def _handle_command(self, cmd: ControllerCommand) -> None:
        """Dispatch a single controller command."""
        try:
            if cmd.command_type == CommandType.TOGGLE_PLAYBACK:
                self._toggle_playback()
            elif cmd.command_type == CommandType.NEXT_STEP:
                self._next_step()
            elif cmd.command_type == CommandType.CLEAR:
                self._clear_all_scenes()
            elif cmd.command_type == CommandType.ACTIVATE_SEQUENCE:
                if time.time() - self.last_manual_sequence_time < 1.0:
                    logger.debug(f"Ignoring automated sequence {cmd.data['index']} due to recent manual intervention.")
                else:
                    self._activate_deactivate_sequence(cmd.data["index"], force_activate=cmd.data.get("force_activate", False))
            elif cmd.command_type == CommandType.SAVE_SEQUENCE:
                self.sequence_ctrl.save_sequence(
                    cmd.data["index"],
                    cmd.data["steps"],
                    cmd.data.get("loop", True),
                    loop_count=cmd.data.get("loop_count", 1),
                    next_sequences=cmd.data.get("next_sequences"),
                )
            elif cmd.command_type == CommandType.DELETE_SEQUENCE:
                self.sequence_ctrl.delete_sequence(cmd.data["index"])
            elif cmd.command_type == CommandType.SWITCH_PILOT:
                self.switch_pilot(cmd.data["pilot_index"])
            elif cmd.command_type == CommandType.ACTIVATE_SCENES:
                self.scene_ctrl.activate_scenes(
                    cmd.data["scenes"],
                    controlled=cmd.data.get("controlled", True),
                )
            elif cmd.command_type == CommandType.BUTTON_EVENT:
                event = ButtonEvent(
                    button_type=ButtonType(cmd.data["type"]),
                    coordinates=tuple(cmd.data["coordinates"]),
                    pressed=cmd.data["pressed"],
                    source="gui",
                )
                self.input_handler.handle_button_event(event)
            else:
                logger.warning("Unknown command type: %s", cmd.command_type)
        except Exception as exc:
            logger.error("Error handling command %s: %s", cmd.command_type, exc)

    # --- Public post helpers (call from any thread) ---

    def post_toggle_playback(self) -> None:
        self.command_queue.post(ControllerCommand(CommandType.TOGGLE_PLAYBACK))

    def post_next_step(self) -> None:
        self.command_queue.post(ControllerCommand(CommandType.NEXT_STEP))

    def post_clear(self) -> None:
        self.command_queue.post(ControllerCommand(CommandType.CLEAR))

    def post_activate_sequence(self, index: t.Tuple[int, int], force_activate: bool = False) -> None:
        self.command_queue.post(ControllerCommand(CommandType.ACTIVATE_SEQUENCE, {"index": index, "force_activate": force_activate}))

    def post_save_sequence(
        self,
        index: t.Tuple[int, int],
        steps: list,
        loop: bool = True,
        loop_count: int = 1,
        next_sequences: t.Optional[list] = None,
    ) -> None:
        self.command_queue.post(ControllerCommand(CommandType.SAVE_SEQUENCE, {
            "index": index,
            "steps": steps,
            "loop": loop,
            "loop_count": loop_count,
            "next_sequences": next_sequences,
        }))

    def post_delete_sequence(self, index: t.Tuple[int, int]) -> None:
        self.command_queue.post(ControllerCommand(CommandType.DELETE_SEQUENCE, {"index": index}))

    def post_switch_pilot(self, pilot_index: int) -> None:
        self.command_queue.post(ControllerCommand(CommandType.SWITCH_PILOT, {"pilot_index": pilot_index}))

    def post_activate_scenes(self, scenes: list, controlled: bool = True) -> None:
        self.command_queue.post(ControllerCommand(CommandType.ACTIVATE_SCENES, {
            "scenes": scenes,
            "controlled": controlled,
        }))

    def post_button_event(self, button_type: str, coordinates: t.Tuple[int, int], pressed: bool) -> None:
        self.command_queue.post(ControllerCommand(CommandType.BUTTON_EVENT, {
            "type": button_type,
            "coordinates": coordinates,
            "pressed": pressed,
        }))

    # --- Read-only snapshot helpers (safe from GUI thread) ---

    def get_sequence_indices(self) -> t.Set[t.Tuple[int, int]]:
        return self.sequence_ctrl.get_all_indices()

    def get_sequence(self, index: t.Tuple[int, int]):
        return self.sequence_ctrl.get_sequence(index)

    def get_playback_state(self):
        return self.sequence_ctrl.playback_state

    def get_current_step_index(self) -> int:
        return self.sequence_ctrl.current_step_index

    def get_active_sequence(self):
        return self.active_sequence

    def get_active_scenes(self):
        return self.scene_ctrl.get_active_scenes()

    def get_loop_setting(self, index):
        return self.sequence_ctrl.get_loop_setting(index)

    def get_loop_count(self, index):
        return self.sequence_ctrl.get_loop_count(index)

    def get_followup_sequences(self, index):
        return self.sequence_ctrl.get_followup_sequences(index)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main(simulation: bool = False):
    """Main entry point."""
    controller = LightController(simulation=simulation)
    controller.run_main_loop()


if __name__ == "__main__":
    main()
