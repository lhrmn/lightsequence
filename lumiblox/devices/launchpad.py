"""
Novation Launchpad MK2 driver (mido + python-rtmidi).

LED writes are queued by ``set_led`` and sent in batched, rate-limited SysEx
messages by ``flush``. Input is polled non-blocking. A vanished MIDI port marks
the device disconnected so ``DeviceMonitor`` can reconnect it via ``connect``.
"""

import logging
import threading
import time
import typing as t

import mido
import numpy as np
from lumiblox.common.config import get_config
from lumiblox.common.enums import ButtonType
from lumiblox.common.utils import hex_to_rgb
from lumiblox.common.device_state import DeviceManager, DeviceType
from lumiblox.midi.midi_manager import midi_manager

logger = logging.getLogger(__name__)

# SysEx header for the Launchpad MK2 (Novation manufacturer ID + device ID 0x18)
SYSEX_HEADER = [0x00, 0x20, 0x29, 0x02, 0x18]
SYSEX_SET_LED_RGB = 0x0B
SYSEX_SET_ALL_LEDS = 0x0E
SYSEX_SELECT_LAYOUT = 0x22
LAYOUT_SESSION = 0x00

# Top row round buttons are CC 104-111; grid and right column are notes 11-89
TOP_ROW_CC_START = 104


class LaunchpadMK2:
    PORT_KEYWORD = "Launchpad MK2"
    # Max LED updates sent to the device per second. Without a cap the controller
    # loop (~1ms) floods CoreMIDI on macOS and freezes the Launchpad firmware.
    MAX_FLUSH_RATE_HZ = 60
    # Mk2 accepts up to 80 LEDs in one "set LED RGB" SysEx message
    MAX_LEDS_PER_SYSEX = 80
    # How often to check that the MIDI port still exists (unplug detection)
    PORT_CHECK_INTERVAL = 2.0

    def __init__(self, device_manager: t.Optional[DeviceManager] = None):
        self.BOUNDS_SCENES = np.array([[0, 1], [8, 5]])
        self.BOUNDS_PRESETS = np.array([[0, 6], [7, 8]])
        self.BOUNDS_TOP = np.array([[0, 0], [7, 0]])
        self.BOUNDS_RIGHT = np.array([[8, 1], [8, 8]])

        self.pixel_buffer_output = np.zeros(
            (9, 9, 3), dtype=float
        )  # 9x9 grid for Launchpad MK2

        # Track hardware LED state (0-63 range) to avoid unnecessary updates
        self.hardware_led_state = np.zeros((9, 9, 3), dtype=int)

        self.config = get_config()

        # Device state management
        self.device_manager = device_manager
        self.input: t.Optional[mido.ports.BaseInput] = None
        self.output: t.Optional[mido.ports.BaseOutput] = None
        self.is_connected = False

        # LED changes waiting to be sent: led number -> (r, g, b)
        self._pending_leds: t.Dict[int, t.Tuple[int, int, int]] = {}
        self._last_flush = 0.0
        self._last_port_check = 0.0
        # Guards ports and the pending LED queue (connect runs on the monitor thread)
        self._lock = threading.RLock()

        # Attempt initial connection
        self.connect()

    # ------------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Connect to Launchpad MK2 hardware."""
        with self._lock:
            if self.device_manager:
                self.device_manager.set_connecting(DeviceType.LAUNCHPAD)

            self._close_ports()
            self.input = midi_manager.open_input_by_keyword(self.PORT_KEYWORD)
            self.output = midi_manager.open_output_by_keyword(self.PORT_KEYWORD)

            if self.input is None or self.output is None:
                self._close_ports()
                self.is_connected = False
                if self.device_manager:
                    self.device_manager.set_disconnected(DeviceType.LAUNCHPAD)
                logger.warning("Could not open Launchpad MK2")
                return False

            try:
                # Drop stale input from before the connect
                for _ in self.input.iter_pending():
                    pass
                self._send_sysex([SYSEX_SELECT_LAYOUT, LAYOUT_SESSION])
                self._send_sysex([SYSEX_SET_ALL_LEDS, 0])
            except Exception as e:
                self._handle_io_error("Error initializing Launchpad", e)
                return False

            self._pending_leds.clear()
            self.hardware_led_state.fill(0)
            self._last_port_check = time.monotonic()
            self.is_connected = True

        # Restore foreground LEDs lost by the reset (e.g. after a replug)
        self._restore_foreground()

        if self.device_manager:
            self.device_manager.set_connected(DeviceType.LAUNCHPAD)
        logger.info("Connected to Launchpad MK2")
        return True

    def _close_ports(self) -> None:
        midi_manager.close_port(self.input)
        midi_manager.close_port(self.output)
        self.input = None
        self.output = None

    def _handle_io_error(self, context: str, error: Exception) -> None:
        """Mark the device as failed so DeviceMonitor reconnects it."""
        logger.error(f"{context}: {error}")
        with self._lock:
            self.is_connected = False
            self._pending_leds.clear()
            self._close_ports()
        if self.device_manager:
            self.device_manager.set_error(DeviceType.LAUNCHPAD, f"{context}: {error}")

    def _check_port_present(self) -> bool:
        """Periodically verify the port still exists; rtmidi does not report unplugs."""
        now = time.monotonic()
        if now - self._last_port_check < self.PORT_CHECK_INTERVAL:
            return True
        self._last_port_check = now

        if midi_manager.find_port_name(self.PORT_KEYWORD, midi_manager.get_input_names()):
            return True

        logger.warning("Launchpad MK2 MIDI port disappeared")
        with self._lock:
            self.is_connected = False
            self._pending_leds.clear()
            self._close_ports()
        if self.device_manager:
            self.device_manager.set_disconnected(DeviceType.LAUNCHPAD)
        return False

    def _restore_foreground(self) -> None:
        brightness = self.config.data["brightness_foreground"]
        for x in range(9):
            for y in range(9):
                color = self.pixel_buffer_output[x, y]
                if color.any():
                    self.set_led(x, y, [c * brightness for c in color])

    # ------------------------------------------------------------------
    # LED output
    # ------------------------------------------------------------------

    def _send_sysex(self, data: t.List[int]) -> None:
        if self.output is not None:
            self.output.send(mido.Message("sysex", data=SYSEX_HEADER + data))

    def _reset_leds(self) -> None:
        """Turn off all LEDs."""
        with self._lock:
            self._pending_leds.clear()
            self._send_sysex([SYSEX_SET_ALL_LEDS, 0])

    def flush(self, force: bool = False) -> None:
        """Send pending LED changes in batched SysEx messages, rate-limited."""
        if not self.is_connected:
            return

        now = time.monotonic()
        if not force and now - self._last_flush < 1.0 / self.MAX_FLUSH_RATE_HZ:
            return

        try:
            with self._lock:
                if not self._pending_leds:
                    return
                items = list(self._pending_leds.items())
                self._pending_leds.clear()
                self._last_flush = now

                for i in range(0, len(items), self.MAX_LEDS_PER_SYSEX):
                    data = [SYSEX_SET_LED_RGB]
                    for led, rgb in items[i : i + self.MAX_LEDS_PER_SYSEX]:
                        data.extend((led, *rgb))
                    self._send_sysex(data)
        except Exception as e:
            self._handle_io_error("LED write error", e)

    def set_led(self, x: int, y: int, color: t.List[float]) -> None:
        """Set LED at absolute (x, y) coordinates to specified color. rgb values float 0-1.0"""
        if not self.is_connected:
            return

        max_x, max_y = self.pixel_buffer_output.shape[0], self.pixel_buffer_output.shape[1]
        if not (0 <= x < max_x and 0 <= y < max_y):
            logger.warning("LED coordinates out of bounds: (%s, %s)", x, y)
            return

        # Scale to 0-63 (Launchpad MK2 range)
        color_scaled = [max(0, min(63, int(c * 63))) for c in color]

        # Check if the hardware state already matches the desired color
        if np.array_equal(self.hardware_led_state[x, y], color_scaled):
            return  # Skip update - LED is already at the correct color

        # Queue for the next flush() and track the new state
        led = TOP_ROW_CC_START + x if y == 0 else 91 - (10 * y) + x
        with self._lock:
            self._pending_leds[led] = (color_scaled[0], color_scaled[1], color_scaled[2])
        self.hardware_led_state[x, y] = color_scaled

    def set_button_led(
        self,
        button_type: ButtonType,
        relative_coords: t.List[int],
        color: t.List[float] | str,
    ) -> None:
        """Set LED using button type and relative coordinates."""
        if not self.is_connected:
            return

        if len(relative_coords) < 2:
            logger.warning(f"Invalid coordinates {relative_coords}")
            return

        abs_x, abs_y = self._relative_to_absolute_coords(button_type, relative_coords)
        if abs_x is None or abs_y is None:
            logger.warning(f"Invalid coordinates for {button_type}: {relative_coords}")
            return

        if isinstance(color, str):
            color = hex_to_rgb(color)

        # multiply with foreground brightness
        brightness = self.config.data["brightness_foreground"]
        adjusted_color = [c * brightness for c in color]

        self.pixel_buffer_output[abs_x, abs_y] = color
        self.set_led(abs_x, abs_y, adjusted_color)

    def _relative_to_absolute_coords(
        self, button_type: ButtonType, relative_coords: t.List[int]
    ) -> t.Tuple[t.Optional[int], t.Optional[int]]:
        """Convert relative coordinates to absolute coordinates based on button type."""
        rel_x, rel_y = relative_coords[0], relative_coords[1]

        if button_type == ButtonType.SCENE:
            abs_x = self.BOUNDS_SCENES[0][0] + rel_x
            abs_y = self.BOUNDS_SCENES[0][1] + rel_y
            return abs_x, abs_y

        if button_type == ButtonType.SEQUENCE:
            abs_x = self.BOUNDS_PRESETS[0][0] + rel_x
            abs_y = self.BOUNDS_PRESETS[0][1] + rel_y
            return abs_x, abs_y

        if button_type == ButtonType.CONTROL:
            return rel_x, rel_y

        return None, None

    def clear_leds(self) -> None:
        """Clear all LEDs on the Launchpad."""
        if not self.is_connected:
            return
        try:
            self._reset_leds()
        except Exception as e:
            self._handle_io_error("LED write error", e)
        self.pixel_buffer_output.fill(0)  # Clear pixel buffer
        self.hardware_led_state.fill(0)  # Clear hardware state tracking

    def close(self) -> None:
        """Close connection to Launchpad MK2."""
        with self._lock:
            if self.is_connected:
                try:
                    self._reset_leds()
                except Exception as e:
                    logger.debug(f"Error clearing LEDs on close: {e}")
            self._close_ports()
            self.is_connected = False
            self.hardware_led_state.fill(0)  # Reset hardware state tracking

        if self.device_manager:
            self.device_manager.set_disconnected(DeviceType.LAUNCHPAD)
        logger.info("Disconnected from Launchpad MK2")

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    @staticmethod
    def _message_to_xy(msg: mido.Message) -> t.Optional[t.Tuple[int, int, bool]]:
        """Map a raw MIDI message to (x, y, pressed) in the 9x9 XY grid."""
        if msg.type == "control_change" and TOP_ROW_CC_START <= msg.control < TOP_ROW_CC_START + 8:
            return msg.control - TOP_ROW_CC_START, 0, msg.value > 0
        if msg.type in ("note_on", "note_off") and 11 <= msg.note <= 89:
            x = msg.note % 10 - 1
            y = 9 - msg.note // 10
            if 0 <= x <= 8:
                return x, y, msg.type == "note_on" and msg.velocity > 0
        return None

    def get_button_events(
        self,
    ) -> t.Optional[t.Dict[str, t.Any]]:
        """Get button events with smart type classification and relative coordinates."""
        if not self.is_connected or self.input is None:
            return None

        if not self._check_port_present():
            return None

        try:
            # Skip messages that aren't button events (e.g. SysEx replies)
            while True:
                with self._lock:
                    if self.input is None:
                        return None
                    msg = self.input.poll()
                if msg is None:
                    return None
                button = self._message_to_xy(msg)
                if button is not None:
                    break
        except Exception as e:
            self._handle_io_error("Button read error", e)
            return None

        x, y, state = button

        # Determine button type and calculate relative coordinates
        button_type = ButtonType.UNKNOWN
        relative_coords = [0, 0]

        # Check if button is in scenes area
        if (
            self.BOUNDS_SCENES[0][0] <= x <= self.BOUNDS_SCENES[1][0]
            and self.BOUNDS_SCENES[0][1] <= y <= self.BOUNDS_SCENES[1][1]
        ):
            button_type = ButtonType.SCENE
            relative_coords = [
                x - self.BOUNDS_SCENES[0][0],
                y - self.BOUNDS_SCENES[0][1],
            ]

        # Check if button is in presets area
        elif (
            self.BOUNDS_PRESETS[0][0] <= x <= self.BOUNDS_PRESETS[1][0]
            and self.BOUNDS_PRESETS[0][1] <= y <= self.BOUNDS_PRESETS[1][1]
        ):
            button_type = ButtonType.SEQUENCE
            relative_coords = [
                x - self.BOUNDS_PRESETS[0][0],
                y - self.BOUNDS_PRESETS[0][1],
            ]

        # Check if button is in top area
        elif (
            self.BOUNDS_TOP[0][0] <= x <= self.BOUNDS_TOP[1][0]
            and self.BOUNDS_TOP[0][1] <= y <= self.BOUNDS_TOP[1][1]
        ):
            button_type = ButtonType.CONTROL
            relative_coords = [x, y]

        # Check if button is in right area
        elif (
            self.BOUNDS_RIGHT[0][0] <= x <= self.BOUNDS_RIGHT[1][0]
            and self.BOUNDS_RIGHT[0][1] <= y <= self.BOUNDS_RIGHT[1][1]
        ):
            button_type = ButtonType.CONTROL
            relative_coords = [x, y]

        return {"type": button_type, "index": relative_coords, "active": state}
