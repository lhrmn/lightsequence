"""
LightSoftware MIDI Communication Utilities  (mido + python-rtmidi)

Clean utility functions for communicating with LightSoftware via loopMIDI.
Only tested with DasLight 4.
"""

import logging
import typing as t

import mido

from lumiblox.common.device_state import DeviceManager, DeviceType
from lumiblox.common.constants import ROWS_PER_PAGE
from lumiblox.midi.midi_manager import midi_manager

if t.TYPE_CHECKING:
    from lumiblox.common.config import ConfigManager

logger = logging.getLogger(__name__)


class LightSoftware:
    def __init__(
        self,
        device_manager: t.Optional[DeviceManager] = None,
        config: t.Optional["ConfigManager"] = None,
    ):
        # Scene mapping: relative coordinates (x, y) to MIDI notes
        self._scene_to_note_map = self._build_scene_note_mapping()

        # Reverse mapping for feedback processing
        self._note_to_scene_map = {v: k for k, v in self._scene_to_note_map.items()}
        
        # MIDI connection variables (mido port objects)
        self.midi_out = None  # type: t.Any
        self.midi_in = None   # type: t.Any
        
        # Connection state flag
        self.connection_good = False
        
        # Device state management
        self.device_manager = device_manager

        # MIDI output values
        if config:
            midi_output_config = config.data.get("midi_output", {})
            self.on_value = midi_output_config.get("on_value", 127)
            self.off_value = midi_output_config.get("off_value", 0)
        else:
            self.on_value = 127
            self.off_value = 0

    def _build_scene_note_mapping(self) -> t.Dict[t.Tuple[int, int], int]:
        """
        Build mapping from page-local scene coordinates to MIDI notes.

        Only covers one page (ROWS_PER_PAGE rows).  Different pages
        reuse the same notes but are distinguished by MIDI channel
        (channel = page index).
        """
        scene_map = {}
        base_notes = [81, 71, 61, 51, 41]

        for x in range(9):  # 9 columns
            for local_y in range(len(base_notes)):
                note = base_notes[local_y] + x
                if 0 <= note <= 127:
                    scene_map[(x, local_y)] = note

        return scene_map

    def _scene_to_note_and_channel(self, scene_index: t.Tuple[int, int]) -> t.Optional[t.Tuple[int, int]]:
        """Return (note, channel) for an absolute scene coordinate, or None."""
        x, y = scene_index
        page = y // ROWS_PER_PAGE
        local_y = y % ROWS_PER_PAGE
        note = self._scene_to_note_map.get((x, local_y))
        if note is None:
            return None
        return note, page

    def _close_ports(self) -> None:
        """Close existing MIDI port objects."""
        midi_manager.close_port(self.midi_out)
        midi_manager.close_port(self.midi_in)
        self.midi_out = None
        self.midi_in = None

    def connect_midi(self) -> bool:
        """
        Connect to LightSoftware via loopMIDI ports.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.device_manager:
                self.device_manager.set_connecting(DeviceType.LIGHT_SOFTWARE)

            # Close any stale port handles before opening new ones
            self._close_ports()

            # Open input (feedback FROM LightSoftware via loopMIDI "lightsoftware_out")
            self.midi_in = midi_manager.open_input_by_keyword("lightsoftware_out")
            if self.midi_in:
                logger.info("✅ LightSoftware IN: %s", self.midi_in.name)
            else:
                logger.error("❌ No LightSoftware_out MIDI input found")

            # Open output (commands TO LightSoftware via loopMIDI "lightsoftware_in")
            self.midi_out = midi_manager.open_output_by_keyword("lightsoftware_in")
            if self.midi_out:
                logger.info("✅ LightSoftware OUT: %s", self.midi_out.name)
            else:
                logger.error("❌ No LightSoftware_in MIDI output found")

            # Update connection state
            if self.midi_out and self.midi_in:
                self.connection_good = True
                if self.device_manager:
                    self.device_manager.set_connected(DeviceType.LIGHT_SOFTWARE)
                logger.info("✅ Successfully connected to LightSoftware MIDI")
                return True
            else:
                self.connection_good = False
                if self.device_manager:
                    self.device_manager.set_disconnected(DeviceType.LIGHT_SOFTWARE)
                logger.error(
                    "❌ Failed to connect to LightSoftware - missing MIDI ports"
                )
                return False

        except Exception as e:
            logger.error("LightSoftware MIDI connection failed: %s", e)
            self.connection_good = False
            if self.device_manager:
                self.device_manager.set_error(DeviceType.LIGHT_SOFTWARE, str(e))
            return False

    def set_scene_state(self, scene_index: t.Tuple[int, int], active: bool) -> None:
        """Set scene state explicitly using a note-on message with velocity for on/off.

        Uses ``midi_manager.safe_send`` for automatic retry on transient
        I/O errors and marks the connection as bad when recovery fails so
        that ``DeviceMonitor`` can trigger a full reconnect.
        """
        if not self.connection_good:
            return

        if not midi_manager.is_port_alive(self.midi_out):
            logger.warning("LightSoftware output port is closed – marking disconnected")
            self._mark_disconnected("Output port closed")
            return

        result = self._scene_to_note_and_channel(scene_index)
        if result is None:
            logger.warning("No MIDI note mapped for scene coordinates %s", scene_index)
            return

        scene_note, channel = result
        velocity = self.on_value if active else self.off_value

        try:
            msg = mido.Message("note_on", note=scene_note, velocity=velocity, channel=channel)
            ok = midi_manager.safe_send(self.midi_out, msg)
            if ok:
                logger.debug(
                    "Sent to LightSoftware: Scene %s -> note %s, velocity %s",
                    scene_index,
                    scene_note,
                    velocity,
                )
            else:
                self._mark_disconnected("safe_send failed")
        except Exception as e:
            logger.error("MIDI send error: %s", e)
            self._mark_disconnected(f"Send error: {e}")

    def get_scene_coordinates_for_note(
        self, note: int, channel: int = 0
    ) -> t.Optional[t.Tuple[int, int]]:
        """
        Get absolute scene coordinates for a MIDI note and channel.

        Args:
            note: MIDI note number
            channel: MIDI channel (used as page index)

        Returns:
            Tuple of (x, y) absolute coordinates or None if not found
        """
        local = self._note_to_scene_map.get(note)
        if local is None:
            return None
        x, local_y = local
        return (x, local_y + channel * ROWS_PER_PAGE)

    def process_feedback(self) -> t.Dict[t.Tuple[int, int], bool]:
        """
        Process MIDI feedback from LightSoftware and return LED state changes.

        Uses the MIDI channel to determine which page a note belongs to,
        then returns absolute scene coordinates as keys.

        Validates port liveness before reading.  On failure the connection
        is marked bad so ``DeviceMonitor`` triggers reconnection.

        Returns:
            Dictionary of scene_coords -> state changes (True=on, False=off)
        """
        changes: t.Dict[t.Tuple[int, int], bool] = {}

        if not self.connection_good:
            return changes

        if not midi_manager.is_port_alive(self.midi_in):
            self._mark_disconnected("Input port closed during feedback")
            return changes

        try:
            for msg in self.midi_in.iter_pending():
                state = self._feedback_state(msg)
                if state is None:
                    continue
                scene = self.get_scene_coordinates_for_note(msg.note, msg.channel)
                if scene is not None:
                    changes[scene] = state

        except Exception as e:
            logger.error("MIDI feedback processing error: %s", e)
            self._mark_disconnected(f"Feedback error: {e}")

        return changes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _feedback_state(msg: "mido.Message") -> t.Optional[bool]:
        """Map a feedback message to scene state: True=on, False=off, None=ignore.

        DasLight reports "off" as note_off (not note_on velocity 0), and "on" as
        note_on with the mapping's configured velocity (often just 1). A click
        also emits a transient low-velocity note_on right before the real state
        message, so processing messages in order always ends on the right state.
        """
        if msg.type == "note_off":
            return False
        if msg.type == "note_on":
            return msg.velocity > 0
        return None

    def _mark_disconnected(self, reason: str) -> None:
        """Centralised helper to flag a broken connection."""
        if not self.connection_good:
            return  # already flagged – avoid log spam
        self.connection_good = False
        logger.warning("LightSoftware connection lost: %s", reason)
        if self.device_manager:
            self.device_manager.set_error(DeviceType.LIGHT_SOFTWARE, reason)

    def close(self) -> None:
        """
        Clean shutdown of LightSoftware MIDI connections.

        Only closes this component's ports.  The shared MIDI subsystem
        is shut down via ``midi_manager.shutdown()`` at application exit.
        Idempotent – safe to call multiple times.
        """
        try:
            self._close_ports()
            logger.info("✅  LightSoftware MIDI ports closed")
        except Exception as e:
            logger.error("Error closing LightSoftware MIDI: %s", e)
        finally:
            self.connection_good = False
            if self.device_manager:
                self.device_manager.set_disconnected(DeviceType.LIGHT_SOFTWARE)
