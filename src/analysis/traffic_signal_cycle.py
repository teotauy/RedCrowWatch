#!/usr/bin/env python3
"""
Traffic Signal Cycle Manager for NYC Intersection

Implements the 88-second traffic signal cycle:
- 25s: Walk signal (white walk + orange countdown)
- 36s: Expressway offramp green/yellow
- 27s: 19th street green/yellow
- Plus NYC right-on-red allowance
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class SignalPhase(Enum):
    """Traffic signal phases"""
    WALK_SIGNAL = "walk_signal"  # 25s - pedestrians can walk
    EXPRESSWAY_GREEN = "expressway_green"  # 36s - offramp has green
    STREET_GREEN = "street_green"  # 27s - 19th street has green


class TrafficSignalCycle:
    """
    Manages the 88-second traffic signal cycle for NYC intersection
    """
    
    def __init__(self, config: Optional[Dict] = None, cycle_start_time: Optional[datetime] = None):
        """
        Initialize traffic signal cycle
        
        Args:
            config: Configuration dictionary with cycle settings
            cycle_start_time: When the cycle started (defaults to now)
        """
        if config and 'traffic_signal_cycle' in config:
            cycle_config = config['traffic_signal_cycle']
            self.cycle_duration = cycle_config.get('total_cycle_duration', 88)
            phases_config = cycle_config.get('phases', {})
            self.phase_durations = {
                SignalPhase.WALK_SIGNAL: phases_config.get('walk_signal', {}).get('duration', 25),
                SignalPhase.EXPRESSWAY_GREEN: phases_config.get('expressway_green', {}).get('duration', 36),
                SignalPhase.STREET_GREEN: phases_config.get('street_green', {}).get('duration', 27)
            }
            self.right_on_red_allowed = cycle_config.get('right_on_red_allowed', True)
        else:
            # Default values
            self.cycle_duration = 88  # Total cycle time in seconds
            self.phase_durations = {
                SignalPhase.WALK_SIGNAL: 25,  # Walk signal + countdown
                SignalPhase.EXPRESSWAY_GREEN: 36,  # Offramp green + yellow
                SignalPhase.STREET_GREEN: 27  # 19th street green + yellow
            }
            self.right_on_red_allowed = True
        
        # Calculate phase start times within cycle
        self.phase_start_times = {}
        cumulative_time = 0
        for phase, duration in self.phase_durations.items():
            self.phase_start_times[phase] = cumulative_time
            cumulative_time += duration
        
        # Set cycle start time
        self.cycle_start_time = cycle_start_time or datetime.now()
        
        logger.info(f"Traffic signal cycle initialized. Cycle duration: {self.cycle_duration}s")
        logger.info(f"Phase durations: Walk={self.phase_durations[SignalPhase.WALK_SIGNAL]}s, "
                   f"Expressway={self.phase_durations[SignalPhase.EXPRESSWAY_GREEN]}s, "
                   f"Street={self.phase_durations[SignalPhase.STREET_GREEN]}s")
    
    def get_current_phase(self, timestamp: datetime) -> SignalPhase:
        """
        Get the current traffic signal phase at given timestamp
        
        Args:
            timestamp: Current time
            
        Returns:
            Current signal phase
        """
        # Calculate time elapsed since cycle start
        elapsed = (timestamp - self.cycle_start_time).total_seconds()
        
        # Handle multiple cycles
        cycle_position = elapsed % self.cycle_duration
        
        # Determine current phase
        if cycle_position < self.phase_start_times[SignalPhase.WALK_SIGNAL] + self.phase_durations[SignalPhase.WALK_SIGNAL]:
            return SignalPhase.WALK_SIGNAL
        elif cycle_position < self.phase_start_times[SignalPhase.EXPRESSWAY_GREEN] + self.phase_durations[SignalPhase.EXPRESSWAY_GREEN]:
            return SignalPhase.EXPRESSWAY_GREEN
        else:
            return SignalPhase.STREET_GREEN
    
    def get_phase_info(self, timestamp: datetime) -> Dict:
        """
        Get detailed information about current phase
        
        Args:
            timestamp: Current time
            
        Returns:
            Dictionary with phase information
        """
        current_phase = self.get_current_phase(timestamp)
        elapsed = (timestamp - self.cycle_start_time).total_seconds()
        cycle_position = elapsed % self.cycle_duration
        
        # Calculate time within current phase
        phase_start = self.phase_start_times[current_phase]
        time_in_phase = cycle_position - phase_start
        phase_duration = self.phase_durations[current_phase]
        time_remaining = max(0, phase_duration - time_in_phase)
        
        # Determine sub-phase (e.g., green vs yellow)
        sub_phase = self._get_sub_phase(current_phase, time_in_phase)
        
        return {
            'phase': current_phase,
            'phase_name': current_phase.value,
            'time_in_phase': time_in_phase,
            'time_remaining': time_remaining,
            'sub_phase': sub_phase,
            'cycle_position': cycle_position,
            'cycle_progress': (cycle_position / self.cycle_duration) * 100
        }
    
    def _get_sub_phase(self, phase: SignalPhase, time_in_phase: float) -> str:
        """
        Get sub-phase within a main phase (e.g., green vs yellow)
        
        Args:
            phase: Current main phase
            time_in_phase: Time elapsed in current phase
            
        Returns:
            Sub-phase name
        """
        if phase == SignalPhase.WALK_SIGNAL:
            if time_in_phase < 10:  # First 10 seconds - white walk signal
                return "walk_white"
            else:  # Remaining 15 seconds - orange countdown
                return "walk_countdown"
        
        elif phase == SignalPhase.EXPRESSWAY_GREEN:
            if time_in_phase < 30:  # First 30 seconds - green
                return "green"
            else:  # Last 6 seconds - yellow
                return "yellow"
        
        elif phase == SignalPhase.STREET_GREEN:
            if time_in_phase < 22:  # First 22 seconds - green
                return "green"
            else:  # Last 5 seconds - yellow
                return "yellow"
        
        return "unknown"
    
    def is_red_light_violation(self, timestamp: datetime, vehicle_zone: str, 
                              vehicle_direction: str, is_right_turn: bool = False) -> bool:
        """
        Determine if a vehicle movement constitutes a red light violation
        
        Args:
            timestamp: Time of vehicle movement
            vehicle_zone: Zone where vehicle is located
            vehicle_direction: Direction of vehicle movement
            is_right_turn: Whether this is a right turn
            
        Returns:
            True if this is a red light violation
        """
        phase_info = self.get_phase_info(timestamp)
        current_phase = phase_info['phase']
        sub_phase = phase_info['sub_phase']
        
        # NYC allows right turns on red (except where posted)
        if is_right_turn and self.right_on_red_allowed:
            return False
        
        # Determine if movement is allowed based on phase and zone
        if current_phase == SignalPhase.WALK_SIGNAL:
            # During walk signal, most vehicle movement is prohibited
            # Exception: vehicles already in intersection may complete movement
            return vehicle_zone not in ['intersection_core']
        
        elif current_phase == SignalPhase.EXPRESSWAY_GREEN:
            # Expressway offramp has green, other movements may be restricted
            if vehicle_zone == 'expressway_offramp':
                return sub_phase == "yellow" and vehicle_zone == 'intersection_core'
            else:
                # Other zones may have red light
                return vehicle_zone in ['intersection_core', 'one_way_street_approach']
        
        elif current_phase == SignalPhase.STREET_GREEN:
            # 19th street has green, other movements may be restricted
            if vehicle_zone in ['one_way_street_approach', 'one_way_avenue_approach']:
                return sub_phase == "yellow" and vehicle_zone == 'intersection_core'
            else:
                # Other zones may have red light
                return vehicle_zone in ['intersection_core', 'expressway_offramp']
        
        return True  # Default to violation if phase is unknown
    
    def get_allowed_movements(self, timestamp: datetime) -> Dict[str, List[str]]:
        """
        Get list of allowed vehicle movements for current phase
        
        Args:
            timestamp: Current time
            
        Returns:
            Dictionary mapping zones to allowed movement directions
        """
        phase_info = self.get_phase_info(timestamp)
        current_phase = phase_info['phase']
        sub_phase = phase_info['sub_phase']
        
        allowed = {
            'expressway_offramp': [],
            'one_way_street_approach': [],
            'one_way_avenue_approach': [],
            'intersection_core': [],
            'bike_lane': [],
            'pedestrian_crossing': []
        }
        
        if current_phase == SignalPhase.WALK_SIGNAL:
            # Minimal vehicle movement during walk signal
            if sub_phase == "walk_white":
                # Only vehicles already in intersection can complete movement
                allowed['intersection_core'] = ['complete_movement']
            # During countdown, no new vehicle entry
        
        elif current_phase == SignalPhase.EXPRESSWAY_GREEN:
            # Expressway offramp has priority
            if sub_phase == "green":
                allowed['expressway_offramp'] = ['straight', 'right_turn', 'left_turn']
                allowed['intersection_core'] = ['complete_movement']
            elif sub_phase == "yellow":
                # Yellow - only complete movement, no new entries
                allowed['intersection_core'] = ['complete_movement']
        
        elif current_phase == SignalPhase.STREET_GREEN:
            # 19th street has priority
            if sub_phase == "green":
                allowed['one_way_street_approach'] = ['straight', 'right_turn', 'left_turn']
                allowed['one_way_avenue_approach'] = ['straight', 'right_turn']
                allowed['intersection_core'] = ['complete_movement']
            elif sub_phase == "yellow":
                # Yellow - only complete movement, no new entries
                allowed['intersection_core'] = ['complete_movement']
        
        return allowed
    
    def reset_cycle(self, new_start_time: Optional[datetime] = None):
        """
        Reset the traffic signal cycle
        
        Args:
            new_start_time: New cycle start time (defaults to now)
        """
        self.cycle_start_time = new_start_time or datetime.now()
        logger.info(f"Traffic signal cycle reset. New start time: {self.cycle_start_time}")
    
    def get_cycle_schedule(self) -> List[Dict]:
        """
        Get the complete cycle schedule
        
        Returns:
            List of phase information for the entire cycle
        """
        schedule = []
        current_time = self.cycle_start_time
        
        for phase in SignalPhase:
            phase_start = current_time + timedelta(seconds=self.phase_start_times[phase])
            phase_end = phase_start + timedelta(seconds=self.phase_durations[phase])
            
            schedule.append({
                'phase': phase,
                'phase_name': phase.value,
                'start_time': phase_start,
                'end_time': phase_end,
                'duration': self.phase_durations[phase]
            })
        
        return schedule
