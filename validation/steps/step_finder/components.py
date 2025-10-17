
from validation.components import (FREEMOCAP_PARQUET, 
                                   QUALISYS_PARQUET, 
                                   FREEMOCAP_GAIT_EVENTS, 
                                   QUALISYS_GAIT_EVENTS,
                                   LEFT_FOOT_STEPS,
                                    RIGHT_FOOT_STEPS)

REQUIRES = [QUALISYS_PARQUET, FREEMOCAP_PARQUET]
PRODUCES = [FREEMOCAP_GAIT_EVENTS, QUALISYS_GAIT_EVENTS, LEFT_FOOT_STEPS, RIGHT_FOOT_STEPS]