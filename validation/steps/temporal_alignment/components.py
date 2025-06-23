from validation.components import (QUALISYS_MARKERS, 
                                   QUALISYS_START_TIME, 
                                   QUALISYS_SYNCED_JOINT_CENTERS,
                                   QUALISYS_SYNCED_MARKER_DATA, 
                                   QUALISYS_COM, 
                                   FREEMOCAP_PRE_SYNC_JOINT_CENTERS, 
                                   FREEMOCAP_TIMESTAMPS,
                                   FREEMOCAP_PREALPHA_TIMESTAMPS,
)

REQUIRES = [FREEMOCAP_TIMESTAMPS, 
            QUALISYS_MARKERS, 
            QUALISYS_START_TIME,
            FREEMOCAP_PRE_SYNC_JOINT_CENTERS]


# REQUIRES = [FREEMOCAP_PREALPHA_TIMESTAMPS, 
#             QUALISYS_MARKERS, 
#             QUALISYS_START_TIME,
#             FREEMOCAP_PRE_SYNC_JOINT_CENTERS]


PRODUCES = [QUALISYS_SYNCED_JOINT_CENTERS, QUALISYS_SYNCED_MARKER_DATA]