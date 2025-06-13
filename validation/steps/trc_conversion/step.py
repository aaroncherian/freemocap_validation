from validation.steps.trc_conversion.core.convert_to_trc import create_freemocap_trc, create_qualisys_trc, TRCResult
from validation.components import FREEMOCAP_JOINT_CENTERS, FREEMOCAP_TRC, QUALISYS_SYNCED_MARKER_DATA, QUALISYS_TRC
from validation.pipeline.base import ValidationStep
from validation.steps.trc_conversion.components import REQUIRES, PRODUCES

class TRCConversionStep(ValidationStep):
    REQUIRES = REQUIRES
    PRODUCES = PRODUCES

    def calculate(self):
        self.logger.info("Starting spatial alignment")


        freemocap_trc_result:TRCResult = create_freemocap_trc(
            tracker_name=self.ctx.project_config.freemocap_tracker,
            tracked_points_data=self.data[FREEMOCAP_JOINT_CENTERS.name],
        )

        qualisys_trc_result:TRCResult = create_qualisys_trc(
            df = self.data[QUALISYS_SYNCED_MARKER_DATA.name]
        )

        self.outputs[FREEMOCAP_TRC.name] = freemocap_trc_result
        self.outputs[QUALISYS_TRC.name] = qualisys_trc_result




