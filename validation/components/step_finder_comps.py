from validation.components import DataComponent
from validation.utils.io_helpers import save_plotly_fig

LEFT_FOOT_STEPS = DataComponent(
    name="left_foot_steps_figure",
    filename="left_foot_gait_events.html",
    relative_path="validation/{tracker}/gait_events",
    saver=save_plotly_fig,
)

RIGHT_FOOT_STEPS = DataComponent(
    name="right_foot_steps_figure",
    filename="right_foot_gait_events.html",
    relative_path="validation/{tracker}/gait_events",
    saver=save_plotly_fig,
)
