from enum import Enum, unique
from validation.components import FREEMOCAP_JOINT_CENTERS, FREEMOCAP_RIGID_JOINT_CENTERS


@unique
class MetricsVariant(str, Enum):
    joint_centers = "non_rigid"
    rigid_joint_centers = "rigid"

VARIANT_TO_COMPONENT = {
    MetricsVariant.joint_centers: FREEMOCAP_JOINT_CENTERS,
    MetricsVariant.rigid_joint_centers: FREEMOCAP_RIGID_JOINT_CENTERS
}


