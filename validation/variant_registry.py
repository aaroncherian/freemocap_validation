from enum import Enum, unique
from validation.components import FREEMOCAP_JOINT_CENTERS, FREEMOCAP_RIGID_JOINT_CENTERS


@unique
class MetricsVariant(str, Enum):
    joint_centers = "3d_xyz"
    rigid_joint_centers = "rigid_3d_xyz"

VARIANT_TO_COMPONENT = {
    MetricsVariant.joint_centers: FREEMOCAP_JOINT_CENTERS,
    MetricsVariant.rigid_joint_centers: FREEMOCAP_RIGID_JOINT_CENTERS
}


