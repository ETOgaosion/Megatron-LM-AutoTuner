from AutoTuner.utils.structs import InputTestCase

from ..configs.config_struct import ProfileConfig, ProfileMode
from .launcher import Launcher


class LaunchNsysProfileForOps(Launcher):
    def __init__(
        self,
        profile_config: ProfileConfig,
        test_cases: list[InputTestCase],
        model_name: str,
        override_model_kwargs: dict,
        override_tf_config_kwargs: dict,
        fix_compute_amount: bool = True,
        tp_comm_overlap_cfg: str = None,
    ):
        assert (
            profile_config.profile_mode == ProfileMode.nsys_profile
        ), "Nsys profile should enable nsys profile mode."
        super().__init__(
            profile_config=profile_config,
            test_cases=test_cases,
            model_name=model_name,
            override_model_kwargs=override_model_kwargs,
            override_tf_config_kwargs=override_tf_config_kwargs,
            fix_compute_amount=fix_compute_amount,
            tp_comm_overlap_cfg=tp_comm_overlap_cfg,
        )
