from AutoTuner.utils.nested_dict import NestedDict
from AutoTuner.utils.structs import InputTestCase

from ..configs.config_struct import ProfileConfig, ProfileMode
from .launcher import Launcher


class LaunchDataCollectionForOps(Launcher):
    def __init__(
        self,
        profile_config: ProfileConfig,
        test_cases: list[InputTestCase],
        model_name: str,
        override_model_kwargs: dict,
        override_tf_config_kwargs: dict,
        theoretical_flops: bool = False,
        theoretical_activations: bool = False,
    ):
        assert (
            profile_config.profile_mode == ProfileMode.collect_data
        ), "Data collection should not be in profile mode."
        super().__init__(
            profile_config=profile_config,
            test_cases=test_cases,
            model_name=model_name,
            override_model_kwargs=override_model_kwargs,
            override_tf_config_kwargs=override_tf_config_kwargs,
            theoretical_flops=theoretical_flops,
            theoretical_activations=theoretical_activations,
        )

        self.total_timing_db = NestedDict()
        self.total_memory_db = {"weights": {}, "activations": NestedDict()}

    def run_op(self, op_name: str, test_case_idxs: list[int]):
        op_class_instance = super().run_op(op_name, test_case_idxs=test_case_idxs)
        timing_db, memory_db = op_class_instance.get_results()
        self.total_timing_db.merge(timing_db)
        self.total_memory_db["weights"].update(memory_db["weights"])
        self.total_memory_db["activations"].merge(memory_db["activations"])

    def run_op_list(self, op_name_list: list[str], test_case_idxs: list[int]):
        for op_name in op_name_list:
            print(f"Running operator: {op_name}")
            self.run_op(op_name, test_case_idxs=test_case_idxs)

    def run_all_supported_ops(self, test_case_idxs: list[int]):
        self.run_op_list(self.all_supported_ops, test_case_idxs=test_case_idxs)

    def return_results(self):
        return self.total_timing_db, self.total_memory_db
