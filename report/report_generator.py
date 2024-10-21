import torch
import json
import pickle
import os
import numpy as np
from .utils import (
    plot_report_metric,
    split_metric_path,
    build_metrics_dict,
    CustomJSONEncoder,
)


class ReportGenerator:
    def __init__(
        self,
        save_path,
        main_device,
        metrics,
        trainer,
        params_to_save,
        best_metrics_obj,
        plot_args={},
    ):
        self.save_path = save_path
        self.main_device = main_device
        self.metrics = metrics
        self.trainer = trainer
        self.params_to_save = params_to_save
        self.best_metrics_obj = best_metrics_obj
        self.plot_args = plot_args

        if self.params_to_save == "all":
            self.params_to_save = list(self.trainer.__dict__.keys())

        os.makedirs(self.save_path, exist_ok=True)

        self.save_model_ckpts_path = os.path.join(self.save_path, "model_ckpts")
        os.makedirs(self.save_model_ckpts_path, exist_ok=True)

        self.save_model_configs_path = os.path.join(self.save_path, "model_configs")
        os.makedirs(self.save_model_configs_path, exist_ok=True)

        self.save_plots_path = os.path.join(self.save_path, "train_plots")
        os.makedirs(self.save_plots_path, exist_ok=True)

        self.global_metrics_dict = build_metrics_dict(self.metrics)
        self.epoch_metrics_dict = build_metrics_dict(self.metrics)

        self.current_epoch = -1
        self.best_metrics_value = {
            path: np.inf if self.best_metrics_obj[path] == "min" else -np.inf
            for path in self.best_metrics_obj.keys()
        }

    def init_epoch_metrics_dict(self, epoch, device=None):
        if (device is None) or (device == self.main_device):
            if epoch > self.current_epoch:

                self.current_epoch += 1
                self.epoch_metrics_dict = build_metrics_dict(self.metrics)

    def get_last_global_metric(self, path):
        key, graph, line = split_metric_path(path, 3)

        return (
            self.global_metrics_dict[key][graph][line][-1]
            if len(self.global_metrics_dict[key][graph][line]) > 0
            else np.nan
        )

    def get_last_epoch_metric(self, path):
        key, graph, line = split_metric_path(path, 3)

        return (
            self.epoch_metrics_dict[key][graph][line][-1]
            if len(self.epoch_metrics_dict[key][graph][line]) > 0
            else np.nan
        )

    def add_epoch_metric(self, path, value, device=None):
        key, graph, line = split_metric_path(path, 3)

        if (device is None) or (device == self.main_device):
            self.epoch_metrics_dict[key][graph][line].append(value)

    def update_global_metrics(self, device=None):
        if (device is None) or (device == self.main_device):
            for key in self.global_metrics_dict.keys():
                for graph in self.global_metrics_dict[key].keys():
                    for line in self.global_metrics_dict[key][graph].keys():
                        if len(self.epoch_metrics_dict[key][graph][line]) > 0:
                            value = np.mean(self.epoch_metrics_dict[key][graph][line])
                            self.global_metrics_dict[key][graph][line].append(value)

    def save_params(
        self,
        device=None,
    ):
        if (device is None) or (device == self.main_device):
            # Create a dictionary of the parameters to save
            params_dict = {
                param: self.trainer.__dict__[param] for param in self.params_to_save
            }

            # Save the dictionary to a JSON file
            with open(os.path.join(self.save_path, "trainer_params.json"), "w") as f:
                json.dump(params_dict, f, cls=CustomJSONEncoder, indent=4)

    def save_models(
        self,
        models,
        device=None,
    ):
        if (device is None) or (device == self.main_device):
            for name, model in models.items():
                save_path = os.path.join(self.save_model_ckpts_path, f"{name}.pt")
                torch.save(model.state_dict(), save_path)

    def save_model_configs(
        self,
        model_configs,
        device=None,
    ):
        if (device is None) or (device == self.main_device):
            for name, config in model_configs.items():
                save_path = os.path.join(self.save_model_configs_path, f"{name}.pkl")

                with open(save_path, "wb") as f:
                    pickle.dump(config, f)

    def save_best_models(
        self,
        models,
        device=None,
    ):
        if (device is None) or (device == self.main_device):
            for path, value in self.best_metrics_value.items():
                new_value = self.get_last_global_metric(path)

                if ((self.best_metrics_obj[path] == "min") and (new_value < value)) or (
                    (self.best_metrics_obj[path] == "max") and (new_value > value)
                ):
                    self.best_metrics_value[path] = new_value
                    metric_save_name = path.replace("\\", "_").replace("/", "_")

                    for name, model in models.items():
                        save_path = os.path.join(
                            self.save_model_ckpts_path,
                            f"best_{metric_save_name}_{name}.pt",
                        )
                        torch.save(model.state_dict(), save_path)

    def save_metrics(self, device=None):
        if (device is None) or (device == self.main_device):
            with open(os.path.join(self.save_path, "train_metrics.json"), "w") as f:
                json.dump(self.global_metrics_dict, f, cls=CustomJSONEncoder, indent=4)

    def save_plots(self, device=None):
        if (device is None) or (device == self.main_device):
            for metric_key, metric_value in self.global_metrics_dict.items():
                plot_report_metric(
                    metric_key=metric_key,
                    metric_value=metric_value,
                    save_path=self.save_plots_path,
                    **self.plot_args,
                )
