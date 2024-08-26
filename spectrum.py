# spectrum.py
import torch
from transformers import AutoModelForCausalLM, AutoConfig
import numpy as np
import json
from prompt_toolkit.shortcuts import checkboxlist_dialog, input_dialog
import argparse
from tqdm import tqdm
import os
import time


class ModelModifier:
    def __init__(self, model_name=None, top_percent=50, batch_size=1):
        self.model_name = model_name
        self.top_percent = top_percent
        self.batch_size = batch_size
        self.model = self._load_model() if model_name else None
        self.layer_snr = {}
        self.layer_types = []

    def _load_model(self):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto",
            )
        except KeyError as e:
            print(f"Error loading model: {e}")
            print("Attempting to load with custom configuration...")
            config = AutoConfig.from_pretrained(self.model_name)
            config.rope_scaling = {"type": "linear", "factor": 1.0}
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=config,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto",
            )

        self._ensure_rope_scaling(model)
        return model

    @staticmethod
    def _ensure_rope_scaling(model):
        if not hasattr(model.config, "rope_scaling"):
            model.config.rope_scaling = {"type": "linear"}
        elif not isinstance(model.config.rope_scaling, dict):
            model.config.rope_scaling = {"type": "linear"}
        elif "type" not in model.config.rope_scaling:
            model.config.rope_scaling["type"] = "linear"

    def get_weight_types(self):
        weight_types = set()
        for name, module in self.model.named_modules():
            parts = name.split(".")
            if any(hasattr(module, attr) for attr in ["weight", "bias", "inv_freq"]):
                layer_index = next(
                    (i for i, part in enumerate(parts) if part.isdigit()), -1
                )
                weight_type = (
                    ".".join(parts[layer_index + 1 :]) if layer_index != -1 else name
                )
                weight_types.add(weight_type)
        return list(weight_types)

    def interactive_select_weights(self):
        weight_types = self.get_weight_types()
        sorted_weight_types = self._sort_weight_types(weight_types)
        selected_types = checkboxlist_dialog(
            title="Select Weight Types",
            text="Deselect the weight types you do not want to scan for SNR:",
            values=[(wt, wt) for wt in sorted_weight_types],
            default_values=sorted_weight_types,
        ).run()
        self.layer_types = selected_types
        return selected_types

    @staticmethod
    def _sort_weight_types(weight_types):
        categories = {}
        for wt in weight_types:
            category = wt.split(".")[0]
            categories.setdefault(category, []).append(wt)
        sorted_categories = {
            k: sorted(v)
            for k, v in sorted(categories.items(), key=lambda item: item[0])
        }
        return [wt for sublist in sorted_categories.values() for wt in sublist]

    def calculate_snr_for_layer(self, layer_type):
        layers = [
            (name, module)
            for name, module in self.model.named_modules()
            if layer_type in name and hasattr(module, "weight")
        ]
        num_batches = (len(layers) + self.batch_size - 1) // self.batch_size

        with tqdm(
            total=num_batches, unit="batch", desc=f"Calculating SNR for {layer_type}"
        ) as progress_bar:
            for i in range(0, len(layers), self.batch_size):
                batch_layers = layers[i : i + self.batch_size]
                for name, module in batch_layers:
                    weights = module.weight.detach()
                    if weights.ndim < 2:
                        weights = weights.unsqueeze(0)
                    S = torch.linalg.svdvals(weights)
                    max_singular_value = S[0]
                    sigma_estimated = self._estimate_sigma_with_full_iqr(S)
                    n, m = weights.shape[-2:]
                    mp_threshold = self._marchenko_pastur_threshold(
                        sigma_estimated, n, m
                    )
                    signal = S[S > mp_threshold].sum()
                    noise = S[S <= mp_threshold].sum()
                    snr = signal / noise if noise != 0 else float("inf")
                    snr_ratio = snr / max_singular_value
                    self.layer_snr[name] = {"type": layer_type, "snr": snr_ratio.item()}
                progress_bar.update(1)

    @staticmethod
    def _marchenko_pastur_threshold(sigma, n, m):
        beta = n / m if n < m else m / n
        return sigma * np.sqrt((1 + np.sqrt(beta)) ** 2)

    @staticmethod
    def _estimate_sigma_with_full_iqr(S):
        q75, q25 = torch.quantile(S, torch.tensor([0.75, 0.25]))
        return (q75 - q25) / 1.349

    def assess_layers_snr(self, selected_weight_types):
        start_time = time.time()
        with tqdm(
            total=len(selected_weight_types),
            unit="type",
            desc="Calculating SNR for types",
        ) as progress_bar:
            for layer_type in selected_weight_types:
                self.calculate_snr_for_layer(layer_type)
                progress_bar.update(1)
        print(f"Total time taken: {time.time() - start_time:.2f} seconds")

    def save_snr_to_json(self):
        model_name_slug = self.model_name.replace("/", "-").replace("_", "-")
        directory = "model_snr_results"
        filename = os.path.join(directory, f"snr_results_{model_name_slug}.json")

        os.makedirs(directory, exist_ok=True)

        serializable_data = {
            layer_name: {
                "snr": (
                    info["snr"].item()
                    if isinstance(info["snr"], torch.Tensor)
                    else info["snr"]
                ),
                "type": str(info["type"]),
            }
            for layer_name, info in self.layer_snr.items()
        }

        with open(filename, "w") as file:
            json.dump(serializable_data, file, indent=4)

        print(f"Results saved to {filename}")
        self._save_top_snr_ratios_to_json(filename)
        self._generate_unfrozen_params_yaml(filename)

    def _generate_unfrozen_params_yaml(self, json_filename, top_percent=None):
        top_percent = top_percent if top_percent is not None else self.top_percent
        with open(json_filename, "r") as file:
            snr_data = json.load(file)

        unfrozen_parameters = {}
        for layer_name, info in snr_data.items():
            layer_type = info["type"]
            unfrozen_parameters.setdefault(layer_type, []).append(
                (layer_name, info["snr"])
            )

        top_layers_by_type = {
            layer_type: [
                layer[0]
                for layer in sorted(layers, key=lambda x: x[1], reverse=True)[
                    : int(len(layers) * top_percent / 100)
                ]
            ]
            for layer_type, layers in unfrozen_parameters.items()
        }

        json_file_base = os.path.splitext(os.path.basename(json_filename))[0]
        yaml_filename = f"{json_file_base}_unfrozenparameters_{top_percent}percent.yaml"

        with open(yaml_filename, "w") as file:
            file.write("unfrozen_parameters:\n")
            file.write("- ^lm_head.weight$\n")
            file.write("- ^model.embed_tokens.weight$\n")
            for layer_type, layer_names in top_layers_by_type.items():
                file.write(f"# {layer_type} layers\n")
                for layer_name in layer_names:
                    file.write(f"- {layer_name}\n")

        print(f"Top {top_percent}% SNR layers saved to {yaml_filename}")

    def _save_top_snr_ratios_to_json(self, json_filename, filename=None):
        with open(json_filename, "r") as file:
            snr_data = json.load(file)

        all_snr_layers = {}
        for layer_name, info in snr_data.items():
            layer_type = info["type"]
            all_snr_layers.setdefault(layer_type, []).append((layer_name, info["snr"]))

        all_snr_layers = {
            layer_type: dict(sorted(layers, key=lambda x: x[1], reverse=True))
            for layer_type, layers in all_snr_layers.items()
        }

        json_file_base = os.path.splitext(os.path.basename(json_filename))[0]
        filename = f"{json_file_base}_sorted.json" if filename is None else filename

        with open(filename, "w") as file:
            json.dump(all_snr_layers, file, indent=4)
        print(f"All SNR layers sorted and saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Process SNR data for layers.")
    parser.add_argument(
        "--model-name", type=str, required=True, help="Model name or path to the model"
    )
    parser.add_argument(
        "--top-percent",
        type=int,
        default=50,
        help="Top percentage of layers to select, overriding the default",
    )
    args = parser.parse_args()

    model_name_slug = args.model_name.replace("/", "-").replace("_", "-")
    snr_file_path = os.path.join(
        "model_snr_results", f"snr_results_{model_name_slug}.json"
    )

    if os.path.exists(snr_file_path):
        print(f"Found existing SNR results file for {args.model_name}")
        modifier = ModelModifier(top_percent=args.top_percent)
        modifier._generate_unfrozen_params_yaml(snr_file_path, args.top_percent)
    else:
        print(
            f"No existing SNR results file found for {args.model_name}. Proceeding with SNR calculation."
        )
        batch_size = int(
            input_dialog(title="Batch Size", text="Enter the batch size:").run() or 1
        )
        modifier = ModelModifier(
            model_name=args.model_name,
            batch_size=batch_size,
            top_percent=args.top_percent,
        )
        selected_weight_types = modifier.interactive_select_weights()
        if selected_weight_types:
            modifier.assess_layers_snr(selected_weight_types)
            modifier.save_snr_to_json()
            print("Finished SNR scanning and data saved.")
        else:
            print("No weight types selected.")


if __name__ == "__main__":
    main()
