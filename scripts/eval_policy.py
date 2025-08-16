# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import pathlib
import pickle
import warnings
from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np
import tyro

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import BasePolicy, Gr00tPolicy
from gr00t.utils.eval import calc_mse_for_single_trajectory

warnings.simplefilter("ignore", category=FutureWarning)

"""
Example command:

NOTE: provide --model_path to load up the model checkpoint in this script,
        else it will use the default host and port via RobotInferenceClient

python scripts/eval_policy.py --plot --model-path nvidia/GR00T-N1.5-3B
"""


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""

    host: str = "localhost"
    """Host to connect to."""

    port: int = 5555
    """Port to connect to."""

    plot: bool = False
    """Whether to plot the images."""

    modality_keys: List[str] = field(default_factory=lambda: ["right_arm", "left_arm"])
    """Modality keys to evaluate."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_only"
    """Data config to use."""

    steps: int = None
    """Number of steps to evaluate. If None, will evaluate all steps in each trajectory."""

    trajs: int = None
    """Number of trajectories to evaluate. If None, will evaluate all trajectories in the dataset."""

    action_horizon: int = None
    """Action horizon to evaluate. If None, will use the data config's action horizon."""

    video_backend: Literal["decord", "torchvision_av"] = "decord"
    """Video backend to use for various codec options. h264: decord or av: torchvision_av"""

    dataset_path: str = "demo_data/robot_sim.PickNPlace/"
    """Path to the dataset."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    """Embodiment tag to use."""

    model_path: str = None
    """Path to the model checkpoint."""

    denoising_steps: int = 4
    """Number of denoising steps to use."""

    save_plot_path: str = None
    """Path to save the plot."""

    save_data_path: str = None
    """Path to save trajectory inference data (JSON, pickle, NPZ files)."""


def save_trajectory_data(
    pred_actions: np.ndarray,
    gt_actions: np.ndarray,
    prediction_points: list,
    metrics: dict,
    traj_id: int,
    save_path: pathlib.Path,
    inference_times: list = None,
    detailed_inference_data: list = None,
):
    """Save detailed trajectory data for offline analysis."""

    # Create trajectory-specific directory
    traj_dir = save_path / f"trajectory_{traj_id}"
    traj_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays in metrics to lists for JSON serialization
    metrics_json_safe = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_json_safe[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            metrics_json_safe[key] = value.item()
        else:
            metrics_json_safe[key] = value

    # Helper function to convert numpy types to JSON serializable types
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, dict):
            return {key: convert_to_json_serializable(val) for key, val in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    # Convert prediction_points to JSON serializable format
    prediction_points_json = []
    for step, horizon in prediction_points:
        prediction_point = {
            "step": int(step),
            "horizon_prediction": convert_to_json_serializable(horizon)
        }
        prediction_points_json.append(prediction_point)

    # Prepare data for export
    trajectory_data = {
        "trajectory_id": traj_id,
        "metadata": {
            "trajectory_length": metrics.get("trajectory_length", len(gt_actions)),
            "steps_evaluated": metrics.get("steps_evaluated", len(pred_actions)),
            "completion_rate": metrics.get("trajectory_completion", 1.0),
            "prediction_points_count": len(prediction_points),
        },
        "metrics": metrics_json_safe,
        "arrays": {
            "predicted_actions": pred_actions.tolist(),
            "ground_truth_actions": gt_actions.tolist(),
        },
        "prediction_points": prediction_points_json,
        "timing": inference_times if inference_times else [],
        "detailed_inference_data": detailed_inference_data if detailed_inference_data else [],
    }

    # Save as JSON (human-readable)
    json_path = traj_dir / "data.json"
    with open(json_path, "w") as f:
        json.dump(trajectory_data, f, indent=2)

    # Save as pickle (preserves numpy arrays exactly)
    pickle_path = traj_dir / "data.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(
            {
                "pred_actions": pred_actions,
                "gt_actions": gt_actions,
                "prediction_points": prediction_points,
                "metrics": metrics,  # Keep original metrics with numpy arrays
                "inference_times": inference_times,
                "detailed_inference_data": detailed_inference_data,
            },
            f,
        )

    # Save individual arrays as NPZ
    npz_path = traj_dir / "arrays.npz"
    np.savez(
        npz_path,
        predicted_actions=pred_actions,
        ground_truth_actions=gt_actions,
        **{f"horizon_{i}": convert_to_json_serializable(horizon) for i, (step, horizon) in enumerate(prediction_points)},
    )

    print(f"  Saved trajectory data to: {traj_dir}")
    return traj_dir

# You'll need to modify calc_mse_for_single_trajectory to return the data
def calc_mse_for_single_trajectory_with_data(
    policy, dataset, traj_id, modality_keys, steps, action_horizon, plot=False, save_plot_path=None
):
    """Modified version that returns prediction data along with MSE."""
    
    # Call the original function with return_data=True to get the additional data
    return calc_mse_for_single_trajectory(
        policy, dataset, traj_id, modality_keys, steps, action_horizon, plot, save_plot_path, return_data=True
    )


def main(args: ArgsConfig):
    data_config = DATA_CONFIG_MAP[args.data_config]

    # Set action_horizon from data config if not provided
    if args.action_horizon is None:
        args.action_horizon = len(data_config.action_indices)
        print(f"Using action_horizon={args.action_horizon} from data config '{args.data_config}'")

    # Create save_plot_path directory if it doesn't exist
    if args.save_plot_path is not None:
        import os
        os.makedirs(args.save_plot_path, exist_ok=True)
        print(f"Created/verified directory: {args.save_plot_path}")

    # Create save_data_path directory if it doesn't exist
    if args.save_data_path is not None:
        save_data_path = pathlib.Path(args.save_data_path)
        save_data_path.mkdir(parents=True, exist_ok=True)
        print(f"Created/verified data directory: {save_data_path}")
    else:
        save_data_path = None

    if args.model_path is not None:
        import torch

        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy: BasePolicy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        policy: BasePolicy = RobotInferenceClient(host=args.host, port=args.port)

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    print("Current modality config: \n", modality)

    # Create the dataset
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag=args.embodiment_tag,
    )

    print(len(dataset))
    # Make a prediction
    obs = dataset[0]
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    for k, v in dataset.get_step_data(0, 0).items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    print("Total trajectories:", len(dataset.trajectory_lengths))
    print("All trajectories:", dataset.trajectory_lengths)
    
    # If trajs is None, evaluate all trajectories in the dataset
    if args.trajs is None:
        num_trajs = len(dataset.trajectory_lengths)
        print(f"Evaluating all {num_trajs} trajectories in the dataset")
    else:
        num_trajs = min(args.trajs, len(dataset.trajectory_lengths))
        print(f"Evaluating {num_trajs} trajectories")
    
    print("Running on all trajs with modality keys:", args.modality_keys)

    all_mse = []
    all_trajectory_data = []  # Store data for summary
    
    for traj_id in range(num_trajs):
        # Get the length of current trajectory
        traj_length = dataset.trajectory_lengths[traj_id]
        
        # Use trajectory length if steps is None, otherwise use specified steps
        if args.steps is None:
            steps_to_eval = traj_length
            print(f"Running trajectory {traj_id}/{num_trajs}: evaluating all {steps_to_eval} steps")
        else:
            steps_to_eval = min(args.steps, traj_length)
            print(f"Running trajectory {traj_id}/{num_trajs}: evaluating {steps_to_eval}/{traj_length} steps")
        
        # Create trajectory-specific save path
        if args.save_plot_path is not None:
            traj_save_path = os.path.join(args.save_plot_path, f"trajectory_{traj_id}.png")
        else:
            traj_save_path = None
        
        import time
        start_time = time.time()
        
        # Call the modified calc_mse_for_single_trajectory that returns additional data
        result = calc_mse_for_single_trajectory_with_data(
            policy,
            dataset,
            traj_id,
            modality_keys=args.modality_keys,
            steps=steps_to_eval,
            action_horizon=args.action_horizon,
            plot=args.plot,
            save_plot_path=traj_save_path,
        )
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        if isinstance(result, tuple):
            mse, pred_actions, gt_actions, prediction_points = result
        else:
            mse = result
            pred_actions = gt_actions = prediction_points = None
        
        print(f"MSE: {mse}, Time taken: {inference_time:.2f} seconds")
        all_mse.append(mse)
        
        # Save trajectory data if path is provided and data is available
        if save_data_path is not None and pred_actions is not None:
            metrics = {
                "mse": float(mse),
                "trajectory_length": int(traj_length),
                "steps_evaluated": int(steps_to_eval),
                "inference_time": float(inference_time),
                "action_horizon": int(args.action_horizon),
                "modality_keys": list(args.modality_keys),  # Ensure it's a list of strings
            }
            
            save_trajectory_data(
                pred_actions=pred_actions,
                gt_actions=gt_actions,
                prediction_points=prediction_points,
                metrics=metrics,
                traj_id=traj_id,
                save_path=save_data_path,
                inference_times=[float(inference_time)],
            )
            
            all_trajectory_data.append({
                "trajectory_id": int(traj_id),
                "mse": float(mse),
                "inference_time": float(inference_time),
                "steps_evaluated": int(steps_to_eval),
                "trajectory_length": int(traj_length),
            })

    # Save summary data
    if save_data_path is not None:
        # Helper function to convert numpy types to JSON serializable types
        def convert_summary_to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, dict):
                return {key: convert_summary_to_json_serializable(val) for key, val in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_summary_to_json_serializable(item) for item in obj]
            else:
                return obj

        summary_data = {
            "evaluation_summary": {
                "total_trajectories": int(num_trajs),
                "average_mse": float(np.mean(all_mse)),
                "mse_std": float(np.std(all_mse)),
                "total_evaluation_time": float(sum([t["inference_time"] for t in all_trajectory_data])),
                "configuration": {
                    "data_config": args.data_config,
                    "action_horizon": int(args.action_horizon),
                    "modality_keys": args.modality_keys,
                    "embodiment_tag": args.embodiment_tag,
                    "denoising_steps": int(args.denoising_steps),
                }
            },
            "trajectory_summaries": convert_summary_to_json_serializable(all_trajectory_data),
        }
        
        summary_path = save_data_path / "evaluation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)
        print(f"Saved evaluation summary to: {summary_path}")
    
    print("Average MSE across all trajs:", np.mean(all_mse))
    print("Done")
    exit()


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    main(config)
