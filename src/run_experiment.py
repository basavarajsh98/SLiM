#!/usr/bin/env python3
"""
Example script demonstrating how to use the dynamic training system.
This script shows how to run different types of experiments with the refactored train.py
"""

from train import main


def run_detoxification_experiment():
    """Run detoxification experiment with custom configuration."""
    print("=" * 60)
    print("RUNNING DETOXIFICATION EXPERIMENT")
    print("=" * 60)

    # Custom config for detoxification
    detox_config = {
        "experiment_type": "detoxification",
        "epochs": 5,  # Reduced for demo
        "max_steps": 100,  # Limit steps for demo
        "eval_frequency": 25,
        "prompt_text": "I think",
        "save_model_path": "results/SLiM_detoxification_demo",
    }

    main(experiment_type="detoxification", custom_config=detox_config)


def run_emotion_steering_experiment():
    """Run emotion steering experiment with custom configuration."""
    print("=" * 60)
    print("RUNNING EMOTION STEERING EXPERIMENT")
    print("=" * 60)

    # Custom config for emotion steering
    emotion_config = {
        "experiment_type": "emotion_steering",
        "epochs": 3,  # Reduced for demo
        "max_steps": 50,  # Limit steps for demo
        "eval_frequency": 25,
        "prompt_text": "i feel",
        "save_model_path": "results/SLiM_emotion_steering_demo",
    }

    main(experiment_type="emotion_steering", custom_config=emotion_config)


def run_multi_state_steering_experiment():
    """Run multi-state steering experiment with custom configuration."""
    print("=" * 60)
    print("RUNNING MULTI-STATE STEERING EXPERIMENT")
    print("=" * 60)

    # Custom config for multi-state steering
    multi_state_config = {
        "experiment_type": "multi_state_steering",
        "epochs": 5,  # Reduced for demo
        "max_steps": 100,  # Limit steps for demo
        "eval_frequency": 25,
        "prompt_text": "I think",
        "save_model_path": "results/SLiM_multi_state_steering_demo",
    }

    main(experiment_type="multi_state_steering", custom_config=multi_state_config)


def run_language_steering_experiment():
    """Run language steering experiment with custom configuration."""
    print("=" * 60)
    print("RUNNING LANGUAGE STEERING EXPERIMENT")
    print("=" * 60)

    # Custom config for language steering
    language_config = {
        "experiment_type": "language_steering",
        "epochs": 5,  # Reduced for demo
        "max_steps": 100,  # Limit steps for demo
        "eval_frequency": 25,
        "prompt_text": "I think",
        "save_model_path": "results/SLiM_language_steering_demo",
    }

    main(experiment_type="language_steering", custom_config=language_config)


def run_topic_steering_experiment():
    """Run topic steering experiment with custom configuration."""
    print("=" * 60)
    print("RUNNING TOPIC STEERING EXPERIMENT")
    print("=" * 60)

    # Custom config for topic steering
    topic_config = {
        "experiment_type": "topic_steering",
        "epochs": 5,  # Reduced for demo
        "max_steps": 100,  # Limit steps for demo
        "eval_frequency": 25,
        "prompt_text": "I think",
        "save_model_path": "results/SLiM_topic_steering_demo",
    }

    main(experiment_type="topic_steering", custom_config=topic_config)


if __name__ == "__main__":
    print("SLiM Dynamic Training System")
    print("=" * 60)
    print("Available experiment types:")
    print("1. Detoxification")
    print("2. Emotion steering")
    print("3. Multi-state steering")
    print("4. Language steering")
    print("5. Topic steering")
    print("=" * 60)

    # You can uncomment one of these to run a specific experiment
    # run_detoxification_experiment()
    # run_emotion_steering_experiment()
    # run_multi_state_steering_experiment()
    # run_language_steering_experiment()
    # run_topic_steering_experiment()

    print("To run an experiment, uncomment one of the function calls above.")
    print(
        "Or modify the config.yaml file to set 'experiment_type' and run train.py directly."
    )
