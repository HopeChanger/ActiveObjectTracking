# Multi-Target Active Object Tracking with Monte Carlo Tree Search and Target Motion Modeling
This repository is the python implementation of Multi-Target Active Object Tracking with Monte Carlo Tree Search and Target Motion Modeling.

## 2D Environments
![Env](https://github.com/HopeChanger/ActiveObjectTracking/images/env.jpg)

## Training
### Train the Multi-Agent Network
For example, if you want to save the model to `dir` "results/test", and use the football environment with 6 cameras.

You can use the following command:
```
python3 src/main.py --config=iql with use_tensorboard=True save_model=True evaluate=False tau=0.1 local_results_path="results/test" config_path="./settings/zq-PoseEnvBase.json"
```

## Evaluation
After training the model, you can test the effect of each module.
### MARL
MARL only, without MCTS and prediction.
```
python3 src/main.py --config=iql with checkpoint_path="./results/test/models/***" config_path="./settings/zq-PoseEnvBase.json" evaluate_with_mcts=False fixed_test_data=True
```

### Ours-
Use MCTS without prediction.
```
python3 src/main.py --config=iql with checkpoint_path="./results/test/models/***" config_path="./settings/zq-PoseEnvBase.json" predict_pos=False evaluate_with_mcts=True fixed_test_data=True
```

### Ours
Use MCTS and prediction.
```
python3 src/main.py --config=iql with checkpoint_path="./results/test/models/***" config_path="./settings/zq-PoseEnvBase.json" predict_pos=True evaluate_with_mcts=True fixed_test_data=True
```

## Citation
