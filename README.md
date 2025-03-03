# Optimizing Camera Motion with MCTS and Target Motion Modeling in Multi-Target Active Object Tracking
This repository is the python implementation of Optimizing Camera Motion with MCTS and Target Motion Modeling in Multi-Target Active Object Tracking.

## 2D Environments
6 environments with different parameters.

| Environment name | Camera nums | Target nums | Field size | File name |
|:-----:|:-----:|:-----:|:-----:|:-----:|
| Volleyball_A | 6 | 12 | 2400*1200 | pq-PoseEnvBase.json |
| Basketball_A | 6 | 10 | 2240*1200 | lq-PoseEnvBase.json |
| Football_A | 6 | 22 | 2100*1360 | zq-PoseEnvBase.json |
| Volleyball_B | 4 | 12 | 2400*1200 | pq-4cam-PoseEnvBase.json |
| Basketball_B | 4 | 10 | 2240*1200 | lq-4cam-PoseEnvBase.json |
| Football_B | 4 | 22 | 2100*1360 | zq-4cam-PoseEnvBase.json |

![image](https://github.com/HopeChanger/ActiveObjectTracking/blob/master/render/output.jpg)

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
