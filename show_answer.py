from rl2023.answer_sheet import *

if __name__ == "__main__":
    print(question5_1())

    """
    DQN - linear decay
    Finished run with hyperparameters epsilon_start:1.0_exploration_fraction:0.75. Mean final score: 310.577 +- 32.79688345796831
    
    Finished run with hyperparameters epsilon_start:1.0_exploration_fraction:0.25. Mean final score: 308.193 +- 38.339959341832035
    
    Finished run with hyperparameters epsilon_start:1.0_exploration_fraction:0.01. Mean final score: 345.757 +- 26.298703603452054
    
    Exp decay
    Finished run with hyperparameters epsilon_start:1.0_epsilon_decay:1.0. Mean final score: 137.172 +- 30.13924152256581
    
    Finished run with hyperparameters epsilon_start:1.0_epsilon_decay:0.75. Mean final score: 172.07300000000004 +- 26.00318668719065
    
    Finished run with hyperparameters epsilon_start:1.0_epsilon_decay:0.001. Mean final score: 362.889 +- 23.151401755017382
    """

    """
    Reinforce
    
    Finished run with hyperparameters learning_rate:0.6. Mean final score: -500.0 +- 0.0
    
    
    """

    """
    DDPG
    
    Finished run with hyperparameters critic_hidden_size:[32, 32]_policy_hidden_size:[32, 32]. Mean final score: -141.32244440931768 +- nan
    
    Finished run with hyperparameters critic_hidden_size:[32, 32]_policy_hidden_size:[64, 64]. Mean final score: -185.6905460055932 +- nan
    
    Finished run with hyperparameters critic_hidden_size:[32, 32]_policy_hidden_size:[128, 128]. Mean final score: -137.68327410799276 +- nan
    
    Finished run with hyperparameters critic_hidden_size:[64, 64]_policy_hidden_size:[32, 32]. Mean final score: -127.81469187623803 +- nan
    
    Finished run with hyperparameters critic_hidden_size:[64, 64]_policy_hidden_size:[64, 64]. Mean final score: -124.42750264230479 +- nan
    """

    """
    Q4 - 128, 64, 32
    
    T:\IDE\Anaconda\envs\rl2023\python.exe T:/WorkSpace/Python-RL-Learn/rl2023/exercise4/train_ddpg.py
    T:\IDE\Anaconda\envs\rl2023\lib\site-packages\gym\core.py:317: DeprecationWarning: WARN: Initializing wrapper in old step API which returns one bool instead of two. It is recommended to set `new_step_api=True` to use new step API. This will be the default behaviour in future.
      deprecation(
    T:\IDE\Anaconda\envs\rl2023\lib\site-packages\gym\wrappers\step_api_compatibility.py:39: DeprecationWarning: WARN: Initializing environment in old step API which returns one bool instead of two. It is recommended to set `new_step_api=True` to use new step API. This will be the default behaviour in future.
      deprecation(
      5%|▌         | 20030/400000 [03:52<43:46, 144.67it/s]Evaluation at timestep 20030 returned a mean returns of -159.19508608853482
     10%|█         | 41405/400000 [08:08<41:02, 145.64it/s]Evaluation at timestep 41405 returned a mean returns of -105.79572994490435
     15%|█▌        | 60480/400000 [12:07<39:44, 142.37it/s]Evaluation at timestep 60480 returned a mean returns of -87.60848306084631
     20%|██        | 80083/400000 [16:09<37:00, 144.06it/s]Evaluation at timestep 80083 returned a mean returns of -146.80908181769908
     25%|██▌       | 101358/400000 [20:22<34:45, 143.22it/s]Evaluation at timestep 101358 returned a mean returns of -116.3915970595735
     30%|███       | 120558/400000 [24:24<34:36, 134.55it/s]Evaluation at timestep 120558 returned a mean returns of -112.4291036291743
     35%|███▌      | 141491/400000 [28:41<30:37, 140.68it/s]Evaluation at timestep 141491 returned a mean returns of -86.45559645843926
     40%|████      | 160588/400000 [31:15<27:09, 146.91it/s]Evaluation at timestep 160588 returned a mean returns of -82.03623415483857
     45%|████▌     | 181229/400000 [35:20<24:40, 147.80it/s]Evaluation at timestep 181229 returned a mean returns of -105.13636506879544
     50%|█████     | 201101/400000 [37:49<22:33, 146.90it/s]Evaluation at timestep 201101 returned a mean returns of -119.50535736242072
     55%|█████▌    | 220113/400000 [41:41<20:17, 147.70it/s]Evaluation at timestep 220113 returned a mean returns of -67.53748782547927
     60%|██████    | 240711/400000 [44:08<18:09, 146.22it/s]Evaluation at timestep 240711 returned a mean returns of -126.48649670968558
     65%|██████▌   | 260033/400000 [46:30<15:54, 146.62it/s]Evaluation at timestep 260033 returned a mean returns of -98.9568464269809
     70%|███████   | 280168/400000 [50:14<13:38, 146.41it/s]Evaluation at timestep 280168 returned a mean returns of -140.17485906655367
     75%|███████▌  | 300273/400000 [53:09<11:16, 147.52it/s]Evaluation at timestep 300273 returned a mean returns of -68.36519932651304
     80%|████████  | 320082/400000 [55:41<09:05, 146.43it/s]Evaluation at timestep 320082 returned a mean returns of -70.99315443723178
     85%|████████▌ | 340582/400000 [59:12<06:42, 147.47it/s]Evaluation at timestep 340582 returned a mean returns of 92.5527783270473
     90%|█████████ | 360691/400000 [1:01:50<04:27, 146.79it/s]Evaluation at timestep 360691 returned a mean returns of -56.12971582330105
     95%|█████████▌| 380028/400000 [1:04:35<02:15, 146.99it/s]Evaluation at timestep 380028 returned a mean returns of -46.2872863639572
    400487it [1:07:15, 99.23it/s] 
    Evaluation at timestep 400487 returned a mean returns of -36.80751898938511
    Saving to:  bipedal_q4_latest_256_128_64.pt
    
    Process finished with exit code 0
    """

    """
    Q4 - 256 - 128 - 64
    
    
    """

    """
    Q5
    "policy_learning_rate": 0.001,
    "critic_learning_rate": 0.01,
    "critic_hidden_size": [512, 256],
    "policy_hidden_size": [128, 64],
    "gamma": 0.99,
    "tau": 0.05,
    "batch_size": 64,
    "buffer_capacity": int(1e6),
    
      5%|▌         | 20180/400000 [03:13<1:02:06, 101.92it/s]Evaluation at timestep 20180 returned a mean returns of -120.68661196951321
     10%|█         | 41311/400000 [08:19<57:02, 104.79it/s]Evaluation at timestep 41311 returned a mean returns of -152.75433206631394
     15%|█▌        | 61381/400000 [13:13<53:58, 104.55it/s]Evaluation at timestep 61381 returned a mean returns of -126.30957223861911
     20%|██        | 80040/400000 [18:00<54:23, 98.04it/s]Evaluation at timestep 80040 returned a mean returns of -127.15745696018946
     25%|██▌       | 100096/400000 [23:00<51:23, 97.25it/s]Evaluation at timestep 100096 returned a mean returns of -111.31245713921216
     30%|███       | 121240/400000 [28:11<47:14, 98.34it/s]Evaluation at timestep 121240 returned a mean returns of -96.68243923076248
     35%|███▌      | 140050/400000 [31:28<44:27, 97.44it/s]Evaluation at timestep 140050 returned a mean returns of -103.81251472669904
     40%|████      | 160064/400000 [35:15<40:50, 97.89it/s]Evaluation at timestep 160064 returned a mean returns of -122.07149349404574
     45%|████▌     | 180277/400000 [40:17<37:15, 98.30it/s]Evaluation at timestep 180277 returned a mean returns of 10.77583355440257
     50%|█████     | 200680/400000 [44:01<35:06, 94.63it/s]Evaluation at timestep 200680 returned a mean returns of -70.91281600940457
     55%|█████▌    | 220042/400000 [47:35<30:34, 98.07it/s]Evaluation at timestep 220042 returned a mean returns of -117.06775787203969
     60%|██████    | 240402/400000 [51:13<27:07, 98.05it/s]Evaluation at timestep 240402 returned a mean returns of -79.77584254974002
     65%|██████▌   | 260891/400000 [55:41<23:33, 98.42it/s]Evaluation at timestep 260891 returned a mean returns of 86.34149860297778
     70%|███████   | 280117/400000 [59:18<20:41, 96.59it/s]Evaluation at timestep 280117 returned a mean returns of -62.35909512693972
     75%|███████▌  | 300510/400000 [1:04:33<17:35, 94.29it/s]Evaluation at timestep 300510 returned a mean returns of -128.91046796575256
     80%|████████  | 320094/400000 [1:08:00<13:40, 97.43it/s]Evaluation at timestep 320094 returned a mean returns of -93.75611479824389
     85%|████████▌ | 340344/400000 [1:12:02<10:13, 97.25it/s]Evaluation at timestep 340344 returned a mean returns of 34.24199216031719
     90%|█████████ | 360032/400000 [1:15:45<06:53, 96.76it/s]Evaluation at timestep 360032 returned a mean returns of -50.44645453609407
     95%|█████████▌| 380077/400000 [1:20:23<03:23, 97.83it/s]Evaluation at timestep 380077 returned a mean returns of 230.66787024982844
    400170it [1:23:55, 79.48it/s] 
    Evaluation at timestep 400170 returned a mean returns of -56.00242923915105
    Saving to:  bipedal_q5_latest.pt
    
    Process finished with exit code 0
    """