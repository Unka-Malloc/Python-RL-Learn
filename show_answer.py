from rl2023.answer_sheet import *

if __name__ == "__main__":
    print(len(question2_4()))

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