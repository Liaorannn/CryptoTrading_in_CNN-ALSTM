# CryptoTrading_in_CNN-ALSTM
> HKUST MAFM AI_Finance course project.


### Useful Links:
- [Rainbow DQN](https://paperswithcode.com/method/rainbow-dqn)
- [Strategy Demo](https://github.com/aifin-hkust/aifin-hkust.github.io/tree/master/2021/project3/demo)
- [PPO Tricks](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- [Stable Baseline 3](https://araffin.github.io/post/sb3/)
- [Stabel Basiline 3 Instruction](https://stable-baselines3.readthedocs.io/en/master/index.html)



### DeepLearning
***
**Model:**
- ALSTM: attention + LSTM
- padding-maske + CNN + Attention + Dense-net


#### Data
***

4 * 5 : [BTC, ETH, LTC, XRP] * [0 H L C V]

Testing Data set: 
	20160 data; 
	14 day; 
	2023-04-19--2023-05-02; 
	0 error days
Training Data set: 
	838079 data; 
	582 day; 
	2021-09-14--2023-04-18; 
	1 error days

#### Model
***
1. Whole model, 4 channel bitcoin, output 4 channel signal;
2. 4 model: each predict 1 bitcoin signal
3. Volume model: predict future volume
### Reinforcement Learning
***
**State:**
- 6 ohlcvt, postion-value, cash-value, total-value, average-price?

**Action:**
- 1: 20action (-10, 10)  
- 2: position

**Rewards**
- SharpRatio + Punishment of trading Volume
- Risk aversion function

**Model:**
- DQN
- PPO
- DDPG


### TODO List
- [x] Check the data form
- [x] Dataset Generator
  - [x] Factors
  - [x] Labels
  - 
- [x] ALSTM Training
- [ ] RL Environment
- [ ] PPO
- [ ] DDPG
