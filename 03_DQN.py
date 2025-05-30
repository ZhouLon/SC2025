# Copyright (C) 2025 [周龙/华南理工大学生物科学与工程学院] Email:1922450589@qq.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

#Additional terms:
# 作者联系方式:Email:1922450589@qq.com
# 1. 任何学术用途需与作者联系
# 2. 商业使用前需联系作者获得书面授权
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from Bio import SeqIO
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

restypes = {
    "A":0,
    "R":1,
    "N":2,
    "D":3,
    "C":4,
    "Q":5,
    "E":6,
    "G":7,
    "H":8,
    "I":9,
    "L":10,
    "K":11,
    "M":12,
    "F":13,
    "P":14,
    "S":15,
    "T":16,
    "W":17,
    "Y":18,
    "V":19,
}

def one_hot_encode(sequence, max_length=238):
    # 预分配张量
    encoding_long = torch.zeros((max_length, 20), dtype=torch.float32)
    encoding_short = torch.zeros((max_length,), dtype=torch.float32)

    # 一次性将序列转换为索引
    indices = [restypes.get(aa, -1) for aa in sequence[:max_length]]
    
    # 创建有效氨基酸的掩码
    valid_mask = torch.tensor([i != -1 for i in indices], dtype=torch.bool)
    valid_indices = torch.tensor([i for i in indices if i != -1], dtype=torch.long)
    positions = torch.arange(len(indices))[valid_mask]
    
    # 向量化赋值
    encoding_long[positions, valid_indices] = 1.0  #[238,20]
    encoding_short[positions] = valid_indices.float() #[238]
    
    return encoding_long, encoding_short

# 定义Q网络
class QNet(nn.Module):
    def __init__(self,EMPTY=10, seq_len=238, input_dim=20):
        super(QNet, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.EMPTY = EMPTY
        
        self.embedding_mlp = nn.Sequential(
            nn.Linear(seq_len*20, 8192),
            nn.ReLU(),
            nn.Linear(8192, 2048),
            nn.LayerNorm(2048),  
        )
        self.mlp_i = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, seq_len)  
        )
        self.mlp_j = nn.Sequential(
            nn.Linear(2048+seq_len, 8192),
            nn.ReLU(),
            nn.Linear(8192, 2048),
            nn.ReLU(),
            nn.Linear(2048, 20+EMPTY)  # 20种氨基酸 + EMPTY个空位（预测到空位则停止突变）
        )
        

    def forward(self, features_long):
        B, L, D = features_long.shape
        
        # 1. 氨基酸嵌入
        x = self.embedding_mlp(features_long.reshape(B,4760))  
        # 2. 计算突变对的概率
        i_logits = self.mlp_i(x)  # (B, L)
        j_logits = self.mlp_j(torch.cat([x,i_logits],dim=-1))  # (B, 2048+238) -> (B, 20) 把突变的位置告诉氨基酸突变头

        return i_logits, j_logits

#加载训练好的预训练模型
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.dropout=nn.Dropout(0.2)

        # Onehot Embedding层
        self.onehot_mlp=nn.Sequential(
            nn.Linear(238*20, 8192),
            nn.LayerNorm(8192),
            nn.ReLU(),
            self.dropout,
            nn.Linear(8192, 2048),
        )
        
        # 输出头
        self.out_mlp = nn.Sequential(
            nn.Linear(2048, 256),  
            nn.ReLU(),
            nn.Linear(256, 1),  
        )
    def forward(self, features_long):
        B, L, _=features_long.shape
        onehot_embedding=self.onehot_mlp(features_long.reshape(B,-1))
        x=self.out_mlp(onehot_embedding)
        return x

class DQN_Net(object):
    def __init__(self, EPSILON, MEMORY_CAPACITY,TARGET_NET_UPDATE, BATCH_SIZE, GAMMA,
                 EPSILON_DECAY, lr,EMPTY,MAX_HIGH_VALUE_MEMORY,EXPETATION,MIN_EPSILON):
        # 初始化策略网络和目标网络
        self.EPSILON = EPSILON
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.TARGET_NET_UPDATE = TARGET_NET_UPDATE
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.EPSILON_DECAY = EPSILON_DECAY
        self.MIN_EPSILON=MIN_EPSILON
        self.EMPTY=EMPTY
        self.strategy_net = QNet(EMPTY).to(device)# 策略网络（行为网络），负责当前动作选择
        self.target_net = QNet(EMPTY).to(device)# 目标网络，用于稳定训练目标，隔一段时间复制过来
        self.target_net.load_state_dict(self.strategy_net.state_dict()) # 初始化为相同参数
        self.learn_step = 0  # 初始化学习步数记录

        # 普通回放池
        self.memory_counter = 0  # 初始化回放池存储量
        self.memory = torch.zeros((self.MEMORY_CAPACITY, 4760*2+3), dtype=torch.float32).to(device)
        self.optimizer = torch.optim.Adam(self.strategy_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

        # 高价值回放池
        self.EXPETATION=EXPETATION  #高价值阈值
        self.MAX_HIGH_VALUE_MEMORY=MAX_HIGH_VALUE_MEMORY
        self.high_value_memory_counter=0 #计数器
        self.hv_memory_path='./results/high_value_memory.pt'
        self.high_value_memory = torch.zeros((MAX_HIGH_VALUE_MEMORY, 4760*2+3), dtype=torch.float32).to(device)
        self.high_value_counter_list=[]# 计数，查看高价值序列出现趋势

        #训练记录
        self.q_eval_record = []
        self.q_target_record = []
        self.reward_record = []
                
        

    def choose_action(self, x):
        # e-greedy动作选择:小于阈值，随机探索；大于阈值，贪婪策略
        if np.random.uniform() < self.EPSILON:
            i_action = torch.randint(low=0, high=238, size=(x.shape[0],), device=x.device)
            j_action = torch.randint(low=0, high=20+self.EMPTY, size=(x.shape[0],), device=x.device)
        else:
            i,j = self.strategy_net(x)
            i_probs = F.softmax(i,dim=-1)
            j_probs = F.softmax(j,dim=-1)  

            i_dist = torch.distributions.Categorical(i_probs)
            j_dist = torch.distributions.Categorical(j_probs)

            # 从概率分布里采样动作
            i_action = i_dist.sample()
            j_action = j_dist.sample()
        return i_action.unsqueeze(1),j_action.unsqueeze(1)

    def save_high_value_memory(self):
        # 只保存有效数据（非零部分）
        valid_data = self.high_value_memory[:self.high_value_memory_counter]
        torch.save({
            'transitions': valid_data,
        }, self.hv_memory_path)

    def store_transition(self, s, a_i, a_j, new_r, s_next,env):
        transitions = torch.cat([
            s.flatten(start_dim=1),      # [B, 238*20]
            s_next.flatten(start_dim=1), # [B, 238*20]
            a_i,                 # [B, 1]
            a_j,                 # [B, 1]
            new_r           # [B, 1]
        ], dim=1)                        # -> [B, 4760*2+3]
        
        # 批量存储
        batch_size = s.size(0)
        indices = torch.arange(
            self.memory_counter, 
            self.memory_counter + batch_size
        ) % self.MEMORY_CAPACITY
        
        self.memory[indices] = transitions.clone()  # 单次拷贝
        self.memory_counter += batch_size

        #存储高价值突变
        # 计算s_next的预测值
        s_next_value = env.valuenet(s_next)
        high_value_mask = (s_next_value >= self.EXPETATION).squeeze()
        # 存储高价值transition到特殊回放池
        if high_value_mask.any():
            high_value_transitions = transitions[high_value_mask]
            hv_batch_size = high_value_transitions.size(0)
            
            hv_indices = torch.arange(
                self.high_value_memory_counter,
                self.high_value_memory_counter + hv_batch_size
            ) % (self.MAX_HIGH_VALUE_MEMORY)
            self.high_value_memory[hv_indices] = high_value_transitions.clone()
            self.high_value_memory_counter += hv_batch_size
            
            if self.high_value_memory_counter>self.MAX_HIGH_VALUE_MEMORY:
                self.save_high_value_memory()
                exit()
        self.high_value_counter_list.append(self.high_value_memory_counter)

    def update_epsilon(self):
        # 自定义epsilon衰减函数，目的是随着训练的进行，智能体趋于利用已有知识，而非继续探索
        self.EPSILON = max(self.EPSILON * self.EPSILON_DECAY, self.MIN_EPSILON)

    def learn(self):
        # 如果到达目标网络更新轮数，将策略网络的参数复制给目标网络
        if self.learn_step % self.TARGET_NET_UPDATE == 0:
            self.target_net.load_state_dict(self.strategy_net.state_dict())
        self.learn_step += 1

        # 随机抽取BATCH_SIZE个历史经验数据
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)#抽取结果为数组
        b_memory = self.memory[sample_index, :]#选取抽取的结果，构成一个batch
        
        B=b_memory.shape[0]
        # 分别取出当前batch内的s、a、r和s_next的值
        b_s = b_memory[:, :4760].reshape(B,238,20)
        b_s_next = b_memory[:, 4760:4760*2].reshape(B,238,20)
        b_a_i = b_memory[:,-3].long()
        b_a_j = b_memory[:,-2].long()
        b_r = b_memory[:,  -1]


        # 计算 Q_eval (策略网络预测的 Q 值)
        q_eval_i, q_eval_j = self.strategy_net(b_s)  
        #选择实际执行的动作对应的 Q 值
        q_eval_i = q_eval_i.gather(1, b_a_i.unsqueeze(1))  # [B, 1]
        q_eval_j = q_eval_j.gather(1, b_a_j.unsqueeze(1))  # [B, 1]
        q_eval = (q_eval_i + q_eval_j) / 2  # 合并 Q 值（或自定义逻辑）

        #计算 Q_next (目标网络预测的 Q 值)
        with torch.no_grad():
            q_next_i, q_next_j = self.target_net(b_s_next)
            q_next = (q_next_i.max(1)[0] + q_next_j.max(1)[0]) / 2 

        # TD 目标
        q_target = b_r.unsqueeze(1) + self.GAMMA * q_next.unsqueeze(1)  # [B, 1]

        #记录训练
        self.q_eval_record.append(q_eval.mean().item())
        self.q_target_record.append(q_target.mean().item())
        self.reward_record.append(b_r.mean().item())

        # 计算损失
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Enviroment(object):
    def __init__(self):
        self.valuenet=RegressionModel()
        #加载预训练模型
        self.valuenet.load_state_dict(torch.load('./pth/eval_model.pt'))
        self.valuenet.to(device).eval()

    def step(self,s,i_action,j_action):
        B, L, _ = s.shape
        batch_idx = torch.arange(B, device=s.device)
        i_action,j_action=i_action.squeeze(1),j_action.squeeze(1)
        # 创建布尔张量表示j是否是氨基酸(0-19)
        mask = (j_action < 20)
        # 将原始氨基酸位置清零——符合mask的突变位点
        next_s = s.clone()
        next_s[batch_idx[mask], i_action[mask], :] = 0.0
        next_s[batch_idx[mask], i_action[mask], j_action[mask]] = 1.0
        with torch.no_grad():
            r=self.valuenet(next_s)-self.valuenet(s)#进化前后变化

        return next_s,r

if __name__ == '__main__':
    BATCH_SIZE = 128
    lr = 1e-4
    EPSILON = 1.0           # 贪婪阈值
    GAMMA = 0.8             # 折扣系数
    TARGET_NET_UPDATE = 50  # 目标网络更新频率
    # 定义epsilon的衰减因子
    EPSILON_DECAY = 0.999
    MIN_EPSILON=0.01  #最小的epsilon
    MEMORY_CAPACITY = 1000000  # 经验回放池大小，约占显存40G
    EMPTY=1
    NUM_EPOCHS=10000
    NUM_CYCLES=1
    MAX_HIGH_VALUE_MEMORY=100 #高亮度序列
    EXPETATION=50 #高价值阈值
    STEP=3 #突变探索步数
    BONUS=False #是否开启额外奖励


    dqn = DQN_Net(EPSILON, MEMORY_CAPACITY, TARGET_NET_UPDATE, BATCH_SIZE, GAMMA,
                  EPSILON_DECAY, lr,EMPTY,MAX_HIGH_VALUE_MEMORY,EXPETATION,MIN_EPSILON)
    env = Enviroment()
    reward_list = []
    max_reward_list=[]
    datas = []
    file='./inputs/待突变序列.fasta'
    for record in tqdm(SeqIO.parse(file, "fasta"),desc='fasta加载中...'):
        seq = str(record.seq).upper()
        datas.append(one_hot_encode(seq)[0].to(device))

    pbar = tqdm(range(NUM_EPOCHS), desc='Initializing')
    for i in pbar:
        s=datas[i%len(datas)].unsqueeze(0).repeat(128, 1, 1)
        # 定期更新epsilon
        dqn.update_epsilon()
        for k in range(STEP): #突变探索步数
            # 使用DQN网络选择一个动作a
            a_i,a_j = dqn.choose_action(s)
            # 执行动作a，获取下一个状态s_next，奖励r
            s_next, r= env.step(s,a_i,a_j)
            # 奖励
            new_r = r
            #额外奖励
            if BONUS:
                #暂时无奖励
                continue
            
            # 将状态转移元组存储到DQN的回放池中
            dqn.store_transition(s, a_i,a_j, new_r, s_next,env)
            reward_list.append(r.sum().item())
            max_reward_list.append(r.max().item())

            # 把突变作为下一次初始序列
            s=s_next

            # 如果回放池中的存储量超过了容量阈值，则调用DQN的学习方法，从回放池中抽取样本进行学习
            if dqn.memory_counter > MEMORY_CAPACITY:
                for _ in range(NUM_CYCLES):#对于一个经验池，进行重复采样训练
                    dqn.learn()

        # 更新描述和统计信息
        pbar.set_description(f'Training (EPSILON: {dqn.EPSILON:.4f}),高价值数: {dqn.high_value_memory_counter}')
        pbar.update(1)


torch.save(dqn.target_net.state_dict(),'./pth/dqn.pt')
dqn.save_high_value_memory()