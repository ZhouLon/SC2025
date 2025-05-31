# Copyright (C) 2025 [周龙/华南理工大学生物科学与工程学院]
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

# Additional terms:
# Contact: Email: 1922450589@qq.com
# For any academic use, please contact the author
# Prior written authorization from the author is required for commercial use
# 作者联系方式:Email:1922450589@qq.com
# 任何学术用途需与作者联系
# 商业使用前需联系作者获得书面授权
import torch
from torch import nn
import pandas as pd

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
restypes = {
    "A": 0,
    "R": 1,
    "N": 2,
    "D": 3,
    "C": 4,
    "Q": 5,
    "E": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "L": 10,
    "K": 11,
    "M": 12,
    "F": 13,
    "P": 14,
    "S": 15,
    "T": 16,
    "W": 17,
    "Y": 18,
    "V": 19,
}


def inspect_high_value_memory(file_path):
    try:
        data = torch.load(file_path)
        print(f"文件包含 {len(data['transitions'])} 条记录")
        print("第一条记录的形状:", data['transitions'][0].shape)

        transitions = data['transitions']
        long = transitions[:, 4760:4760 * 2].reshape(-1, 238, 20)

    except Exception as e:
        print("读取失败:", str(e))
    return long
def onehot_to_sequence(onehot, knowledge):
    """将one-hot编码转换回序列字符串"""
    amino_acids = list(restypes.keys())
    seq = []
    for pos in range(onehot.shape[1]):
        aa_idx = torch.argmax(onehot[0, pos, :]).item()
        seq.append(amino_acids[aa_idx])

    for k in knowledge:
        o_aa = k[0]
        pos = int(k[1:-1]) - 1
        a_aa = k[-1]
        if o_aa == seq[pos]:
            seq[pos] = a_aa

    return ''.join(seq)

def compare_with_wildtype(mutant_seq, wildtype_seq):
    """比较突变体与野生型序列的差异"""
    differences = []
    for i, (wt_aa, mut_aa) in enumerate(zip(wildtype_seq, mutant_seq)):
        if wt_aa != mut_aa:
            differences.append(f"{wt_aa}{i + 1}{mut_aa}")
        if len(differences) > 6:  # 长度超出六就跳过
            return False
    return ":".join(differences)


class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.dropout = nn.Dropout(0.2)
        # Onehot Embedding层
        self.onehot_mlp = nn.Sequential(
            nn.Linear(238 * 20, 8192),
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
        B, L, _ = features_long.shape
        onehot_embedding = self.onehot_mlp(features_long.reshape(B, -1))
        x = self.out_mlp(onehot_embedding)
        return x


valuenet = RegressionModel()
valuenet.load_state_dict(torch.load('./pth/eval_model.pt'))
valuenet.to(device).eval()

# 使用示例
long = inspect_high_value_memory("./results/high_value_memory.pt")
bri = valuenet(long)
# 加载野生型序列
wildtype_seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

# 创建DataFrame来存储结果
results = {
    'Brightness': [],
    'Mutations': [],
    'Sequence': [],
}
knowledge = ['S65T']
# 处理每个高价值序列
for i in range(len(long)):
    # 将one-hot转换回序列
    seq = onehot_to_sequence(long[i:i + 1], knowledge)
    # 计算与野生型的差异
    mutations = compare_with_wildtype(seq, wildtype_seq)
    if mutations == False:
        continue
    # 获取亮度值
    brightness = bri[i].item()
    # 记录结果
    results['Brightness'].append(round(brightness, 4))
    results['Mutations'].append(mutations)
    results['Sequence'].append(seq)

# 创建DataFrame并保存为CSV
df = pd.DataFrame(results)
# 按序列去重（保留最高亮度的记录）,按亮度降序排列
df = df.sort_values('Brightness', ascending=False).drop_duplicates('Sequence')
print(f"有 {len(df)} 条高价值序列")

# 和官方筛选库进行比较筛选
print('接下来筛选序列是否在官方数据库')
df_del = pd.read_csv('./inputs/Exclusion_List.csv', names=['sequences-not-submit'], skiprows=1)
df = df[~df['Sequence'].isin(df_del['sequences-not-submit'])]

# 重新按亮度排序
df = df.sort_values('Brightness', ascending=False)

csv_path = './results/high_value_mutations.csv'
df.to_csv(csv_path, index=False)

print(f"结果已保存到 {csv_path}")
print(f"共得到了 {len(df)} 条高价值序列")
print("亮度最高的条序列:")
print(df.head(20)[['Brightness', 'Mutations']])