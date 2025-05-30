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
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
def csv_to_fasta_biopython(csv_file, fasta_file):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)
    # 用于存储已见过的序列和对应的信息
    seen_sequences = {}
    # 创建 SeqRecord 对象列表
    records = []
    for i, row in df.iterrows():
        sequence = row['Sequence']
        # 如果序列已经存在，跳过添加
        if sequence in seen_sequences:
            print(sequence,row['GFP_Type'],row['Mutation'],row['Activity'])
            continue
        # 记录这个序列
        seen_sequences[sequence] = True

        seq = Seq(sequence)
        seq_id = f"{i}-{row['GFP_Type']}-{row['Activity']}-{row['Activity_log']}"

        # 创建 SeqRecord 对象
        record = SeqRecord(
            seq,
            id=seq_id,
            description=f"{row['Mutation']}"
        )
        records.append(record)

    # 写入 FASTA 文件
    SeqIO.write(records, fasta_file, "fasta")
    return len(records)  # 返回写入的唯一序列数量


if __name__ == "__main__":
    input_csv = "./nMut/GFP_1Mut.csv"  # 输入 CSV 文件
    output_fasta = "./nMut/GFP_GFP_1Mut.fasta"  # 输出 FASTA 文件

    unique_count = csv_to_fasta_biopython(input_csv, output_fasta)
    print(f"FASTA file saved to {output_fasta}")

#tape-embed unirep ./data/origin/GFP_sequences_tape.fasta ./data/output/GFP_sequences_tape.npz babbler-1900 --tokenizer unirep --batch_size 256 --gpu 0
