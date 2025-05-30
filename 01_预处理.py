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
import openpyxl
import pandas as pd
from openpyxl.utils import column_index_from_string
import math
import copy
# 定义四种GFP的原始序列
GFP_SEQUENCES = {
    'avGFP': 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK',
    'amacGFP': 'MSKGEELFTGIVPVLIELDGDVHGHKFSVRGEGEGDADYGKLEIKFICTTGKLPVPWPTLVTTLSYGILCFARYPEHMKMNDFFKSAMPEGYIQERTIFFQDDGKYKTRGEVKFEGDTLVNRIELKGMDFKEDGNILGHKLEYNFNSHNVYIMPDKANNGLKVNFKIRHNIEGGGVQLADHYQTNVPLGDGPVLIPINHYLSCQTAISKDRNETRDHMVFLEFFSACGHTHGMDELYK',
    'ppluGFP': 'MPAMKIECRITGTLNGVEFELVGGGEGTPEQGRMTNKMKSTKGALTFSPYLLSHVMGYGFYHFGTYPSGYENPFLHAINNGGYTNTRIEKYEDGGVLHVSFSYRYEAGRVIGDFKVVGTGFPEDSVIFTDKIIRSNATVEHLHPMGDNVLVGSFARTFSLRDGGYYSFVVDSHMHFKSAIHPSILQNGGPMFAFRRVEELHSNTELGIVEYQHAFKTPIAFA',
    'cgreGFP': 'MTALTEGAKLFEKEIPYITELEGDVEGMKFIIKGEGTGDATTGTIKAKYICTTGDLPVPWATILSSLSYGVFCFAKYPRHIADFFKSTQPDGYSQDRIISFDNDGQYDVKAKVTYENGTLYNRVTVKGTGFKSNGNILGMRVLYHSPPHAVYILPDRKNGGMKIEYNKAFDVMGGGHQMARHAQFNKPLGAWEEDYPLYHHLTVWTSFGKDPDDDETDHLTIVEVIKAVDLETYR'
}

def read_excel_data(file_path, sheet_name=None, columns=('A', 'B', 'C')):
    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook[sheet_name] if sheet_name else workbook.active

    data = []
    for row_idx, row in enumerate(sheet.iter_rows(min_row=2, values_only=True), start=2):
        mut_desc, gfp_type, activity = [row[column_index_from_string(col) - 1] for col in columns]
        original_seq = GFP_SEQUENCES.get(gfp_type)
        Valid = True
        if mut_desc =='WT':
            new_seq=original_seq
        else :
            mutation_parts = mut_desc.split(':')
            position=[]
            original_seq_list = list(original_seq)
            for part in mutation_parts:
                start_aa=part[0]
                posi=int(part[1:-1])
                last_aa = part[-1]
                if start_aa =='.' or last_aa =='.':
                    Valid=False
                    break
                if start_aa =='*' or last_aa =='*':
                    Valid=False
                    break
                if posi not in position:
                    position.append(posi)
                else:
                    Valid=False
                    break
                if original_seq_list[posi] == start_aa:
                    original_seq_list[posi] = last_aa
                else:
                    Valid=False
                    break
            new_seq = ''.join(original_seq_list)
        if Valid:
            data.append((new_seq,activity,math.pow(math.e,activity), gfp_type,mut_desc))

    return pd.DataFrame(data, columns=['Sequence',  'Activity_log','Activity','GFP_Type', 'Mutation'])


def read_tsv_data(file_path):
    """
    读取TSV格式的突变数据并处理

    参数:
        file_path: TSV文件路径
        reference_sequence: 野生型参考序列

    返回:
        包含处理后的数据的DataFrame
    """
    # 读取TSV文件
    df = pd.read_csv(file_path, sep='\t')

    data = []

    for _, row in df.iterrows():
        aa_mutations = row['aaMutations']
        median_brightness = row['medianBrightness']
        reference_sequence=copy.deepcopy(GFP_SEQUENCES['avGFP'])
        valid = True
        new_seq = reference_sequence

        if aa_mutations == 'WT':
            # 野生型，直接使用参考序列
            new_aa_mutations_seq ='WT'
            pass
        else:
            # 处理突变描述
            mutation_parts = aa_mutations.split(':')
            positions = []
            original_seq_list = list(reference_sequence)
            new_aa_mutations=[]
            for part in mutation_parts:
                # 解析突变描述，例如SA108D
                orig_aa = part[1]  # 原始氨基酸 (S)
                position = int(part[2:-1])+1  # 突变位置+1，是因为该TSV文件中，起始密码子M消失了
                new_aa = part[-1]  # 新氨基酸 (D)
                new_aa_mutations.append(orig_aa+str(int(part[2:-1])+1)+new_aa)
                # print(orig_aa,position,new_aa)
                # 验证突变描述是否有效
                if orig_aa == '.' or new_aa == '.':
                    valid = False
                    break
                if orig_aa == '*' or new_aa == '*':
                    valid = False
                    break
                if position in positions:
                    valid = False  # 重复突变位置
                    break

                positions.append(position)

                # 检查参考序列中该位置的氨基酸是否匹配
                if original_seq_list[position] != orig_aa:  # 注意Python是0-based索引
                    print('not e')
                    valid = False
                    break

                # 应用突变
                original_seq_list[position] = new_aa
            new_aa_mutations_seq=':'.join(new_aa_mutations)
            new_seq = ''.join(original_seq_list)

        if valid:
            # 计算指数转换后的亮度值
            brightness_exp = math.pow(math.e, median_brightness)

            data.append({
                'Sequence': new_seq,
                'Brightness_log': median_brightness,
                'Brightness': brightness_exp,
                'gfp_type': 'avGFP',
                'Mutation': new_aa_mutations_seq
            })

    return pd.DataFrame(data)



if __name__ == "__main__":
    #读取Excel数据
    # input_file = "GFP data.xlsx"  # 输入Excel文件
    # output_csv = "GFP_data_processed.csv"  # 输出处理后的DataFrame
    #
    # # 生成DataFrame并保存
    # df = read_excel_data(input_file)
    # df.to_csv(output_csv, index=False)
    # print(f"Processed data saved to {output_csv}")

    #读取TSV文件
    # 读取数据
    input_file = 'amino_acid_genotypes_to_brightness.tsv'  # 输入TSV文件
    output_csv = "avGFP_from_essay.csv"  # 输出处理后的DataFrame
    df = read_tsv_data(input_file)
    df.to_csv(output_csv, index=False)
    print(f"Processed data saved to {output_csv}")