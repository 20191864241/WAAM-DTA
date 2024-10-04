import esm
import numpy as np
from collections import OrderedDict
import json
import torch


esm1b, esm1b_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
# esm1b, esm1b_alphabet = esm.pretrained.esm1_t6_43M_UR50S()
esm1b = esm1b.eval().cpu()
esm1b_batch_converter = esm1b_alphabet.get_batch_converter()

# 使用一个空字符串作为标识符，将输入字符串包装成一个元组
def convert_to_tuples(seq_string: str):
    return [('', seq_string)]

def protein_embedding(prot_list, dataset):
    esm1b_contacts, token_representations = {}, {}
    i = 0
    for prot in prot_list:
        print("running at: " + str(i))
        prot = convert_to_tuples(prot)
        _, _, esm1b_batch_tokens = esm1b_batch_converter(prot)
        b = esm1b_batch_tokens.shape[1:3][0]

        if b > 1024:
            tokens = esm1b_batch_tokens[:, 0:1024]
        else:
            tokens = torch.nn.functional.pad(esm1b_batch_tokens, pad=(0, 1024 - b, 0, 0), mode='constant', value=1)

        tokens = torch.tensor(tokens.numpy())
        with torch.no_grad():
            tokens=tokens.cpu()
            results = esm1b(tokens, repr_layers=[33], return_contacts=True)
            token_representations[i] = results["representations"][33].cpu()# 特征
            contacts = esm1b.predict_contacts(tokens).cpu()# 特征图（adj）
        d=torch.tensor(np.zeros([contacts.shape[0],1022])).unsqueeze(2)
        e=torch.cat([d,contacts,d],dim=2)
        d=torch.tensor(np.zeros([contacts.shape[0],1024])).unsqueeze(1)
        esm1b_contacts[i] = torch.cat([d,e,d],dim=1)
        print(i,token_representations[i].shape,esm1b_contacts[i].shape)
        np.savez('data/node/' + dataset +'.npz', dict=token_representations)
        np.savez('data/edge/' + dataset +'.npz', dict=esm1b_contacts)
        i += 1
    # torch.cuda.empty_cache()


def generate_protein_pretraining_representation(dataset, prots):
    data_dict = {}  # 数据字典
    prots_tuple = [(str(i), prots[i][:1022]) for i in range(len(prots))]  # 创建蛋白质元组
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()  # 加载transformer模型和字母表
    batch_converter = alphabet.get_batch_converter()  # 获取批处理转换器
    i = 0
    batch = 1

    while (batch*i) < len(prots):  # 循环处理蛋白质
        print('converting protein batch: '+ str(i))  # 打印转换蛋白质批次信息
        if (i + batch) < len(prots):  # 判断是否有下一个批次
            pt = prots_tuple[batch*i:batch*(i+1)]  # 获取当前批次的蛋白质元组
        else:
            pt = prots_tuple[batch*i:]  # 获取剩余的蛋白质元组

        batch_labels, batch_strs, batch_tokens = batch_converter(pt)  # 批量转换蛋白质
        #model = model.cuda()
        #batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)  # 获取结果
        token_representations = results["representations"][33].numpy()  # 提取表示
        data_dict[i] = token_representations  # 存储表示
        i += 1
    np.savez('data\DTC.npz', dict=data_dict)  # 保存数据字典为npz文件


dataset = 'kiba'
proteins = json.load(open('data/proteins_'+ dataset +'.txt'), object_pairs_hook=OrderedDict)  # 加载蛋白质
prots = []  # 蛋白质列表
for t in proteins.keys():
    prots.append(proteins[t])
protein_embedding(prots, dataset)
# generate_protein_pretraining_representation(dataset, prots)



