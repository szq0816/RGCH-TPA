import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.clip_model.model import load_download_clip, Transformer


class MLPLayer(nn.Module):
    """
    LND - LND or ND - ND
    """

    def __init__(self, dim_list, dropout=0., activation='relu'):
        super().__init__()

        if activation == 'relu':
            self.activation_layer = nn.ReLU()
        elif activation == 'gelu':
            self.activation_layer = nn.GELU()
        else:
            pass

        self.mlp = nn.Sequential()

        for i in range(len(dim_list) - 2):
            _in = dim_list[i]
            _out = dim_list[i + 1]

            self.mlp.add_module(f"linear_{i}", nn.Linear(_in, _out))
            self.mlp.add_module(f"activate_{i}", self.activation_layer)
            self.mlp.add_module(f"dropout_{i}", nn.Dropout(p=dropout))

        self.mlp.add_module(f"linear_final", nn.Linear(dim_list[-2], dim_list[-1]))

    def forward(self, x):
        return self.mlp(x)


class ResidualMLPs(nn.Module):
    """
    Residual MLPs
    ***D - ***D
    """

    def __init__(self, org_dim, hidden_dim, dropout=0., num_layers=2, activation='relu'):
        super().__init__()
        self.num_layers = num_layers

        if activation == 'relu':
            self.activation_layer = nn.ReLU()
        elif activation == 'gelu':
            self.activation_layer = nn.GELU()
        else:
            pass

        self.mlps = nn.ModuleList(nn.Sequential(
            nn.Linear(org_dim, hidden_dim),
            self.activation_layer,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, org_dim),
        ) for i in range(num_layers))

        self.lns = nn.ModuleList(nn.LayerNorm(org_dim) for i in range(num_layers))

    def forward(self, x):
        for i in range(self.num_layers):
            x = x + self.mlps[i](self.lns[i](x))
        return x


class HashingEncoder(nn.Module):
    """
    hashing encoder, linear projection & tach.
    """

    def __init__(self, org_dim, k_bits, ):
        super().__init__()
        self.fc = nn.Linear(org_dim, k_bits)

    def forward(self, x):
        return torch.tanh(self.fc(x))


class HashingDecoder(nn.Module):
    """
    hashing decoder, MLP & tach.
    """

    def __init__(self, org_bit_dim, recon_bit_dim):
        super().__init__()
        self.mlp = MLPLayer(dim_list=[org_bit_dim, recon_bit_dim, recon_bit_dim])

    def forward(self, x):
        return torch.tanh(self.mlp(x))


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class HashingModel(nn.Module):
    """
    Hashing model
    """

    def __init__(self, clip_info=None, args=None):
        super().__init__()

        self.auxiliary_bit_dim = auxiliary_bit_dim = args.auxiliary_bit_dim
        
        clip_embed_dim = clip_info['embed_dim']

        self.k_bits_list = list(map(int, args.k_bits_list.split(",")))  # str -> list

        self.extend_bits_list = []
        self.extend_bits_list.extend(self.k_bits_list)
        self.extend_bits_list.append(self.auxiliary_bit_dim)

        # share weight.
        self.hash_encoders = nn.ModuleList(
            HashingEncoder(org_dim=clip_embed_dim, k_bits=one)
            for one in self.extend_bits_list
        )
        # share weight.
        self.hash_decoders = nn.ModuleList(
            HashingDecoder(one, auxiliary_bit_dim)
            for one in self.k_bits_list
        )

    def forward(self, img_cls, txt_eos):
        output_dict = {}

        output_dict['img_cls_hash'] = {}
        output_dict['txt_cls_hash'] = {}

        output_dict['img_cls_hash_recon'] = {}
        output_dict['txt_cls_hash_recon'] = {}

        for i, one in enumerate(self.extend_bits_list):
            img_cls_hash = self.hash_encoders[i](img_cls)
            txt_cls_hash = self.hash_encoders[i](txt_eos)

            output_dict['img_cls_hash'][one] = img_cls_hash
            output_dict['txt_cls_hash'][one] = txt_cls_hash

            if one != self.auxiliary_bit_dim:
                img_cls_hash_recon = self.hash_decoders[i](img_cls_hash)
                txt_cls_hash_recon = self.hash_decoders[i](txt_cls_hash)
                output_dict['img_cls_hash_recon'][one] = img_cls_hash_recon
                output_dict['txt_cls_hash_recon'][one] = txt_cls_hash_recon

        return output_dict


class CMCL(nn.Module):
    def __init__(self, args=None):
        super(CMCL, self).__init__()
        self.args = args
        self.clip, clip_info = load_download_clip(self.args.clip_path)

        # freeze CLIP
        if self.args.is_freeze_clip:
            for n, p in self.clip.named_parameters():
                p.requires_grad = False

        self.completion = RetrievalGuidedCompletion(dim=clip_info['embed_dim'], topk=4)

        self.alignment = AlignmentModule(dim=clip_info['embed_dim'], lambda_rec=1.0)

        self.hash = HashingModel(clip_info=clip_info, args=args)

    def forward(self, image, text, key_padding_mask, m1, m2, memoryBank=None):
        img_tokens, _, img_cls = self.clip.encode_image(image)
        txt_tokens, _, new_key_padding_mask, txt_eos = self.clip.encode_text(text, key_padding_mask)

        img_seq = torch.cat([img_cls.unsqueeze(1), img_tokens.transpose(0, 1)], dim=1)  # [B, L_i+1, D]
        txt_seq = torch.cat([txt_eos.unsqueeze(1), txt_tokens.transpose(0, 1)], dim=1)  # [B, L_t+1, D]

        completed_image, completed_text, gen_image_full, gen_text_full = self.completion(img_seq, txt_seq, m1, m2,
                                                                                         memoryBank)
        img_cls_c = completed_image[:, 0, :]  # [B, D]
        img_tokens_c = completed_image[:, 1:, :]  # [B, L_i, D]
        txt_eos_c = completed_text[:, 0, :]  # [B, D]
        txt_tokens_c = completed_text[:, 1:, :]  # [B, L_t, D]

        f_img, f_txt, loss_pt_align, loss_rec, res_img_cls, res_txt_cls = self.alignment(img_cls_c, txt_eos_c, img_tokens_c, txt_tokens_c)

        output_dict = self.hash(f_img, f_txt)
        output_dict['img_seq'] = img_seq  # 存储库使用
        output_dict['txt_seq'] = txt_seq
        # 计算补全损失
        output_dict['completed_image'] = completed_image
        output_dict['completed_text'] = completed_text
        output_dict['gen_image_full'] = gen_image_full
        output_dict['gen_text_full'] = gen_text_full
        # 局部对齐损失
        output_dict['loss_pt_align'] = loss_pt_align
        output_dict['loss_rec'] = loss_rec
        output_dict['res_img_cls'] = res_img_cls
        output_dict['res_txt_cls'] = res_txt_cls
        return output_dict


class AlignmentModule(nn.Module):
    def __init__(self, dim, lambda_rec=1.0, topk=1):
        super().__init__()
        self.dim = dim
        self.topk = topk
        self.lambda_rec = lambda_rec

        # 文本→图像 patch 特征重建 MLP
        self.reconstruct_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2, dim)
        )

        # 对齐特征融合
        self.fusion_img = nn.Linear(dim * 2, dim)
        self.fusion_txt = nn.Linear(dim * 2, dim)

        self.resmlp_i = self.resmlp_t = ResidualMLPs(org_dim=dim, hidden_dim=4 * dim, dropout=0.5, num_layers=2, activation="gelu")

    def forward(self, img_cls_c, txt_eos_c, img_tokens_c, txt_tokens_c):

        img_cls_c = self.resmlp_i(img_cls_c)
        txt_eos_c = self.resmlp_t(txt_eos_c)
        res_img_cls = F.normalize(img_cls_c, dim=-1)
        res_txt_cls = F.normalize(txt_eos_c, dim=-1)

        B, Li, D = img_tokens_c.shape
        Lt = txt_tokens_c.shape[1]

        S = torch.bmm(txt_tokens_c, img_tokens_c.transpose(1, 2))

        A = F.softmax(S, dim=-1)  # [B, Lt, Li]

        P_aligned = torch.bmm(A, img_tokens_c)  # [B, Lt, D]

        P_recon = self.reconstruct_mlp(txt_tokens_c)  # [B, Lt, D]

        # 重建损失
        loss_rec = F.l1_loss(P_recon, P_aligned)

        # s_pos: [B, Lt]
        s_pos, _ = torch.max(S, dim=-1)

        # 负样本（soft negative mining）
        s_neg = S.reshape(B * Lt, Li)
        _, neg_idx = torch.topk(s_neg, k=1, dim=-1, largest=False)
        neg_idx = neg_idx.reshape(B, Lt)

        # 获取负例分数
        s_neg_val = torch.gather(S, dim=-1, index=neg_idx.unsqueeze(-1)).squeeze(-1)

        loss_pt_align = - torch.log(torch.sigmoid(s_pos) + 1e-6).mean() \
                        - torch.log(1 - torch.sigmoid(s_neg_val) + 1e-6).mean()

        txt_local_fused = torch.cat([txt_tokens_c, P_aligned], dim=-1)  # [B, Lt, 2D]
        img_local_fused = torch.cat([img_tokens_c, torch.bmm(A.transpose(1, 2), txt_tokens_c)], dim=-1)

        # 融合降维
        txt_aligned = self.fusion_txt(txt_local_fused)      # [B, Lt, D]
        img_aligned = self.fusion_img(img_local_fused)      # [B, Li, D]

        f_img = img_cls_c + img_aligned.mean(dim=1)
        f_txt = txt_eos_c + txt_aligned.mean(dim=1)

        return f_img, f_txt, loss_pt_align, loss_rec, res_img_cls, res_txt_cls


class MACRouter(nn.Module):
    def __init__(self, hs=512) -> None:
        super().__init__()
        self.linear_rem = nn.Linear(hs, hs)  # 剩余模态特征投影
        self.linear_ret = nn.Linear(hs, hs)  # 检索特征投影

    def forward(self, rem_fea, ret_fea):
        """
        计算路由分数：基于剩余模态特征和检索到的专家特征
        Args:
            rem_fea: [B, S, hs]，剩余模态特征
            ret_fea: [B, K, S, hs]，检索到的topk专家特征（K为专家数量）
        Returns:
            routing_score: [B, K]，归一化的路由分数
        """
        # 特征投影
        rem_fea = self.linear_rem(rem_fea)  # [B, S, hs]
        ret_fea = self.linear_ret(ret_fea)  # [B, K, S, hs]

        # 平均序列维度
        avg_rem_fea = rem_fea.mean(dim=1)  # [B, hs]
        avg_ret_fea = ret_fea.mean(dim=2)  # [B, K, hs]

        # 计算路由分数
        routing_score = torch.matmul(avg_rem_fea.unsqueeze(1), avg_ret_fea.transpose(2, 1))  # [B, 1, K]
        return nn.Softmax(dim=1)(routing_score.squeeze(1))  # [B, K]


class CMoEGenerator(nn.Module):
    def __init__(self, k, hs=512):
        super().__init__()
        self.router = MACRouter(hs=hs)  # 路由模块
        self.num_expert = k  # 专家数量（与topk匹配）
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hs, hs),
                nn.ReLU(),
                nn.Linear(hs, hs)
            ) for _ in range(self.num_expert)
        ])

    def forward(self, rem_fea, ret_fea):
        """
        生成缺失模态特征：路由分数加权融合专家输出
        Args:
            rem_fea: [B, S, hs]，剩余模态特征
            ret_fea: [B, K, S, hs]，检索到的topk专家特征
        Returns:
            gen_fea: [B, S, hs]，生成的缺失模态特征
        """
        # 1. 计算路由分数
        routing_score = self.router(rem_fea, ret_fea)  # [B, K]

        # 2. 专家网络处理检索特征
        expert_outs = []
        for k in range(self.num_expert):
            # 第k个专家处理第k个检索特征 [B, S, hs]
            expert_out = self.experts[k](ret_fea[:, k])
            expert_outs.append(expert_out)
        expert_outs = torch.stack(expert_outs, dim=1)  # [B, K, S, hs]

        # 3. 路由分数加权融合（k维度加权）
        gen_fea = torch.einsum('bksd,bk->bsd', expert_outs, routing_score)  # [B, S, hs]
        return gen_fea


class RetrievalGuidedCompletion(nn.Module):
    def __init__(self, dim=512, topk=4):
        super().__init__()
        self.dim = dim
        self.topk = topk  # 检索专家数量

        # 检索器：基于余弦相似度
        self.cos_sim = nn.CosineSimilarity(dim=-1)

        # CMoE生成器：分别处理文本缺失和图像缺失
        self.text_generator = CMoEGenerator(k=topk, hs=dim)  # 文本缺失时生成文本特征
        self.image_generator = CMoEGenerator(k=topk, hs=dim)  # 图像缺失时生成图像特征

    def retrieve_similar(self, query_feat, memory_feat):
        """
        检索与查询特征格式一致的「全局-局部拼接专家特征」
        Args:
            query_feat: [B, S_query, dim] → 查询的拼接特征（剩余模态）
            memory_feat: [train_num, S_mem, dim] → 记忆库中的完备拼接特征
        Returns:
            topk_ret_fea: [B, topk, S_mem, dim] → 检索到的专家拼接特征
        """
        batch_size = query_feat.shape[0]
        # 用查询特征的全局平均计算相似度（消除序列长度差异）
        query_avg = query_feat.mean(dim=1)  # [B, dim]
        memory_avg = memory_feat.mean(dim=1)  # [train_num, dim]

        # 余弦相似度计算
        sim = self.cos_sim(
            query_avg.unsqueeze(1).repeat(1, memory_avg.shape[0], 1),
            memory_avg.unsqueeze(0).repeat(batch_size, 1, 1)
        )

        # 过滤无效样本+取topk
        valid_mask = (memory_avg.sum(dim=1) != 0).float().unsqueeze(0)
        sim = sim * valid_mask
        _, topk_idx = torch.topk(sim, k=self.topk, dim=1)

        # 提取topk特征
        topk_feats = []
        for i in range(batch_size):
            feats = memory_feat[topk_idx[i]]  # [topk, S, dim]
            topk_feats.append(feats)
        return torch.stack(topk_feats, dim=0)  # [B, topk, S, dim]

    def forward(self, image, text, m1, m2, memoryBank):
        """
        前向传播：所有有效模态样本均生成补全特征（含完备样本）
        Returns:
            completed_image: [B, L_i+1, dim] 补全后图像特征（m1=0的样本替换，m1=1的样本不变）
            completed_text: [B, L_t+1, dim] 补全后文本特征（m2=0的样本替换，m2=1的样本不变）
            gen_image_full: [B, L_i+1, dim] 所有m2=1样本的补全图像特征（含m1=1的完备样本，用于计算补全损失）
            gen_text_full: [B, L_t+1, dim] 所有m1=1样本的补全文本特征（含m2=1的完备样本，用于计算补全损失）
        """
        batch_size = image.shape[0]
        completed_image = image.clone()
        completed_text = text.clone()

        # 初始化完备样本的补全特征存储
        gen_image_full = torch.zeros_like(image)
        gen_text_full = torch.zeros_like(text)

        all_text_exist_mask = (m2 == 1).squeeze(-1)  # [B]
        if all_text_exist_mask.any():
            # 提取这些样本的文本特征（用于检索生成补全图像）
            text_feat_exist = text[all_text_exist_mask]  # [M, S, dim]，M为m2=1的样本数

            retrieved_image = self.retrieve_similar(
                query_feat=text_feat_exist,
                memory_feat=memoryBank['image']
            )  # [M, topk, S, dim]
            # 生成所有m2=1样本的补全图像特征
            gen_image = self.image_generator(text_feat_exist, retrieved_image)  # [M, S, dim]

            # 分情况处理：
            # （1）m1=0（图像缺失）：用补全特征替换原始图像
            image_missing_mask = (m1 == 0) & (m2 == 1)
            image_missing_mask = image_missing_mask.squeeze(-1)  # [B]
            if image_missing_mask.any():
                # 找到m1=0在all_text_exist_mask中的索引
                missing_idx_in_exist = torch.where(image_missing_mask[all_text_exist_mask])[0]
                completed_image[image_missing_mask] = gen_image[missing_idx_in_exist]

            # （2）所有m2=1样本：保存补全图像特征（含m1=1的完备样本）
            gen_image_full[all_text_exist_mask] = gen_image

        # 2. 文本补全：所有m1=1的样本（含m2=0缺失样本 + m2=1完备样本）均生成补全文本特征
        all_image_exist_mask = (m1 == 1).squeeze(-1)  # [B]
        if all_image_exist_mask.any():
            # 提取这些样本的图像特征（用于检索生成补全文本）
            image_feat_exist = image[all_image_exist_mask]  # [N, dim]，N为m1=1的样本数

            retrieved_text = self.retrieve_similar(
                query_feat=image_feat_exist,
                memory_feat=memoryBank['text']
            )  # [N, topk, dim]
            # 生成所有m1=1样本的补全文本特征
            gen_text = self.text_generator(image_feat_exist, retrieved_text)  # [N, dim]

            # 分情况处理：
            # （1）m2=0（文本缺失）：用补全特征替换原始文本
            text_missing_mask = (m2 == 0) & (m1 == 1)
            text_missing_mask = text_missing_mask.squeeze(-1)  # [B]
            if text_missing_mask.any():
                # 找到m2=0在all_image_exist_mask中的索引
                missing_idx_in_exist = torch.where(text_missing_mask[all_image_exist_mask])[0]
                completed_text[text_missing_mask] = gen_text[missing_idx_in_exist]

            # （2）所有m1=1样本：保存补全文本特征（含m2=1的完备样本）
            gen_text_full[all_image_exist_mask] = gen_text

        return completed_image, completed_text, gen_image_full, gen_text_full
