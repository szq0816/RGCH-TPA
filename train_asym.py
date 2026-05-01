from hash_model import CMCL
import time
from torch import nn
# from einops import rearrange
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import scipy.io as scio
import torch.nn.functional as F
from optimization import BertAdam
from utils.calc_utils import calc_neighbor, calc_map_k
from load_data import generate_dataset
from utils import get_logger, clear_logger


dataset_root_path = " "


class TrainerAsym:
    """
        train class
    """
    def __init__(self, args):

        self.args = args

        torch.random.manual_seed(seed=self.args.seed)
        torch.autograd.set_detect_anomaly(True)

        os.makedirs(self.args.save_dir, exist_ok=True)
        self._init_writer()

        self.logger.info('Start logging...')

        if self.args.is_train:
            log_str_args = "\n"
            for para in self.args.__dict__:
                log_str_args += " " * (40 - len(para)) + str(para) + "=" + str(self.args.__dict__[para]) + "\n"
            self.logger.info(log_str_args)
        else:
            self.logger.info(f"pretrained: {self.args.pretrained}")

        self.rank = self.args.rank  # gpu rank

        self._init_dataset()
        self._init_model()

        self.best_avg_map = 0
        self.best_epoch = 0
        self.max_i2t = {16: 0, 32: 0, 64: 0, 128: 0}
        self.max_t2i = {16: 0, 32: 0, 64: 0, 128: 0}
        self.max_avg = {16: 0, 32: 0, 64: 0, 128: 0}

        self.logger.info("Train dataset len: {}".format(len(self.train_loader.dataset)))

        self.k_bits_list = list(map(int, self.args.k_bits_list.split(",")))  # str -> list

        self.extend_bits_list = []
        self.extend_bits_list.extend(self.k_bits_list)
        self.extend_bits_list.append(self.args.auxiliary_bit_dim)

        # buffer
        self.ibuf = {}
        self.tbuf = {}
        self.bbuf = {}

        for one in self.extend_bits_list:
            self.ibuf[one] = torch.randn(self.args.train_num, one).to(self.rank, non_blocking=True)
            self.tbuf[one] = torch.randn(self.args.train_num, one).to(self.rank, non_blocking=True)
            self.bbuf[one] = torch.sign(self.ibuf[one] + self.tbuf[one])

        # MemoryBank
        self.memoryBank = {'image': torch.zeros(self.args.train_num, 50, 512).to(self.rank, non_blocking=True),
                           'text': torch.zeros(self.args.train_num, 33, 512).to(self.rank, non_blocking=True)}

        self.device = torch.device("cuda", self.rank)
        self.run()

    def run(self):
        if self.args.is_train:
            self.train()
        else:
            self.test()

    def _init_writer(self):
        self.logger = get_logger(os.path.join(self.args.save_dir, "train.log" if self.args.is_train else "test.log"))

        with open(os.path.join(self.args.save_dir, "description.txt"), 'w') as f:
            # write description
            f.write("")
            f.close()

    def _init_model(self):
        self.logger.info("init model.")
        self.logger.info("Using ViT & GPT2...")

        self.model = CMCL(args=self.args).to(self.rank)

        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info(f"load pretrained model at {self.args.pretrained}")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"))

        self.model.float()
        self.optimizer = BertAdam(
            [
                {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},
                {'params': self.model.completion.parameters(), 'lr': self.args.lr},
                {'params': self.model.alignment.parameters(), 'lr': self.args.lr},
                {'params': self.model.hash.parameters(), 'lr': self.args.lr},
            ],
            lr=self.args.lr,
            warmup=self.args.warmup_proportion, schedule='warmup_cosine',
            b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * self.args.epochs,
            weight_decay=self.args.weight_decay, max_grad_norm=1.0)

    def _init_dataset(self):
        self.logger.info("init dataset.")
        self.logger.info(f"Using {self.args.dataset} dataset...")

        global dataset_root_path
        self.args.index_file = os.path.join(dataset_root_path, self.args.dataset, self.args.index_file)
        self.args.caption_file = os.path.join(dataset_root_path, self.args.dataset, self.args.caption_file)
        self.args.label_file = os.path.join(dataset_root_path, self.args.dataset, self.args.label_file)

        train_data, query_data, retrieval_data = generate_dataset(captionFile=self.args.caption_file,
                                                                  indexFile=self.args.index_file,
                                                                  labelFile=self.args.label_file,
                                                                  maxWords=self.args.max_words,
                                                                  imageResolution=self.args.resolution,
                                                                  dataset=self.args.dataset,
                                                                  query_num=self.args.query_num,
                                                                  train_num=self.args.train_num,
                                                                  full_ratio=self.args.full_ratio,
                                                                  oimg_ratio=self.args.oimg_ratio,
                                                                  seed=self.args.seed)

        self.train_labels = train_data.get_all_label().float()
        self.query_labels = query_data.get_all_label().float()
        self.retrieval_labels = retrieval_data.get_all_label().float()
        self.d_m1 = retrieval_data.m1
        self.d_m2 = retrieval_data.m2

        self.args.retrieval_num = len(self.retrieval_labels)

        self.args.num_class = self.query_labels.shape[1]

        self.logger.info(f"query shape: {self.query_labels.shape}")
        self.logger.info(f"retrieval shape: {self.retrieval_labels.shape}")

        self.train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=False,
            shuffle=True,
        )
        self.query_loader = DataLoader(
            dataset=query_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=False,
            shuffle=True,
        )
        self.retrieval_loader = DataLoader(
            dataset=retrieval_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=False,
            shuffle=True,
        )

    def change_state(self, mode):
        if mode == "train":
            self.model.train()
        elif mode == "valid":
            self.model.eval()

    def train_epoch(self, epoch):
        self.change_state(mode="train")
        self.logger.info("\n\n\n")
        self.logger.info(
            "####################### Train epochs: %d/%d #######################" % (epoch, self.args.epochs))
        epoch_avg_loss_dict = {'all_loss': 0}

        for image, text, key_padding_mask, label, m1, m2, index in tqdm(self.train_loader):

            image = image.float().to(self.rank, non_blocking=True)
            label = label.float().to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)

            m1 = m1.to(self.rank, non_blocking=True)
            m2 = m2.to(self.rank, non_blocking=True)
            image = torch.mul(image, m1.unsqueeze(-1).unsqueeze(-1))
            text = torch.mul(text, m2)

            key_padding_mask = key_padding_mask.to(self.rank, non_blocking=True)

            output_dict = self.model(image, text, key_padding_mask, m1, m2, self.memoryBank)

            # 更新完备样本的存储库
            complete_mask = (m1 == 1) & (m2 == 1)
            complete_mask = complete_mask.squeeze(-1)
            complete_indices = index[complete_mask.cpu()]
            img_seq = output_dict['img_seq']
            txt_seq = output_dict['txt_seq']
            self.memoryBank['image'][complete_indices] = img_seq[complete_mask].detach()
            self.memoryBank['text'][complete_indices] = txt_seq[complete_mask].detach()

            _B_batch = {}
            for one in self.extend_bits_list:
                img_cls_hash = output_dict['img_cls_hash'][one]
                txt_cls_hash = output_dict['txt_cls_hash'][one]

                self.ibuf[one][index] = img_cls_hash.detach()
                self.tbuf[one][index] = txt_cls_hash.detach()

                _B_batch[one] = self.bbuf[one][index]

            ALL_LOSS_DICT = self.compute_loss(output_dict, label, _B_batch, m1, m2)

            loss = 0
            loss_pt_align = output_dict['loss_pt_align']
            loss_rec = output_dict['loss_rec']
            loss_local = self.args.gamma1 * loss_pt_align + self.args.gamma2 * loss_rec
            loss += loss_local
            for key in ALL_LOSS_DICT:
                loss += ALL_LOSS_DICT[key]
                if key in epoch_avg_loss_dict:
                    epoch_avg_loss_dict[key] += ALL_LOSS_DICT[key]
                else:
                    epoch_avg_loss_dict[key] = ALL_LOSS_DICT[key]
            epoch_avg_loss_dict['all_loss'] += loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # undate B.
        for one in self.extend_bits_list:
            self.bbuf[one] = torch.sign(self.ibuf[one] + self.tbuf[one])
        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}] all loss: {epoch_avg_loss_dict['all_loss'].data / (len(self.train_loader))}")
        self.logger.info(f"lr: {'-'.join([str('%.9f' % itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}")

    def train(self):
        self.logger.info("Start train...")

        for epoch in range(self.args.epochs):
            epoch += 1
            time1 = time.time()
            self.train_epoch(epoch)

            time2 = time.time()
            spend_time = int(time2 - time1)
            self.logger.info(
                f"{self.args.dataset}_{self.args.k_bits_list}. Train epoch [{epoch}], spend {spend_time // 60} min, {spend_time % 60} sec")

            if epoch % self.args.valid_freq == 0:
                self.valid(epoch)

            time3 = time.time()
            spend_time = int(time3 - time2)
            self.logger.info(
                f"{self.args.dataset}_{self.args.k_bits_list}. Valid epoch [{epoch}], spend {spend_time // 60} min, {spend_time % 60} sec")

        self.logger.info(f">>>>>>> FINISHED {self.args.dataset}_full={self.args.full_ratio}_{self.args.k_bits_list}. <<<<<<<")
        self.logger.info(f"Best epoch: {self.best_epoch}, best avg_mAP: {self.best_avg_map}")
        clear_logger()

    def valid(self, epoch):
        self.logger.info("\n")
        self.logger.info(" Valid: %d/%d " % (epoch, self.args.epochs))
        self.change_state(mode="valid")

        qi_dict, qt_dict = self.get_code(self.query_loader, self.args.query_num)
        ri_dict, rt_dict = self.get_code(self.retrieval_loader, self.args.retrieval_num)

        _map_epoch = 0

        for one in self.extend_bits_list:
            _k_ = None
            q_i = qi_dict[one]
            q_t = qt_dict[one]

            r_i = ri_dict[one]
            r_t = rt_dict[one]

            m1_bool = self.d_m1.squeeze(-1).bool()
            m2_bool = self.d_m2.squeeze(-1).bool()
            r_i = r_i[m1_bool]
            r_t = r_t[m2_bool]
            r_i_label = self.retrieval_labels[m1_bool]
            r_t_label = self.retrieval_labels[m2_bool]
            # 计算map
            mAPi2t = calc_map_k(q_i.to(self.device), r_t.to(self.device), self.query_labels.to(self.device),
                                r_t_label.to(self.device), _k_).item()
            mAPt2i = calc_map_k(q_t.to(self.device), r_i.to(self.device), self.query_labels.to(self.device),
                                r_i_label.to(self.device), _k_).item()
            mAPi2i = calc_map_k(q_i.to(self.device), r_i.to(self.device), self.query_labels.to(self.device),
                                r_i_label.to(self.device), _k_).item()
            mAPt2t = calc_map_k(q_t.to(self.device), r_t.to(self.device), self.query_labels.to(self.device),
                                r_t_label.to(self.device), _k_).item()

            avg_map = (mAPi2t + mAPt2i) / 2.0

            self.max_i2t[one] = mAPi2t
            self.max_t2i[one] = mAPt2i
            self.max_avg[one] = avg_map

            if one != self.args.auxiliary_bit_dim:
                _map_epoch += mAPi2t + mAPt2i
            self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}]")
            self.logger.info(f"{one} bits: MAP(i->t): {round(mAPi2t, 4)}, MAP(t->i): {round(mAPt2i, 4)}, Avg_map: {round(avg_map, 4)}")
            self.logger.info(f"{one} bits: MAP(i->i): {round(mAPi2i, 4)}, MAP(t->t): {round(mAPt2t, 4)}")

        _map_epoch /= (2 * len(self.k_bits_list))
        
        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}]")
        self.logger.info(f"dataset={self.args.dataset}  full={self.args.full_ratio}  Avg mAP: {round(_map_epoch, 4)}")

        if _map_epoch > self.best_avg_map:
            self.best_epoch = epoch
            self.best_avg_map = _map_epoch
            self.logger.info("$$$$$$$$$$$$$$$$$$$$ Best avg maps. $$$$$$$$$$$$$$$$$$$$$$$$")
            # self.save_model(epoch)
            # for one in self.extend_bits_list:
            #     _file_name = "hashcode-" + str(one)
            #     self.save_mat(qi_dict[one], qt_dict[one], ri_dict[one], rt_dict[one], _file_name)
            # self.logger.info(f"Save best *.mat data!")
            with open(os.path.join(self.args.save_dir, 'max_map.txt'), 'w') as f:
                # f.write('==================================================\n')
                f.write('==== dataset=%s  full=%.2f  best_epoch=%d avg_map=%3.4f====\n'
                        % (self.args.dataset, self.args.full_ratio, self.best_epoch, self.best_avg_map))
                for i in self.extend_bits_list:
                    f.write('%3d bit: max_i2t: %3.4f, max_t2i: %3.4f, max_avg: %3.4f\n'
                            % (i, self.max_i2t[i], self.max_t2i[i], self.max_avg[i]))
                f.write('==================================================\n\n')

        self.logger.info(f"Best epoch: {self.best_epoch}, best avg_mAP: {round(self.best_avg_map, 4)}")

    def predict_loss(self, pre_feat, ori_feat, mask):
        reconstruction_criterion = nn.MSELoss()
        pair_pre_feat = torch.mul(pre_feat, mask)
        pair_ori_feat = torch.mul(ori_feat, mask)
        loss = reconstruction_criterion(pair_pre_feat, pair_ori_feat)
        return loss

    def hash_loss_group(self, hi, ht, hi_buffer, ht_buffer, label_sim, label, B, K, weight=1, type='-16-bits'):
        ALL_LOSS = {}

        # CLS Intra
        hyper_cls_intra = self.args.hyper_cls_intra
        ALL_LOSS[f'cls_intra_i_{type}'] = weight * hyper_cls_intra * self.bayesian_loss(hi_buffer, hi, label_sim)
        ALL_LOSS[f'cls_intra_t_{type}'] = weight * hyper_cls_intra * self.bayesian_loss(ht_buffer, ht, label_sim)

        # CLS Inter
        hyper_cls_inter = self.args.hyper_cls_inter
        ALL_LOSS[f'cls_inter_likelihood_{type}'] = weight * hyper_cls_inter * \
                                                   (self.bayesian_loss(hi_buffer, ht, label_sim) +
                                                    self.bayesian_loss(ht_buffer, hi, label_sim))

        # quantization loss
        hyper_quan = self.args.hyper_quan
        ALL_LOSS[f'quantization_{type}'] = weight * hyper_quan * (self.quantization_loss(hi, B, K_bits=K)
                                                                  + self.quantization_loss(ht, B, K_bits=K))

        return ALL_LOSS

    def compute_loss(self, output_dict, label, B_batch, m1, m2):
        ALL_LOSS = {}

        label_sim = calc_neighbor(self.train_labels.float().to(self.rank, non_blocking=True), label)

        img_cls_hash = {}
        txt_cls_hash = {}

        for one in self.extend_bits_list:
            img_cls_hash[one] = output_dict['img_cls_hash'][one]
            txt_cls_hash[one] = output_dict['txt_cls_hash'][one]

        weights_list = [1 for _ in self.k_bits_list]
        weights_list.append(self.args.mu)

        for i, one in enumerate(self.extend_bits_list):
            _loss_dict_group = self.hash_loss_group(img_cls_hash[one],
                                                    txt_cls_hash[one],
                                                    self.ibuf[one],
                                                    self.tbuf[one],
                                                    label_sim,
                                                    label,
                                                    B_batch[one],
                                                    K=one,
                                                    weight=weights_list[i],
                                                    type=f"-{one}-bits"
                                                    )
            ALL_LOSS.update(_loss_dict_group)

        # Contrastive Alignment loss
        beta = self.args.beta
        res_img_cls = output_dict['res_img_cls']
        res_txt_eos = output_dict['res_txt_cls']
        ALL_LOSS['Global_infoNCE'] = beta * self.info_nce_loss(
            res_img_cls, res_txt_eos, temperature=self.args.tao_global)

        # Reconstruction target...
        _recon_i = _recon_t = B_batch[self.args.auxiliary_bit_dim]
        # Reconstruction loss...
        for i, one in enumerate(self.k_bits_list):
            img_cls_hash_recon = output_dict['img_cls_hash_recon'][one]
            txt_cls_hash_recon = output_dict['txt_cls_hash_recon'][one]

            mu = self.args.mu
            hyper_recon = self.args.hyper_recon

            ALL_LOSS[f'recon_i_{one}'] = mu * hyper_recon * (
                F.mse_loss(_recon_i, img_cls_hash_recon, reduction='sum')) / (img_cls_hash_recon.shape[0])
            ALL_LOSS[f'recon_t_{one}'] = mu * hyper_recon * (
                F.mse_loss(_recon_t, txt_cls_hash_recon, reduction='sum')) / (txt_cls_hash_recon.shape[0])

        # 补全损失
        gen_image_full = output_dict['gen_image_full']
        gen_text_full = output_dict['gen_text_full']
        completed_image = output_dict['completed_image']
        completed_text = output_dict['completed_text']
        ALL_LOSS['pre_loss'] = self.args.alpha * self.predict_loss(gen_image_full, completed_image, m2.unsqueeze(-1)) \
                               + self.args.alpha * self.predict_loss(gen_text_full, completed_text, m1.unsqueeze(-1))

        return ALL_LOSS

    def get_code(self, data_loader, length: int):
        ibuf = {}
        tbuf = {}

        for one in self.extend_bits_list:
            ibuf[one] = torch.empty(length, one, dtype=torch.float).to(self.device)
            tbuf[one] = torch.empty(length, one, dtype=torch.float).to(self.device)

        for image, text, key_padding_mask, label, m1, m2, index in tqdm(data_loader):
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            key_padding_mask = key_padding_mask.to(self.rank, non_blocking=True)
            index = index.numpy()

            m1 = m1.to(self.rank, non_blocking=True)
            m2 = m2.to(self.rank, non_blocking=True)
            image = torch.mul(image, m1.unsqueeze(-1).unsqueeze(-1))
            text = torch.mul(text, m2)

            output_dict = self.model(image, text, key_padding_mask, m1, m2, self.memoryBank)

            for one in self.extend_bits_list:
                img_cls_hash = output_dict['img_cls_hash'][one].detach()
                txt_cls_hash = output_dict['txt_cls_hash'][one].detach()

                ibuf[one][index, :] = torch.sign(img_cls_hash)
                tbuf[one][index, :] = torch.sign(txt_cls_hash)

        return ibuf, tbuf

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.args.save_dir, "model.pth"))
        # self.logger.info(f"Save model to {os.path.join(self.args.save_dir, 'model.pth')}")

    def save_mat(self, query_img, query_txt, retrieval_img, retrieval_txt, r_i_label, r_t_label, fname='hashcode'):

        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = self.query_labels.numpy()
        retrieval_labels_img = r_i_label.numpy()
        retrieval_labels_txt = r_t_label.numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l_i': retrieval_labels_img,
            'r_l_t': retrieval_labels_txt
        }
        scio.savemat(os.path.join(self.args.save_dir, fname + '.mat'), result_dict)

    def info_nce_loss(self, out_1, out_2, temperature=0.07):
        # out_*: ND
        bz = out_1.size(0)
        targets = torch.arange(bz).type_as(out_1).long()

        scores = out_1.mm(out_2.t())
        scores /= temperature
        scores1 = scores.transpose(0, 1)
        loss0 = F.cross_entropy(scores, targets)
        loss1 = F.cross_entropy(scores1, targets)

        return 0.5 * (loss0 + loss1)

    def bayesian_loss(self, a: torch.Tensor, b: torch.Tensor, label_sim: torch.Tensor):
        # a: ND
        # b: MD
        # label_sim: NM
        s = 0.5 * torch.matmul(a, b.t()).clamp(min=-64, max=64)
        b_loss = -torch.mean(label_sim * s - torch.log(1 + torch.exp(s)))
        return b_loss

    def quantization_loss(self, hash_feature, B, K_bits):
        return F.mse_loss(hash_feature, B, reduction='sum') / (hash_feature.shape[0]) / K_bits

    def test(self):
        if self.args.pretrained == "" or self.args.pretrained == "MODEL_PATH":
            self.logger.error("test step must load a model! please set the --pretrained argument.")
            raise RuntimeError("test step must load a model! please set the --pretrained argument.")

        self.change_state(mode="valid")

        qi_dict, qt_dict = self.get_code(self.query_loader, self.args.query_num)
        ri_dict, rt_dict = self.get_code(self.retrieval_loader, self.args.retrieval_num)

        _map_epoch = 0

        for one in self.extend_bits_list:
            _k_ = None
            q_i = qi_dict[one]
            q_t = qt_dict[one]

            r_i = ri_dict[one]
            r_t = rt_dict[one]

            m1_bool = self.d_m1.squeeze(-1).bool()
            m2_bool = self.d_m2.squeeze(-1).bool()
            r_i = r_i[m1_bool]
            r_t = r_t[m2_bool]
            r_i_label = self.retrieval_labels[m1_bool]
            r_t_label = self.retrieval_labels[m2_bool]

            mAPi2t = calc_map_k(q_i.to(self.device), r_t.to(self.device), self.query_labels.to(self.device),
                                r_t_label.to(self.device), _k_).item()
            mAPt2i = calc_map_k(q_t.to(self.device), r_i.to(self.device), self.query_labels.to(self.device),
                                r_i_label.to(self.device), _k_).item()

            _map_epoch += mAPi2t + mAPt2i
            self.logger.info(f"{one} bits: MAP(i->t): {round(mAPi2t, 4)}, MAP(t->i): {round(mAPt2i, 4)}")
            if one != self.args.auxiliary_bit_dim:
                _map_epoch += mAPi2t + mAPt2i
                self.logger.info(f"{one} bits: MAP(i->t): {round(mAPi2t, 4)}, MAP(t->i): {round(mAPt2i, 4)}")

        _map_epoch /= (2 * len(self.k_bits_list))
        self.logger.info(f"avg mAP: {round(_map_epoch, 5)}")

        for one in self.extend_bits_list:
            _file_name = "hashcode-" + str(one)
            self.save_mat(qi_dict[one], qt_dict[one], ri_dict[one], rt_dict[one], _file_name)

        self.logger.info(">>>>>> Save *.mat data! Exit...")
