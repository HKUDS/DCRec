# -*- coding: utf-8 -*-
# @Time    : 2020/9/19 21:49
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

r"""
S3Rec
################################################

Reference:
    Kun Zhou and Hui Wang et al. "S^3-Rec: Self-Supervised Learning
    for Sequential Recommendation with Mutual Information Maximization"
    In CIKM 2020.

Reference code:
    https://github.com/RUCAIBox/CIKM2020-S3Rec

"""

import random

import torch
from torch import nn
import numpy as np
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class DSSEncoder(nn.Module):
    def __init__(self, hidden_size, num_intents, max_seq_length):
        super(DSSEncoder, self).__init__()
        # self.sas_encoder = SASEncoder(args)
        # prototypical intention vector for each intention
        self.prototypes = nn.ParameterList([nn.Parameter(torch.randn(hidden_size) *
                                                         (1 / np.sqrt(hidden_size)))
                                            for _ in range(num_intents)])

        self.layernorm1 = LayerNorm(hidden_size, eps=1e-12)
        self.layernorm2 = LayerNorm(hidden_size, eps=1e-12)
        self.layernorm3 = LayerNorm(hidden_size, eps=1e-12)
        self.layernorm4 = LayerNorm(hidden_size, eps=1e-12)
        self.layernorm5 = LayerNorm(hidden_size, eps=1e-12)

        self.w = nn.Linear(hidden_size, hidden_size)

        self.b_prime = nn.Parameter(torch.zeros(hidden_size))
        # self.b_prime = BiasLayer(hidden_size, 'zeros')

        # individual alpha for each position
        self.alphas = nn.Parameter(torch.zeros(max_seq_length, hidden_size))

        self.beta_input_seq = nn.Parameter(torch.randn(num_intents, hidden_size) *
                                           (1 / np.sqrt(hidden_size)))

        self.beta_label_seq = nn.Parameter(torch.randn(num_intents, hidden_size) *
                                           (1 / np.sqrt(hidden_size)))

    def _intention_clustering(self,
                              z: torch.Tensor) -> torch.Tensor:
        """
        Method to measure how likely the primary intention at position i
        is related with kth latent category
        :param z:
        :return:
        """
        z = self.layernorm1(z)
        hidden_size = z.shape[2]
        exp_normalized_numerators = list()
        i = 0
        for prototype_k in self.prototypes:
            prototype_k = self.layernorm2(prototype_k)  # [D]
            numerator = torch.matmul(z, prototype_k)  # [B, S]
            exp_normalized_numerator = torch.exp(numerator / np.sqrt(hidden_size))  # [B, S]
            exp_normalized_numerators.append(exp_normalized_numerator)
            if i == 0:
                denominator = exp_normalized_numerator
            else:
                denominator = torch.add(denominator, exp_normalized_numerator)
            i = i + 1

        all_attentions_p_k_i = [torch.div(k, denominator)
                                for k in exp_normalized_numerators]  # [B, S] K times
        all_attentions_p_k_i = torch.stack(all_attentions_p_k_i, -1)  # [B, S, K]

        return all_attentions_p_k_i

    def _intention_weighting(self,
                             z: torch.Tensor) -> torch.Tensor:
        """
        Method to measure how likely primary intention at position i
        is important for predicting user's future intentions
        :param z:
        :return:
        """
        hidden_size = z.shape[2]
        keys_tilde_i = self.layernorm3(z + self.alphas)  # [B, S, D]
        keys_i = keys_tilde_i + torch.relu(self.w(keys_tilde_i))  # [B, S, D]
        query = self.layernorm4(self.b_prime + self.alphas[-1, :] + z[:, -1, :])  # [B, D]
        query = torch.unsqueeze(query, -1)  # [B, D, 1]
        numerators = torch.matmul(keys_i, query)  # [B, S, 1]
        exp_normalized_numerators = torch.exp(numerators / np.sqrt(hidden_size))
        sum_exp_normalized_numerators = exp_normalized_numerators.sum(1).unsqueeze(-1)  # [B, 1] to [B, 1, 1]
        all_attentions_p_i = exp_normalized_numerators / sum_exp_normalized_numerators  # [B, S, 1]
        all_attentions_p_i = all_attentions_p_i.squeeze(-1)  # [B, S]

        return all_attentions_p_i

    def _intention_aggr(self,
                        z: torch.Tensor,
                        attention_weights_p_k_i: torch.Tensor,
                        attention_weights_p_i: torch.Tensor,
                        is_input_seq: bool) -> torch.Tensor:
        """
        Method to aggregate intentions collected at all positions according
        to both kinds of attention weights
        :param z:
        :param attention_weights_p_k_i:
        :param attention_weights_p_i:
        :param is_input_seq:
        :return:
        """
        attention_weights_p_i = attention_weights_p_i.unsqueeze(-1)  # [B, S, 1]
        attention_weights = torch.mul(attention_weights_p_k_i, attention_weights_p_i)  # [B, S, K]
        attention_weights_transpose = attention_weights.transpose(1, 2)  # [B, K, S]
        if is_input_seq:
            disentangled_encoding = self.beta_input_seq + torch.matmul(attention_weights_transpose, z)
        else:
            disentangled_encoding = self.beta_label_seq + torch.matmul(attention_weights_transpose, z)

        disentangled_encoding = self.layernorm5(disentangled_encoding)

        return disentangled_encoding  # [K, D]

    def forward(self,
                is_input_seq: bool,
                z: torch.Tensor):

        attention_weights_p_k_i = self._intention_clustering(z)  # [B, S, K]
        attention_weights_p_i = self._intention_weighting(z)  # [B, S]
        disentangled_encoding = self._intention_aggr(z,
                                                     attention_weights_p_k_i,
                                                     attention_weights_p_i,
                                                     is_input_seq)

        return disentangled_encoding


class DSSRec(SequentialRecommender):
    r"""
    S3Rec is the first work to incorporate self-supervised learning in
    sequential recommendation.

    NOTE:
        Under this framework, we need reconstruct the pretraining data,
        which would affect the pre-training speed.
    """

    def __init__(self, config, dataset):
        super(DSSRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.train_stage = config['train_stage']  # pretrain or finetune
        self.pre_model_path = config['pre_model_path']  # We need this for finetune
        self.mask_ratio = config['mask_ratio']
        self.aap_weight = config['aap_weight']
        self.mip_weight = config['mip_weight']
        self.map_weight = config['map_weight']
        self.sp_weight = config['sp_weight']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # load dataset info
        self.n_items = dataset.item_num + 1  # for mask token
        self.mask_token = self.n_items - 1

        self.batch_size = config['train_batch_size']

        # define layers and loss
        # modules shared by pre-training stage and fine-tuning stage
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.disentangled_encoder = DSSEncoder(self.hidden_size, 4, self.max_seq_length)

        # modules for pretrain
        # add unique dense layer for 4 losses respectively
        self.aap_norm = nn.Linear(self.hidden_size, self.hidden_size)
        self.mip_norm = nn.Linear(self.hidden_size, self.hidden_size)
        self.map_norm = nn.Linear(self.hidden_size, self.hidden_size)
        self.sp_norm = nn.Linear(self.hidden_size, self.hidden_size)
        self.loss_fct = nn.BCELoss(reduction='none')

        # modules for finetune
        if self.loss_type == 'BPR' and self.train_stage == 'finetune':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE' and self.train_stage == 'finetune':
            self.loss_fct = nn.CrossEntropyLoss()
        elif self.train_stage == 'finetune':
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        assert self.train_stage in ['pretrain', 'finetune']
        if self.train_stage == 'pretrain':
            self.apply(self._init_weights)
        else:
            # load pretrained model for finetune
            pretrained = torch.load(self.pre_model_path)
            self.logger.info(f'Load pretrained model from {self.pre_model_path}')
            self.load_state_dict(pretrained['state_dict'])

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def _masked_item_prediction(self, sequence_output, target_item_emb):
        sequence_output = self.mip_norm(sequence_output.view([-1, sequence_output.size(-1)]))  # [B*L H]
        target_item_emb = target_item_emb.view([-1, sequence_output.size(-1)])  # [B*L H]
        score = torch.mul(sequence_output, target_item_emb)  # [B*L H]
        return torch.sigmoid(torch.sum(score, -1))  # [B*L]

    def _segment_prediction(self, context, segment_emb):
        context = self.sp_norm(context)
        score = torch.mul(context, segment_emb)  # [B H]
        return torch.sigmoid(torch.sum(score, dim=-1))  # [B]

    def get_attention_mask(self, sequence, bidirectional=True):
        """
        In the pre-training stage, we generate bidirectional attention mask for multi-head attention.

        In the fine-tuning stage, we generate left-to-right uni-directional attention mask for multi-head attention.
        """
        attention_mask = (sequence > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        if not bidirectional:
            max_len = attention_mask.size(-1)
            attn_shape = (1, max_len, max_len)
            subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
            subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
            subsequent_mask = subsequent_mask.long().to(sequence.device)
            extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    # re-checked the math behind the code, it is CORRECT
    def __seq2seqloss(self,
                      inp_subseq_encodings: torch.Tensor,
                      label_subseq_encodings: torch.Tensor) -> torch.Tensor:
        sqrt_hidden_size = np.sqrt(self.hidden_size)
        product = torch.mul(inp_subseq_encodings, label_subseq_encodings)  # [B, K, D]
        normalized_dot_product = torch.sum(product, dim=-1) / sqrt_hidden_size  # [B, K]
        numerator = torch.exp(normalized_dot_product)  # [B, K]
        inp_subseq_encodings_trans = inp_subseq_encodings.transpose(0, 1)  # [K, B, D]
        inp_subseq_encodings_trans_expanded = inp_subseq_encodings_trans.unsqueeze(1)  # [K, 1, B, D]
        label_subseq_encodings_trans = label_subseq_encodings.transpose(0, 1).transpose(1, 2)  # [K, D, B]
        dot_products = torch.matmul(inp_subseq_encodings_trans_expanded, label_subseq_encodings_trans)  # [K, K, B, B]
        dot_products = torch.exp(dot_products / sqrt_hidden_size)
        dot_products = dot_products.sum(-1)  # [K, K, B]
        temp = dot_products.sum(1)  # [K, B]
        denominator = temp.transpose(0, 1)  # [B, K]
        seq2seq_loss_k = -torch.log2(numerator / denominator)
        seq2seq_loss_k = torch.flatten(seq2seq_loss_k)
        thresh_th = int(np.floor(0.5 * denominator.shape[0] * 4))
        thresh = torch.kthvalue(seq2seq_loss_k, thresh_th)[0]
        conf_indicator = seq2seq_loss_k <= thresh
        conf_seq2seq_loss_k = torch.mul(seq2seq_loss_k, conf_indicator)
        seq2seq_loss = torch.sum(conf_seq2seq_loss_k)
        # seq2seq_loss = torch.sum(seq2seq_loss_k)
        return seq2seq_loss

    # re-checked the math behind the code, it is CORRECT
    def __seq2itemloss(self,
                       inp_subseq_encodings: torch.Tensor,
                       next_item_emb: torch.Tensor) -> torch.Tensor:
        sqrt_hidden_size = np.sqrt(self.hidden_size)
        next_item_emb = torch.transpose(next_item_emb, 1, 2)  # [B, D, 1]
        dot_product = torch.matmul(inp_subseq_encodings, next_item_emb)  # [B, K, 1]
        exp_normalized_dot_product = torch.exp(dot_product / sqrt_hidden_size)
        numerator = torch.max(exp_normalized_dot_product, dim=1)[0]  # [B, 1]

        inp_subseq_encodings_trans = inp_subseq_encodings.transpose(0, 1)  # [K, B, D]
        next_item_emb_trans = next_item_emb.squeeze(-1).transpose(0, 1)  # [D, B]
        # sum of dot products of given input sequence encoding for each intent with all next item embeddings
        dot_products = torch.matmul(inp_subseq_encodings_trans,
                                    next_item_emb_trans) / sqrt_hidden_size  # [K, B, B]
        dot_products = torch.exp(dot_products)  # [K, B, B]
        dot_products = dot_products.sum(-1)
        dot_products = dot_products.transpose(0, 1)  # [B, K]
        # sum across all intents
        denominator = dot_products.sum(-1).unsqueeze(-1)  # [B, 1]
        seq2item_loss_k = -torch.log2(numerator / denominator)  # [B, 1]
        seq2item_loss = torch.sum(seq2item_loss_k)
        return seq2item_loss


    def forward(self, item_seq, bidirectional=True):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        attention_mask = self.get_attention_mask(item_seq, bidirectional=bidirectional)
        trm_output = self.trm_encoder(input_emb, attention_mask, output_all_encoded_layers=True)
        seq_output = trm_output[-1]  # [B L H]
        return seq_output

    def pretrain(
        self, inp_subseq, label_subseq, next_item
    ):
        """Pretrain out model using four pre-training tasks:

            1. Associated Attribute Prediction

            2. Masked Item Prediction

            3. Masked Attribute Prediction

            4. Segment Prediction
        """

        next_item_emb = self.item_embedding(next_item)  # [B, 1, D]

        input_subseq_encoding = self.forward(inp_subseq, bidirectional=False)

        label_subseq_encoding = self.forward(label_subseq, bidirectional=False)

        disent_inp_subseq_encodings = self.disentangled_encoder(True,
                                                                input_subseq_encoding)
        disent_label_seq_encodings = self.disentangled_encoder(False,
                                                               label_subseq_encoding)
        # seq2item loss
        seq2item_loss = self.__seq2itemloss(disent_inp_subseq_encodings, next_item_emb)
        # seq2seq loss
        seq2seq_loss = self.__seq2seqloss(disent_inp_subseq_encodings, disent_label_seq_encodings)

        return seq2item_loss + seq2seq_loss

    def _neg_sample(self, item_set):  # [ , ]
        item = random.randint(1, self.n_items - 1)
        while item in item_set:
            item = random.randint(1, self.n_items - 1)
        return item

    def _padding_zero_at_left(self, sequence):
        # had truncated according to the max_length
        pad_len = self.max_seq_length - len(sequence)
        sequence = [0] * pad_len + sequence
        return sequence
    
    @staticmethod
    def _get_input_label_subsequences(seq_len: int,
                                      sequence: list):
        if seq_len == 2:
            t = 1
        else:
            if not seq_len == 1:
                t = torch.randint(1, seq_len - 1, (1,))
        # input sub-sequence
        if seq_len == 1:
            inp_subseq = sequence
        else:
            inp_subseq = sequence[:t]
        # label sub-sequence
        if seq_len == 1:
            label_subseq = sequence
        else:
            label_subseq = sequence[t:]
        # next item
        if seq_len == 1:
            next_item = [sequence[0]]
        else:
            next_item = [sequence[t]]
        return inp_subseq, label_subseq, next_item

    def __get_items_dss_loss(self,
                             inp_subseq: list,
                             label_subseq: list,
                             next_item: list) -> tuple:
        """
        Method to prepare data instance for Disentangled Self-Supervision
        :param inp_subseq:
        :param label_subseq:
        :param next_item:
        :return:
        """
        # data preparation for DSS loss
        # input sub-sequence
        inp_pad_len = self.max_seq_length - len(inp_subseq)
        inp_pos_items = ([0] * inp_pad_len) + inp_subseq
        inp_pos_items = inp_pos_items[-self.max_seq_length:]
        # label sub-sequence
        len_label_subseq = len(label_subseq)
        label_subseq.reverse()
        label_pad_len = self.max_seq_length - len_label_subseq
        label_pos_items = [0] * label_pad_len + label_subseq
        label_pos_items = label_pos_items[-self.max_seq_length:]
        # label_pos_items.reverse()
        assert len(inp_pos_items) == self.max_seq_length
        assert len(label_pos_items) == self.max_seq_length
        # end of data preparation for DSS loss
        return inp_pos_items, label_pos_items, next_item

    def reconstruct_pretrain_data(self, item_seq, item_seq_len):
        """Generate pre-training data for the pre-training stage."""
        device = item_seq.device
        batch_size = item_seq.size(0)

        end_index = item_seq_len.cpu().numpy().tolist()
        item_seq = item_seq.cpu().numpy().tolist()

        # we will padding zeros at the left side
        # these will be train_instances, after will be reshaped to batch
        sequence_instances = []
        for i, end_i in enumerate(end_index):
            sequence_instances.append(item_seq[i][:end_i])

        # Masked Item Prediction and Masked Attribute Prediction
        # [B * Len]
        inp_pos_items = []
        label_pos_items = []
        next_items = []
        for instance in sequence_instances:
            seq = instance.copy()
            inp_subseq, label_subseq, next_item = self._get_input_label_subsequences(len(seq), seq)
            inp_pos_item, label_pos_item, next_item = self.__get_items_dss_loss(inp_subseq, label_subseq, next_item)
            inp_pos_items.append(inp_pos_item)
            label_pos_items.append(label_pos_item)
            next_items.append(next_item)

        inp_pos_items = torch.tensor(inp_pos_items, dtype=torch.long, device=device).view(batch_size, -1)
        label_pos_items = torch.tensor(label_pos_items, dtype=torch.long, device=device).view(batch_size, -1)
        next_items = torch.tensor(next_items, dtype=torch.long, device=device).view(batch_size, -1)

        return inp_pos_items, label_pos_items, next_items

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        # pretrain
        if self.train_stage == 'pretrain':
            inp_pos_items, label_pos_items, next_item = self.reconstruct_pretrain_data(item_seq, item_seq_len)

            loss = self.pretrain(
                inp_pos_items, label_pos_items, next_item
            )
        # finetune
        else:
            pos_items = interaction[self.POS_ITEM_ID]
            # we use uni-directional attention in the fine-tuning stage
            seq_output = self.forward(item_seq, bidirectional=False)
            seq_output = self.gather_indexes(seq_output, item_seq_len - 1)

            if self.loss_type == 'BPR':
                neg_items = interaction[self.NEG_ITEM_ID]
                pos_items_emb = self.item_embedding(pos_items)
                neg_items_emb = self.item_embedding(neg_items)
                pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
                neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
                loss = self.loss_fct(pos_score, neg_score)
            else:  # self.loss_type = 'CE'
                test_item_emb = self.item_embedding.weight
                logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
                loss = self.loss_fct(logits, pos_items)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, bidirectional=False)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, bidirectional=False)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_items_emb = self.item_embedding.weight[:self.n_items - 1]  # delete masked token
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores

    def fast_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction["item_id_with_negs"]
        seq_output = self.forward(item_seq, bidirectional=False)

        test_item_emb = self.item_embedding(test_item)  # [B, num, H]
        scores = torch.matmul(seq_output.unsqueeze(
            1), test_item_emb.transpose(1, 2)).squeeze()
        return scores
