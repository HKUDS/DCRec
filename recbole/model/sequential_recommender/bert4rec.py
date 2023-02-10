# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 12:08
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

r"""
BERT4Rec
################################################

Reference:
    Fei Sun et al. "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer."
    In CIKM 2019.

Reference code:
    The authors' tensorflow implementation https://github.com/FeiSun/BERT4Rec

"""

import random

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder


class BERT4Rec(SequentialRecommender):

    def __init__(self, config, dataset):
        super(BERT4Rec, self).__init__(config, dataset)

        self.config = config
        # load parameters info
        self.device = config["device"]
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.mask_ratio = config['mask_ratio']

        self.loss_type = config['loss_type']
        self.initializer_range = config['initializer_range']

        # load dataset info
        self.mask_token = self.n_items
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)  # mask token add 1
        self.position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)  # add mask_token at the last
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

        self.loss_fct = nn.CrossEntropyLoss()

        # we only need compute the loss at the masked position
        try:
            assert self.loss_type in ['CE']
        except AssertionError:
            raise AssertionError("Make sure 'loss_type' be CE!")

        # parameters initialization
        self.apply(self._init_weights)

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

    def get_attention_mask(self, item_seq):
        """Generate bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # bidirectional mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return sequence

    def reconstruct_train_data(self, item_seq, last_item = None, max_length=None):
        """
        Mask item sequence for training.
        """
        if max_length is None:
            max_length = self.max_seq_length

        device = item_seq.device
        batch_size = item_seq.size(0)
        zero_last_padding = torch.zeros((item_seq.size(0), 1), device=item_seq.device, dtype=torch.long)
        item_seq = torch.cat((item_seq, zero_last_padding), dim=1)
        seq_lens = torch.count_nonzero(item_seq, dim=1)

        if last_item is not None:
            item_seq[list(range(batch_size)), seq_lens] = last_item

        sequence_instances = item_seq.cpu().numpy().tolist()

        # Masked Item Prediction
        # [B * Len]
        masked_item_sequence = []
        pos_items = []
        for i, instance in enumerate(sequence_instances):
            # WE MUST USE 'copy()' HERE!
            masked_sequence = instance.copy()
            pos_item = []
            for index_id, item in enumerate(instance):
                # padding is 0, the sequence is end
                if item == 0:
                    break
                prob = random.random()
                if prob < self.mask_ratio:
                    pos_item.append(item)
                    masked_sequence[index_id] = self.mask_token

            # 每个序列都必须被mask
            if len(pos_item) == 0:
                index_id = random.randint(0, seq_lens[i])
                item = instance[index_id]
                pos_item.append(item)
                masked_sequence[index_id] = self.mask_token

            masked_item_sequence.append(masked_sequence)
            pos_items.extend(pos_item)

        # [B Len]
        masked_item_sequence = torch.tensor(masked_item_sequence, dtype=torch.long, device=device).view(batch_size, -1)
        # [B*mask_len]
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device)
        return masked_item_sequence, pos_items

    def reconstruct_test_data(self, item_seq, item_seq_len):
        """
        Add mask token at the last position according to the lengths of item_seq
        """
        padding = torch.zeros(item_seq.size(0), dtype=torch.long, device=item_seq.device)  # [B]
        item_seq = torch.cat((item_seq, padding.unsqueeze(-1)), dim=-1)  # [B max_len+1]
        for batch_id, last_position in enumerate(item_seq_len):
            item_seq[batch_id][last_position] = self.mask_token
        return item_seq

    def forward(self, item_seq):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        return output  # [B L H]

    def multi_hot_embed(self, masked_index, max_length):
        """
        For memory, we only need calculate loss for masked position.
        Generate a multi-hot vector to indicate the masked position for masked sequence, and then is used for
        gathering the masked position hidden representation.

        Examples:
            sequence: [1 2 3 4 5]

            masked_sequence: [1 mask 3 mask 5]

            masked_index: [1, 3]

            max_length: 5

            multi_hot_embed: [[0 1 0 0 0], [0 0 0 1 0]]
        """
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(masked_index.size(0), max_length, device=masked_index.device)
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        last_item = interaction[self.ITEM_ID]
        self.max_seq_length = item_seq.shape[1] + 1
        masked_item_seq, pos_items = self.reconstruct_train_data(item_seq, last_item=last_item)

        seq_output = self.forward(masked_item_seq)
        masked_index = (masked_item_seq==self.mask_token)
        # [mask_num, H]
        seq_output = seq_output[masked_index]
        # [item_num, H]
        test_item_emb = self.item_embedding.weight[:self.n_items]  # [item_num H]
        logits = torch.mm(seq_output, test_item_emb.transpose(0, 1))  # [mask_num, item_num]

        loss = self.loss_fct(logits, pos_items)

        if torch.isnan(loss):
            print(masked_item_seq.tolist())
            print(masked_index.tolist())
            input()
        return loss


    def calculate_loss_meta(self, item_seq):
        item_seq = item_seq.to(self.device)
        masked_item_seq, pos_items = self.reconstruct_train_data(item_seq, max_length=self.config["meta_task_lengths"][-1])

        seq_output = self.forward(masked_item_seq)
        masked_index = (masked_item_seq==self.mask_token)
        # [mask_num, H]
        seq_output = seq_output[masked_index]
        # [item_num, H]
        test_item_emb = self.item_embedding.weight[:self.n_items]  # [item_num H]
        logits = torch.mm(seq_output, test_item_emb.transpose(0, 1))  # [mask_num, item_num]

        loss = self.loss_fct(logits, pos_items)

        if torch.isnan(loss):
            print(masked_item_seq.tolist())
            print(masked_index.tolist())
            input()
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        seq_output = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len)  # [B H]
        test_item_emb = self.item_embedding(test_item)
        seq_output = seq_output.expand_as(test_item_emb)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        seq_output = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len)  # [B H]
        test_items_emb = self.item_embedding.weight[:self.n_items]  # delete masked token
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, item_num]
        return scores
    
    def fast_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction["item_id_with_negs"]
        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        seq_output = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len)  # [B H]
        test_item_emb = self.item_embedding(test_item) # [B, num, H]
        scores = torch.matmul(seq_output.unsqueeze(1), test_item_emb.transpose(1,2)).squeeze()
        return scores

