#from sklearn.semi_supervised import LabelSpreading
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from iou_loss import Chamfer1DLoss 
from torch.autograd import Variable


def check_size(tensor, *args):
    size = [a for a in args]
    assert tensor.size() == torch.Size(size), tensor.size()

def skip_add_pyramid(x, seq_len, skip_add="add"):
    if len(x.size()) == 2:
        x = x.unsqueeze(0)
    x_len = x.size()[1] // 2
    even = x[:, torch.arange(0, x_len*2-1, 2).long(), :]
    odd = x[:, torch.arange(1, x_len*2, 2).long(), :]
    if skip_add == "add":
        return (even+odd) / 2, ((seq_len) / 2).int()
    else:
        return even, (seq_len / 2).int()

class EncoderRNN(nn.Module):
    def __init__(self, config):
        super(EncoderRNN, self).__init__()
        self.input_size = config["input_sequence_length"]
        self.hidden_size = config["encoder_hidden"]
        self.layers = config.get("encoder_layers", 1)
        self.dnn_layers = config.get("encoder_dnn_layers", 0)
        self.dropout = config.get("encoder_dropout", 0.)
        self.bi = config.get("bidirectional_encoder", False)
        if self.dnn_layers > 0:
            for i in range(self.dnn_layers):
                self.add_module('dnn_' + str(i), nn.Linear(
                    in_features=self.input_size if i == 0 else self.hidden_size,
                    out_features=self.hidden_size
                ))
        gru_input_dim = self.input_size if self.dnn_layers == 0 else self.hidden_size
        self.rnn = nn.LSTM(
            gru_input_dim,
            self.hidden_size,
            self.layers,
            dropout=self.dropout,
            bidirectional=self.bi,
            batch_first=True)
        self.gpu = config.get("gpu", False)

    def run_dnn(self, x):
        for i in range(self.dnn_layers):
            x = F.relu(getattr(self, 'dnn_'+str(i))(x))
        return x
    '''
    def forward(self, inputs, hidden, input_lengths):
        if self.dnn_layers > 0:
            inputs = self.run_dnn(inputs)
        x = inputs.unsqueeze(0)
        x = x.to(torch.float32)
        #x = torch.nn.utils.rnn.pack_padded_sequence(inputs, input_lengths, batch_first=True)
        #outputs, (hidden, cell)  = self.rnn(x)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=0.)

        if self.bi:
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        return hidden, cell
    '''

    def forward(self, inputs, hidden, input_lengths):
        if self.dnn_layers > 0:
            inputs = self.run_dnn(inputs)
        x = torch.nn.utils.rnn.pack_padded_sequence(inputs, input_lengths, batch_first=True)
        output, state = self.rnn(x, hidden)
        output, _ = pad_packed_sequence(output, batch_first=True, padding_value=0.)

        if self.bi:
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        return output, state

        

    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(2 if self.bi else 1, batch_size, self.hidden_size))
        if self.gpu:
            h0 = h0.cuda()
        return h0


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.batch_size = config["batch_size"]
        self.hidden_size = config["decoder_hidden"]
        embedding_dim = config.get("embedding_dim", None)
        self.embedding_dim = embedding_dim if embedding_dim is not None else self.hidden_size
        self.embedding = nn.Embedding(config.get("n_classes", 32), self.embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            input_size=self.embedding_dim+self.hidden_size if config['decoder'].lower() == 'bahdanau' else self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=config.get("decoder_layers", 1),
            dropout=config.get("decoder_dropout", 0),
            bidirectional=config.get("bidirectional_decoder", False),
            batch_first=True)
        if config['decoder'] != "RNN":
            self.attention = Attention(
                self.batch_size,
                self.hidden_size,
                method=config.get("attention_score", "dot"),
                mlp=config.get("attention_mlp_pre", False))

        self.gpu = config.get("gpu", False)
        self.decoder_output_fn = F.log_softmax if config.get('loss', 'NLL') == 'NLL' else None

    def forward(self, **kwargs):
        """ Must be overrided """
        raise NotImplementedError


class RNNDecoder(Decoder):
    def __init__(self, config):
        super(RNNDecoder, self).__init__(config)
        self.output_size = config.get("n_classes", 32)
        self.character_distribution = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, **kwargs):
        input = kwargs["input"]
        hidden = kwargs["hidden"]
    
        # RNN (Eq 7 paper)
        embedded = self.embedding(input).unsqueeze(0)
        rnn_input = torch.cat((embedded, hidden.unsqueeze(0)), 2)  # NOTE : Tf concats `lambda inputs, attention: array_ops.concat([inputs, attention], -1)`.
        # rnn_output, rnn_hidden = self.rnn(rnn_input.transpose(1, 0), hidden.unsqueeze(0))
        rnn_output, rnn_hidden = self.rnn(embedded.transpose(1, 0), hidden.unsqueeze(0))
        output = rnn_output.squeeze(1)
        output = self.character_distribution(output)

        if self.decoder_output_fn:
            output = self.decoder_output_fn(output, -1)

        return output, rnn_hidden.squeeze(0)


class BahdanauDecoder(Decoder):
    """
        Corresponds to BahdanauAttnDecoderRNN in Pytorch tuto
    """

    def __init__(self, config):
        super(BahdanauDecoder, self).__init__(config)
        self.output_size = config.get("n_classes", 32)
        self.character_distribution = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, **kwargs):
        """

        :param input: [B]
        :param prev_context: [B, H]
        :param prev_hidden: [B, H]
        :param encoder_outputs: [B, T, H]
        :return: output (B), context (B, H), prev_hidden (B, H), weights (B, T)
        """

        input = kwargs["input"]
        prev_hidden = kwargs["prev_hidden"]
        encoder_outputs = kwargs["encoder_outputs"]
        seq_len = kwargs.get("seq_len", None)

        # check inputs
        assert input.size() == torch.Size([self.batch_size])
        assert prev_hidden.size() == torch.Size([self.batch_size, self.hidden_size])

        # Attention weights
        weights = self.attention.forward(prev_hidden, encoder_outputs, seq_len)  # B x T
        context = weights.unsqueeze(1).bmm(encoder_outputs).squeeze(1)  # [B x H]

        # embed characters
        embedded = self.embedding(input).unsqueeze(0)
        assert embedded.size() == torch.Size([1, self.batch_size, self.embedding_dim])

        rnn_input = torch.cat((embedded, context.unsqueeze(0)), 2)

        outputs, hidden = self.rnn(rnn_input.transpose(1, 0), prev_hidden.unsqueeze(0)) # 1 x B x N, B x N

        # output = self.proj(torch.cat((outputs.squeeze(0), context), 1))
        output = self.character_distribution(outputs.squeeze(0))

        if self.decoder_output_fn:
            output = self.decoder_output_fn(output, -1)

        if len(output.size()) == 3:
            output = output.squeeze(1)

        return output, hidden.squeeze(0), weights


class LuongDecoder(Decoder):
    """
        Corresponds to AttnDecoderRNN
    """

    def __init__(self, config):
        super(LuongDecoder, self).__init__(config)
        self.output_size = config.get("n_classes", 32)
        self.character_distribution = nn.Linear(2*self.hidden_size, self.output_size)

    def forward(self, **kwargs):
        """

        :param input: [B]
        :param prev_context: [B, H]
        :param prev_hidden: [B, H]
        :param encoder_outputs: [B, T, H]
        :return: output (B, V), context (B, H), prev_hidden (B, H), weights (B, T)

        https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py
        TF says : Perform a step of attention-wrapped RNN.
        - Step 1: Mix the `inputs` and previous step's `attention` output via
          `cell_input_fn`.
        - Step 2: Call the wrapped `cell` with this input and its previous state.
        - Step 3: Score the cell's output with `attention_mechanism`.
        - Step 4: Calculate the alignments by passing the score through the
          `normalizer`.
        - Step 5: Calculate the context vector as the inner product between the
          alignments and the attention_mechanism's values (memory).
        - Step 6: Calculate the attention output by concatenating the cell output
          and context through the attention layer (a linear layer with
          `attention_layer_size` outputs).
        Args:
          inputs: (Possibly nested tuple of) Tensor, the input at this time step.
          state: An instance of `AttentionWrapperState` containing
            tensors from the previous time step.
        Returns:
          A tuple `(attention_or_cell_output, next_state)`, where:
          - `attention_or_cell_output` depending on `output_attention`.
          - `next_state` is an instance of `AttentionWrapperState`
             containing the state calculated at this time step.
        Raises:
          TypeError: If `state` is not an instance of `AttentionWrapperState`.

        """
        input = kwargs["input"]
        prev_hidden = kwargs["prev_hidden"]
        encoder_outputs = kwargs["encoder_outputs"]
        seq_len = kwargs.get("seq_len", None)

        # RNN (Eq 7 paper)
        embedded = self.embedding(input).unsqueeze(1) # [B, H]
        prev_hidden = prev_hidden.unsqueeze(0)
        #rnn_input = torch.cat((embedded, prev_context), -1) # NOTE : Tf concats `lambda inputs, attention: array_ops.concat([inputs, attention], -1)`.
        #rnn_output, hidden = self.rnn(rnn_input.transpose(1, 0), prev_hidden)
        #rnn_output, hidden = self.rnn(embedded, prev_hidden)
        outputs, (hidden, cell) = self.rnn(embedded, (prev_hidden, encoder_outputs))
        rnn_output = outputs.squeeze(1)

        # Attention weights (Eq 6 paper)
        weights = self.attention.forward(rnn_output, encoder_outputs, seq_len) # B x T
        context = weights.unsqueeze(1).bmm(encoder_outputs).squeeze(1)  # [B x N]

        # Projection (Eq 8 paper)
        # /!\ Don't apply tanh on outputs, it fucks everything up
        output = self.character_distribution(torch.cat((rnn_output, context), 1))

        

        # Apply log softmax if loss is NLL
        if self.decoder_output_fn:
            output = self.decoder_output_fn(output, -1)

        if len(output.size()) == 3:
            output = output.squeeze(1)
        
        return output, hidden.squeeze(0), weights


class Attention(nn.Module):
    """
    Inputs:
        last_hidden: (batch_size, hidden_size)
        encoder_outputs: (batch_size, max_time, hidden_size)
    Returns:
        attention_weights: (batch_size, max_time)
    """
    def __init__(self, batch_size, hidden_size, method="dot", mlp=False):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if method == 'dot':
            pass
        elif method == 'general':
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == "concat":
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
            self.va = nn.Parameter(torch.FloatTensor(batch_size, hidden_size))
        elif method == 'bahdanau':
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
            self.Ua = nn.Linear(hidden_size, hidden_size, bias=False)
            self.va = nn.Parameter(torch.FloatTensor(batch_size, hidden_size))
        else:
            raise NotImplementedError

        self.mlp = mlp
        if mlp:
            self.phi = nn.Linear(hidden_size, hidden_size, bias=False)
            self.psi = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, last_hidden, encoder_outputs, seq_len=None):
        batch_size, seq_lens, _ = encoder_outputs.size()
        if self.mlp:
            last_hidden = self.phi(last_hidden)
            encoder_outputs = self.psi(encoder_outputs)

        attention_energies = self._score(last_hidden, encoder_outputs, self.method)
        # attn_energies = Variable(torch.zeros(batch_size, seq_lens))  # B x S


        return F.softmax(attention_energies, -1)

    def _score(self, last_hidden, encoder_outputs, method):
        """
        Computes an attention score
        :param last_hidden: (batch_size, hidden_dim)
        :param encoder_outputs: (batch_size, max_time, hidden_dim)
        :param method: str (`dot`, `general`, `concat`)
        :return:
        """

        # assert last_hidden.size() == torch.Size([batch_size, self.hidden_size]), last_hidden.size()
        assert encoder_outputs.size()[-1] == self.hidden_size

        if method == 'dot':
            last_hidden = last_hidden.unsqueeze(-1)
            return encoder_outputs.bmm(last_hidden).squeeze(-1)

        elif method == 'general':
            x = self.Wa(last_hidden)
            x = x.unsqueeze(-1)
            return encoder_outputs.bmm(x).squeeze(-1)

        elif method == "concat":
            x = last_hidden.unsqueeze(1)
            x = F.tanh(self.Wa(torch.cat((x, encoder_outputs), 1)))
            return x.bmm(self.va.unsqueeze(2)).squeeze(-1)

        elif method == "bahdanau":
            x = last_hidden.unsqueeze(1)
            out = F.tanh(self.Wa(x) + self.Ua(encoder_outputs))
            return out.bmm(self.va.unsqueeze(2)).squeeze(-1)

        else:
            raise NotImplementedError

    def extra_repr(self):
        return 'score={}, mlp_preprocessing={}'.format(
            self.method, self.mlp)

class EncoderPyRNN(nn.Module):
    def __init__(self, config):
        super(EncoderPyRNN, self).__init__()
        self.input_size = config["n_channels"]
        self.hidden_size = config["encoder_hidden"]
        self.n_layers = config.get("encoder_layers", 1)
        self.dnn_layers = config.get("encoder_dnn_layers", 0)
        self.dropout = config.get("encoder_dropout", 0.)
        self.bi = config.get("bidirectional_encoder", False)
        self.skip_add = config.get("skip_add_pyramid_encoder", "add")
        self.gpu = config.get("gpu", False)

        if self.dnn_layers > 0:
            for i in range(self.dnn_layers):
                self.add_module('dnn_' + str(i), nn.Linear(
                    in_features=self.input_size if i == 0 else self.hidden_size,
                    out_features=self.hidden_size
                ))
        gru_input_dim = self.input_size if self.dnn_layers == 0 else self.hidden_size

        for i in range(self.n_layers):
            self.add_module('pRNN_' + str(i), nn.GRU(
                input_size=gru_input_dim if i == 0 else self.hidden_size,
                hidden_size=self.hidden_size,
                dropout=self.dropout,
                bidirectional=self.bi,
                batch_first=True))

    def run_dnn(self, x):
        for i in range(self.dnn_layers):
            x = F.relu(getattr(self, 'dnn_'+str(i))(x))
        return x

    def run_pRNN(self, inputs, hidden, input_lengths):
        """
        :param input: (batch, seq_len, input_size)
        :param hidden: (num_layers * num_directions, batch, hidden_size)
        :return:
        """
        for i in range(self.n_layers):
            x = torch.nn.utils.rnn.pack_padded_sequence(inputs, input_lengths, batch_first=True)
            output, hidden = getattr(self, 'pRNN_'+str(i))(x, hidden)
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=0.)
            hidden = hidden

            if self.bi:
                output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]

            if i < self.n_layers - 1:
                inputs, input_lengths = skip_add_pyramid(output, input_lengths, self.skip_add)

        return output, hidden, input_lengths

    def forward(self, inputs, hidden, input_lengths):
        if self.dnn_layers > 0:
            inputs = self.run_dnn(inputs)

        outputs, hidden, input_lengths = self.run_pRNN(inputs, hidden, input_lengths)

        if self.bi:
            hidden = torch.sum(hidden, 0)

        return outputs, hidden, input_lengths

    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(2 if self.bi else 1, batch_size, self.hidden_size))
        if self.gpu:
            h0 = h0.cuda()
        return h0

class seq2seq_prefetch(nn.Module):
    """
        Sequence to sequence module
    """

    def __init__(self, config):
        super(seq2seq_prefetch, self).__init__()
        self.SOS = config.get("start_index", 1),
        self.vocab_size = config.get("n_classes", 32)
        self.batch_size = config.get("batch_size", 1)
        self.sampling_prob = config.get("sampling_prob", 0.)
        self.gpu = config.get("gpu", False)
        self.config = config
        # Encoder
        if config["encoder"] == "PyRNN":
            self._encoder_style = "PyRNN"
            self.encoder = EncoderPyRNN(config)
        else:
            self._encoder_style = "RNN"
            self.encoder = EncoderRNN(config)

        # Decoder
        self.use_attention = config["decoder"] != "RNN"
        if config["decoder"] == "Luong":
            self.decoder = LuongDecoder(config)
        elif config["decoder"] == "Bahdanau":
            self.decoder = BahdanauDecoder(config)
        else:
            self.decoder = RNNDecoder(config)

        if config.get('loss') == 'cross_entropy':
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
            config['loss'] = 'cross_entropy'
        elif config.get('loss') == 'Intersection_over_Union':
            self.loss_fn = Chamfer1DLoss()
            config['loss'] = 'Intersection_over_Union'
        else:
            self.loss_fn = torch.nn.NLLLoss(ignore_index=0)
            config['loss'] = 'NLL'
        self.loss_type = config['loss']
        print(config)

    def encode(self, x, x_len):

        batch_size = x.size()[0]
        init_state = self.encoder.init_hidden(batch_size)
        if self._encoder_style == "PyRNN":
            encoder_outputs, encoder_state, input_lengths = self.encoder.forward(x, init_state, x_len)
        else:
            encoder_outputs, encoder_state = self.encoder.forward(x, init_state, x_len)

        #assert encoder_outputs.size()[0] == self.batch_size, encoder_outputs.size()
        assert encoder_outputs.size()[-1] == self.decoder.hidden_size

        if self._encoder_style == "PyRNN":
            return encoder_outputs, encoder_state.squeeze(0), input_lengths
        return encoder_outputs, encoder_state.squeeze(0)

    def decode(self, encoder_outputs, encoder_hidden, targets, targets_lengths, input_lengths):
        """
        Args:
            encoder_outputs: (B, T, H)
            encoder_hidden: (B, H)
            targets: (B, L)
            targets_lengths: (B)
            input_lengths: (B)
        Vars:
            decoder_input: (B)
            decoder_context: (B, H)
            hidden_state: (B, H)
            attention_weights: (B, T)
        Outputs:
            alignments: (L, T, B)
            logits: (B*L, V)
            labels: (B*L)
        """

        batch_size = encoder_outputs.size()[0]
        max_length = targets.size()[1]
        # decoder_attns = torch.zeros(batch_size, MAX_LENGTH, MAX_LENGTH)
        decoder_input = Variable(torch.LongTensor([self.SOS] * batch_size)).squeeze(-1)
        decoder_context = encoder_outputs.transpose(1, 0)[-1]
        decoder_hidden = encoder_hidden

        alignments = Variable(torch.zeros(max_length, encoder_outputs.size(1), batch_size))
        logits = Variable(torch.zeros(max_length, batch_size, self.decoder.output_size))

        if self.gpu:
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()
            logits = logits.cuda()

        for t in range(max_length):

            # The decoder accepts, at each time step t :
            # - an input, [B]
            # - a context, [B, H]
            # - an hidden state, [B, H]
            # - encoder outputs, [B, T, H]

            #check_size(decoder_input, self.batch_size)
            #check_size(decoder_hidden, self.batch_size, self.decoder.hidden_size)

            # The decoder outputs, at each time step t :
            # - an output, [B]
            # - a context, [B, H]
            # - an hidden state, [B, H]
            # - weights, [B, T]

            if self.use_attention:
                #check_size(decoder_context, self.batch_size, self.decoder.hidden_size)
                outputs, decoder_hidden, attention_weights = self.decoder.forward(
                    input=decoder_input.long(),
                    prev_hidden=decoder_hidden,
                    encoder_outputs=encoder_outputs,
                    seq_len=input_lengths)
                alignments[t] = attention_weights.transpose(1, 0)
            else:
                outputs, hidden = self.decoder.forward(
                    input=decoder_input.long(),
                    hidden=decoder_hidden)

            # print(outputs[0])
            logits[t] = outputs

            use_teacher_forcing = random.random() > self.sampling_prob

            if use_teacher_forcing and self.training:
                decoder_input = targets[:, t]

            # SCHEDULED SAMPLING
            # We use the target sequence at each time step which we feed in the decoder
            else:
                # TODO Instead of taking the direct one-hot prediction from the previous time step as the original paper
                # does, we thought it is better to feed the distribution vector as it encodes more information about
                # prediction from previous step and could reduce bias.
                topv, topi = outputs.data.topk(1)
                decoder_input = topi.squeeze(-1).detach()


        labels = targets.contiguous().view(-1)

        if self.loss_type == 'NLL': # ie softmax already on outputs
            mask_value = -float('inf')
            print(torch.sum(logits, dim=2))
        else:
            mask_value = 0

        logits = mask_3d(logits.transpose(1, 0), targets_lengths, mask_value)
        logits = logits.contiguous().view(-1, self.vocab_size)
        

        return logits, labels.long(), alignments

    @staticmethod
    def custom_loss(logits, labels):

        # create a mask by filtering out all tokens that ARE NOT the padding token
        tag_pad_token = 0
        mask = (labels > tag_pad_token).float()

        # count how many tokens we have
        nb_tokens = int(torch.sum(mask).data[0])

        # pick the values for the label and zero out the rest with the mask
        logits = logits[range(logits.shape[0]), labels] * mask

        # compute cross entropy loss which ignores all <PAD> tokens
        ce_loss = -torch.sum(logits) / nb_tokens

        return ce_loss

    def step(self, batch):
        x, y = batch
        if self.gpu:
            x = x.cuda()
            y = y.cuda()
        
        x_len = self.config.get("n_channels", 1)
        y_len = self.config.get("n_classes", 1)
        if self._encoder_style == "PyRNN":
            encoder_out, encoder_state, x_len = self.encode(x, x_len)
        else:
            encoder_out, encoder_state = self.encode(x, x_len)
        logits, labels, alignments = self.decode(encoder_out, encoder_state, y, y_len, x_len)
        return logits, labels, alignments

    def loss(self, batch):
        logits, labels, alignments = self.step(batch)
        if self.config.get('loss') == 'Intersection_over_Union':
            accuracy, loss = iou_pytorch(logits, labels)
            loss = loss 
        else:
            loss = self.loss_fn(logits, labels)
        print(accuracy)
        # loss2 = self.custom_loss(logits, labels)
        return loss, logits, labels, alignments



