import torch
import torch.nn as nn

from text_encoder import PositionalEmbedding


class TextDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len=77,
        embed_dim=512,
        hidden_dim=768,
        n_layers=6,
        n_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = PositionalEmbedding(hidden_dim, max_seq_len)

        self.image_projection = nn.Linear(embed_dim, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nheads=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.output_projections = nn.Linear(hidden_dim, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, image_features, captions=None, attention_mask=None):
        batch_size = image_features.shape[0]

        memory = self.image_projection(image_features).unsqueeze(1)

        if self.training and captions is not None:
            token_embeddings = self.token_embedding(captions)
            seq_len = token_embeddings.size(1)
            position_embeddings = self.position_embedding[:, :seq_len, :]

            embeddings = token_embeddings + position_embeddings

            tgt_mask = self.generate_square_subsequent_mask(seq_len).to(
                captions.device
            )

            output = self.decoder(
                tgt=embeddings,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=(
                    ~attention_mask if attention_mask is not None else None
                ),
            )

            logits = self.output_projections(output)

            return logits

        else:
            device = image_features.device
            input_ids = torch.zeros(
                (batch_size, 1), dtype=torch.long, device=device
            )

            generated_ids = []

            for i in range(self.max_seq_len):
                token_embeddings = self.token_embedding(input_ids)
                position_embeddings = self.position_embedding[
                    :, : input_ids.size(1), :
                ]
                embeddings = token_embeddings + position_embeddings

                tgt_mask = self.generate_square_subsequent_mask(
                    input_ids.size(1)
                ).to(device)

                output = self.decoder(
                    tgt=embeddings, memory=memory, tgt_mask=tgt_mask
                )
                logits = self.output_projections(output[:, -1])
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

                generated_ids.append(next_token)
                input_ids = torch.cat([input_ids, next_token], dim=1)

                if (next_token == 1).all():
                    break

            return torch.cat(generated_ids, dim=1)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask
