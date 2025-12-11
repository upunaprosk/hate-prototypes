# highway_opt.py
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from typing import Optional, Tuple, List

from transformers.models.opt.configuration_opt import OPTConfig
from transformers.models.opt.modeling_opt import OPTPreTrainedModel, OPTDecoder


# ---------------- utilities ----------------
def entropy(logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits, dim=-1).clamp_min(1e-8)
    return -(p * p.log()).sum(dim=-1)


def _select_last_nonpad(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Select last non-padding token per sample.
    hidden_states: (B, T, H)
    attention_mask: (B, T_total) where T_total may be past_len + T (we use only the last T).
    Returns: (B, H)
    """
    B, T, H = hidden_states.shape
    if attention_mask is None:
        idx = torch.full((B,), T - 1, dtype=torch.long, device=hidden_states.device)
    else:
        am = attention_mask[:, -T:]  # drop past_kv columns if any
        positions = torch.arange(T, device=hidden_states.device).unsqueeze(0)  # (1, T)
        masked_pos = am * positions
        idx = masked_pos.argmax(dim=1)  # last index where mask==1
        empty = am.sum(dim=1) == 0
        if empty.any():
            idx = torch.where(empty, torch.full_like(idx, T - 1), idx)
    return hidden_states[torch.arange(B, device=hidden_states.device), idx, :]


class HighwayException(Exception):
    def __init__(self, message, exit_layer: int):
        self.message = message
        self.exit_layer = exit_layer  # 1-based index


# ---------------- per-layer highway head ----------------
class OPTHighway(nn.Module):
    def __init__(self, config: OPTConfig):
        super().__init__()
        hid = config.hidden_size
        self.pooler = nn.Linear(hid, hid)
        self.act = nn.Tanh()
        p_drop = getattr(config, "hidden_dropout_prob", getattr(config, "dropout", 0.1))
        self.dropout = nn.Dropout(p_drop)
        self.classifier = nn.Linear(hid, config.num_labels)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        token_states = _select_last_nonpad(hidden_states, attention_mask)  # (B, H)
        pooled = self.act(self.pooler(token_states))
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits, pooled


# ---------------- wrapper around decoder (no re-registration) ----------------
class OPTDecoderHighway(nn.Module):
    def __init__(self, config: OPTConfig, decoder: OPTDecoder):
        super().__init__()
        self.config = config

        # ---- PABEE flags from config (default off) ----
        self.use_pabee = getattr(config, "use_pabee", False)
        self.patience = int(getattr(config, "patience", 3))

        # Store decoder without re-registering
        self.__dict__["_decoder"] = decoder
        self._layers = tuple(decoder.layers)

        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions

        self.highway = nn.ModuleList([OPTHighway(config) for _ in range(config.num_hidden_layers)])
        self.early_exit_entropy: List[float] = [float(-1.0)] * config.num_hidden_layers

    def set_early_exit_entropy(self, x):
        if isinstance(x, (float, int)):
            self.early_exit_entropy = [float(x)] * len(self.early_exit_entropy)
        else:
            assert len(x) == len(self.early_exit_entropy)
            self.early_exit_entropy = [float(v) for v in x]

    # --- helpers for version differences ---
    def _build_causal_mask(self, attention_mask, input_shape, inputs_embeds, pkv_len):
        dec: OPTDecoder = self.__dict__["_decoder"]
        try:
            return dec._prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, pkv_len)
        except AttributeError:
            try:
                from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
                return _prepare_4d_causal_attention_mask(attention_mask, input_shape, inputs_embeds, pkv_len)
            except Exception:
                from transformers.modeling_attn_mask_utils import _make_causal_mask, _expand_mask
                cm = _make_causal_mask(input_shape, inputs_embeds.dtype, past_key_values_length=pkv_len).to(inputs_embeds.device)
                if attention_mask is not None:
                    cm = cm + _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
                return cm

    def _pos_embeds(self, attention_mask, pkv_len):
        dec: OPTDecoder = self.__dict__["_decoder"]
        if hasattr(dec, "_embed_positions"):
            return dec._embed_positions(attention_mask, pkv_len)
        return dec.embed_positions(attention_mask, pkv_len)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        dec: OPTDecoder = self.__dict__["_decoder"]
        output_attentions = output_attentions or self.output_attentions
        output_hidden_states = output_hidden_states or self.output_hidden_states

        # Reset PABEE state each forward
        if not self.training and self.use_pabee:
            self._pabee_last_pred = None
            self._pabee_counter = None

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both.")
        if input_ids is not None:
            input_shape = input_ids.size()
            inputs_embeds = dec.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("input_ids or inputs_embeds required.")

        bsz, seq_len = input_shape
        pkv_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        mask_seq_len = pkv_len + seq_len

        if attention_mask is None:
            attention_mask = torch.ones(bsz, mask_seq_len, device=inputs_embeds.device)
        elif attention_mask.shape[1] != mask_seq_len:
            raise ValueError(f"attention_mask length {attention_mask.shape[1]} != {mask_seq_len} (past+current)")

        causal_attn_mask = self._build_causal_mask(attention_mask, input_shape, inputs_embeds, pkv_len)
        pos_embeds = self._pos_embeds(attention_mask, pkv_len)

        if dec.project_in is not None:
            inputs_embeds = dec.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        all_hidden_states = () if output_hidden_states else None
        all_highway_exits: Tuple = ()

        for i, layer in enumerate(self._layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask=causal_attn_mask,
                layer_head_mask=(head_mask[i] if head_mask is not None else None),
                past_key_value=(past_key_values[i] if past_key_values is not None else None),
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]

            # highway head
            h_logits, _ = self.highway[i](hidden_states, attention_mask=attention_mask)
            if not self.training:
                h_ent = entropy(h_logits).mean()
                all_highway_exits += ((h_logits, hidden_states, h_ent),)

                # ---------- PABEE ----------
                if self.use_pabee:
                    probs = torch.softmax(h_logits, dim=-1)
                    preds = torch.argmax(probs, dim=-1)
                    if (self._pabee_last_pred is None) or (self._pabee_counter is None):
                        self._pabee_last_pred = preds.clone()
                        self._pabee_counter = torch.ones_like(preds)
                    else:
                        same = preds == self._pabee_last_pred
                        self._pabee_counter = torch.where(
                            same, self._pabee_counter + 1, torch.ones_like(self._pabee_counter)
                        )
                        self._pabee_last_pred = preds.clone()
                    if torch.all(self._pabee_counter >= self.patience):
                        new_output = (h_logits,)
                        if output_hidden_states:
                            new_output += (all_hidden_states,)
                        new_output += (None,)
                        new_output += (all_highway_exits,)
                        self._pabee_last_pred = None
                        self._pabee_counter = None
                        raise HighwayException(new_output, i + 1)

                # ---------- DeeBERT ----------
                elif h_ent.item() < self.early_exit_entropy[i]:
                    new_output = (h_logits,)
                    if output_hidden_states:
                        new_output += (all_hidden_states,)
                    new_output += (None,)
                    new_output += (all_highway_exits,)
                    raise HighwayException(new_output, i + 1)

            else:
                all_highway_exits += ((h_logits, hidden_states),)

        if dec.final_layer_norm is not None:
            hidden_states = dec.final_layer_norm(hidden_states)
        if dec.project_out is not None:
            hidden_states = dec.project_out(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return hidden_states, all_hidden_states, all_highway_exits


# ---------------- main model ----------------
class OPTForSequenceClassificationHighway(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.num_layers = config.num_hidden_layers

        self.decoder = OPTDecoder(config)
        self.highway = OPTDecoderHighway(config, self.decoder)

        out_dim = getattr(config, "word_embed_proj_dim", config.hidden_size)
        p_drop = getattr(config, "hidden_dropout_prob", getattr(config, "dropout", 0.1))
        self.dropout = nn.Dropout(p_drop)
        self.classifier = nn.Linear(out_dim, config.num_labels)

        self.post_init()

    def set_early_exit_entropy(self, x):
        self.highway.set_early_exit_entropy(x)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        output_layer: int = -1,
        train_highway: bool = False,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: bool = True,
    ):
        exit_layer = self.num_layers
        try:
            hidden_states, all_hidden, all_highways = self.highway(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache or False,
                output_attentions=output_attentions or False,
                output_hidden_states=output_hidden_states or False,
            )
            token_states = _select_last_nonpad(hidden_states, attention_mask)
            logits = self.classifier(self.dropout(token_states))
            outputs = (logits, all_hidden, None, all_highways)
        except HighwayException as e:
            outputs = e.message
            exit_layer = e.exit_layer
            logits = outputs[0]

        entropies = None
        if not self.training:
            orig_entropy = entropy(logits).mean().item()
            highway_ents: List[float] = []
            if outputs[-1] is not None:
                for hx in outputs[-1]:
                    if len(hx) >= 3:
                        highway_ents.append(float(hx[2].item()))
            entropies = (orig_entropy, highway_ents)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = MSELoss()(logits.view(-1), labels.view(-1))
            else:
                loss = CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))

            if train_highway and (outputs[-1] is not None):
                h_losses = []
                for hx in outputs[-1][:-1]:
                    h_logits = hx[0]
                    if self.num_labels == 1:
                        h_losses.append(MSELoss()(h_logits.view(-1), labels.view(-1)))
                    else:
                        h_losses.append(CrossEntropyLoss()(h_logits.view(-1, self.num_labels), labels.view(-1)))
                if h_losses:
                    loss = sum(h_losses)

        if (output_layer is not None) and (output_layer >= 0) and (outputs[-1] is not None):
            highway_logits_all = [hx[0] for hx in outputs[-1]]
            idx = max(0, min(output_layer, len(highway_logits_all) - 1))
            logits = highway_logits_all[idx]

        if return_dict:
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": outputs[1],
                "attentions": None,
                "entropies": entropies,
                "exit_layer": exit_layer,
            }
        else:
            return (loss, logits, outputs[1], None, entropies, exit_layer)
