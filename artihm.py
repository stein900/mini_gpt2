# mini_gpt2.py
# Mini GPT-2 byte-level (0..255 + EOS=256) pour additions "A + B = C\n"
# - Dataset ligne-par-ligne avec MASQUAGE (P-1) sur le prompt "A + B = "
# - Greedy (argmax), arrêt sur '\n' (byte 10) ou EOS (256)
# - 100% neurones, aucun vocab spécial

import os, math, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# TF32 (API récente)
try:
    torch.backends.cuda.matmul.fp32_precision = 'high'
except Exception:
    pass
torch.set_float32_matmul_precision('high')

# ---------------------
# Tokenizer byte-level
# ---------------------
class ByteTokenizer:
    def __init__(self, add_eos=True):
        self.default_add_eos = add_eos
        self.vocab_size = 257 if add_eos else 256
        self.EOS = 256 if add_eos else None

    def encode(self, text: str, add_eos=None):
        if add_eos is None:
            add_eos = self.default_add_eos
        ids = list(text.encode('utf-8'))
        if add_eos and self.EOS is not None:
            ids.append(self.EOS)
        return ids

    def decode(self, ids):
        if self.EOS is not None:
            ids = [i for i in ids if i != self.EOS]
        return bytes(ids).decode('utf-8', errors='ignore')

# ---------------------
# Modèle
# ---------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout, block_size):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer('mask', mask.view(1,1,block_size,block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, d_model, n_head, dropout, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp  = MLP(d_model, dropout)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MiniGPT2(nn.Module):
    def __init__(self, vocab_size=257, d_model=128, n_layer=4, n_head=4, block_size=32, dropout=0.1, tie_weights=True):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_head, dropout, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"sequence length {T} > block size {self.block_size}")
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks: x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate_greedy(self, idx, max_new_tokens=16, eos_id=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # (B,1)
            idx = torch.cat((idx, next_id), dim=1)
            t = next_id.item()
            if (eos_id is not None and t == eos_id) or t == 10:  # 10 == '\n'
                break
        return idx

# ---------------------
# Dataset ligne-par-ligne avec masquage (corrigé)
# ---------------------
# ---------------------
# Dataset ligne-par-ligne (add ET sub) avec masquage (P-1)
# ---------------------
class AddLineDataset(torch.utils.data.Dataset):
    """
    Ligne attendue (format strict) : "A + B = C" OU "A - B = C"
    Exemple complet stocké sous forme prompt="A <op> B = " et answer="C\n".
    Entraînement avec labels[:P-1] = -100 pour ne PAS apprendre à recopier le prompt,
    et apprendre dès le 1er token de la réponse.
    """
    def __init__(self, path, tok: ByteTokenizer, max_len_hint=32):
        self.tok = tok
        with open(path, 'r', encoding='utf-8') as f:
            # on enlève juste le \n de fin; les lignes vides sont ignorées
            lines = [ln.rstrip('\n') for ln in f if ln.strip()]
        self.samples = []
        max_len = 0

        for ln in lines:
            parts = ln.split(' ')
            # attendu: [A, <op>, B, '=', C]
            assert len(parts) == 5, f"Ligne invalide (5 tokens attendus): {ln!r}"
            a, op, b, eq, c = parts
            assert op in ['+', '-'], f"Opérateur inconnu (attendu + ou -): {ln!r}"
            assert eq == '=', f"Manque '=' dans: {ln!r}"

            prompt = f"{a} {op} {b} = "
            answer = f"{c}\n"

            p_ids = self.tok.encode(prompt, add_eos=False)
            a_ids = self.tok.encode(answer, add_eos=False)
            ids = p_ids + a_ids
            max_len = max(max_len, len(ids))
            self.samples.append((p_ids, a_ids))

        self.max_len = max(max_len, max_len_hint)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p_ids, a_ids = self.samples[idx]
        ids = p_ids + a_ids

        # entrée/sortie décalées
        inp = ids[:-1]
        lab = ids[1:]

        # Masquage (P-1) : ne pas entraîner sur la recopie du prompt,
        # mais OUI sur le dernier espace après '=' et sur tous les tokens de la réponse.
        P = len(p_ids)
        mask_len = max(0, P - 1)
        for i in range(mask_len):
            lab[i] = -100

        # Padding fixe jusqu'à max_len-1
        PAD = 32  # espace (évite de biaiser vers '0')
        while len(inp) < self.max_len - 1:
            inp.append(PAD)
            lab.append(-100)

        return torch.tensor(inp, dtype=torch.long), torch.tensor(lab, dtype=torch.long)


# ---------------------
# Utils
# ---------------------
def set_seed(seed=1337):
    import random, numpy as np
    random.seed(seed); os.environ["PYTHONHASHSEED"]=str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    try: np.random.seed(seed)
    except: pass

# ---------------------
# Main (train + démo)
# ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="additions_1_9.txt")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--out", type=str, default="mini_gpt2.pt")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--prompt", type=str, default="4 + 5 = ")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | CUDA: {torch.version.cuda}")

    tok = ByteTokenizer(add_eos=True)

    ds = AddLineDataset(args.data, tok, max_len_hint=32)
    block_size = ds.max_len
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        pin_memory=(device=="cuda"), num_workers=0
    )

    model = MiniGPT2(
        vocab_size=tok.vocab_size, d_model=args.d_model, n_layer=args.n_layer,
        n_head=args.n_head, block_size=block_size, dropout=args.dropout, tie_weights=True
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.steps > 0:
        model.train()
        it = iter(loader); t0 = time.time()
        for step in range(1, args.steps+1):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader); x, y = next(it)
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 100 == 0:
                dt = time.time()-t0; t0=time.time()
                print(f"step {step}/{args.steps} | loss {loss.item():.4f} | {dt:.1f}s")

        torch.save({
            "model": model.state_dict(),
            "vocab_size": tok.vocab_size,
            "d_model": args.d_model,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "block_size": block_size,
            "dropout": args.dropout,
        }, args.out)
        print(f"Checkpoint → {args.out}")
    else:
        print("steps=0 : inférence uniquement (pas de sauvegarde).")

    # démo greedy
    model.eval()
    ctx = torch.tensor([tok.encode(args.prompt, add_eos=False)], dtype=torch.long, device=device)
    out = model.generate_greedy(ctx, max_new_tokens=args.max_new_tokens, eos_id=tok.EOS)[0].tolist()
    print("\n=== échantillon ===\n" + tok.decode(out))

if __name__ == "__main__":
    main()
