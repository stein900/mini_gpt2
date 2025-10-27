# test_arith.py
import re, random, argparse, torch
from mini_gpt2 import MiniGPT2, ByteTokenizer

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    tok = ByteTokenizer(add_eos=True)
    model = MiniGPT2(
        vocab_size=ckpt.get("vocab_size", tok.vocab_size),
        d_model=ckpt["d_model"],
        n_layer=ckpt["n_layer"],
        n_head=ckpt["n_head"],
        block_size=ckpt["block_size"],
        dropout=ckpt.get("dropout", 0.1),
        tie_weights=True,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, tok

def _normalize_query(s: str):
    """Retourne (a, op, b) en normalisant les espaces et l’opérateur ('+'|'-')."""
    s = s.strip()
    # match "a op b" avec ou sans espaces (ex: "5-2", "5 - 2")
    m = re.match(r"^\s*(\d+)\s*([+-])\s*(\d+)\s*$", s)
    if not m:
        return None
    a = int(m.group(1))
    op = m.group(2)
    b = int(m.group(3))
    return a, op, b

@torch.no_grad()
def predict_arith(model, tok, a: int, op: str, b: int, device, max_new_tokens=8):
    """Construit le prompt EXACT comme dans le corpus: 'A <op> B = ' puis greedy jusqu'à \\n/EOS."""
    assert op in ['+','-']
    prompt = f"{a} {op} {b} = "
    ctx = torch.tensor([tok.encode(prompt, add_eos=False)], dtype=torch.long, device=device)
    out = model.generate_greedy(ctx, max_new_tokens=max_new_tokens, eos_id=tok.EOS)[0].tolist()
    txt = tok.decode(out)
    # On ne garde que la complétion après le prompt, avant le premier '\n'
    comp = txt[len(prompt):].split("\n", 1)[0].strip()
    # Résultat peut être négatif (si ton corpus en contient)
    m = re.match(r"^-?\d+", comp)
    pred = int(m.group(0)) if m else None
    return pred, comp, prompt

def gold(a: int, op: str, b: int) -> int:
    return a + b if op == '+' else a - b

def eval_random(model, tok, lo, hi, n_tests, device, ops=('+','-'), max_new_tokens=8, seed=1337):
    rnd = random.Random(seed)
    counts = { '+': 0, '-': 0 }
    hits   = { '+': 0, '-': 0 }
    for _ in range(n_tests):
        a = rnd.randint(lo, hi)
        b = rnd.randint(lo, hi)
        op = rnd.choice(ops)
        # si tu n'autorises pas les négatifs en test pour "-", force a>=b :
        if op == '-' and a < b:
            a, b = b, a
        pred, _, _ = predict_arith(model, tok, a, op, b, device, max_new_tokens)
        g = gold(a, op, b)
        counts[op] += 1
        hits[op]   += int(pred == g)
    total = sum(counts.values())
    correct = sum(hits.values())
    acc_total = correct / total if total else 0.0
    acc_add = (hits['+'] / counts['+']) if counts['+'] else None
    acc_sub = (hits['-'] / counts['-']) if counts['-'] else None
    return {
        'total': total, 'correct': correct, 'acc_total': acc_total,
        'counts': counts, 'hits': hits, 'acc_add': acc_add, 'acc_sub': acc_sub
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="arith.pt")
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=99)
    ap.add_argument("--tests", type=int, default=200)
    ap.add_argument("--repl", action="store_true", help="mode interactif")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--ops", type=str, default="both", choices=["add","sub","both"],
                    help="jeux d'opérations en mode eval_random")
    ap.add_argument("--nonneg_sub", type=int, default=1,
                    help="en REPL, si op='-', force a>=b si =1 (comme le corpus non-négatif)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | CUDA: {torch.version.cuda}")

    model, tok = load_model(args.ckpt, device)

    if args.repl:
        print("Entrer des requêtes du type: 5-2  ou  12 + 7   (Ctrl+C pour quitter)")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye."); break
            parsed = _normalize_query(line)
            if not parsed:
                print("Format attendu:  a+b  |  a - b  (espaces optionnels)")
                continue
            a, op, b = parsed
            if op == '-' and args.nonneg_sub and a < b:
                a, b = b, a
            pred, raw, prompt = predict_arith(model, tok, a, op, b, device, args.max_new_tokens)
            print(f"{prompt}{pred}   (raw: {raw!r})")
    else:
        ops = { "add": ('+',), "sub": ('-',), "both": ('+','-') }[args.ops]
        stats = eval_random(model, tok, args.lo, args.hi, args.tests, device, ops, args.max_new_tokens, args.seed)
        print(f"Total accuracy: {stats['correct']}/{stats['total']} = {stats['acc_total']:.2%}")
        if stats['counts']['+'] > 0:
            print(f"  +  {stats['hits']['+']}/{stats['counts']['+']} = {stats['acc_add']:.2%}")
        if stats['counts']['-'] > 0:
            print(f"  -  {stats['hits']['-']}/{stats['counts']['-']} = {stats['acc_sub']:.2%}")

if __name__ == "__main__":
    main()
