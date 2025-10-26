# test_additions.py
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

@torch.no_grad()
def predict_sum(model, tok, a, b, device, max_new_tokens=8):
    prompt = f"{a} + {b} = "
    ctx = torch.tensor([tok.encode(prompt, add_eos=False)], dtype=torch.long, device=device)
    out = model.generate_greedy(ctx, max_new_tokens=max_new_tokens, eos_id=tok.EOS)[0].tolist()
    txt = tok.decode(out)
    comp = txt[len(prompt):].split("\n", 1)[0].lstrip()
    m = re.match(r"(\d+)", comp) or re.search(r"(\d+)", comp)
    return (int(m.group(1)) if m else None), comp

def eval_random(model, tok, lo, hi, n_tests, device):
    ok = 0
    for _ in range(n_tests):
        a = random.randint(lo, hi); b = random.randint(lo, hi)
        pred, _ = predict_sum(model, tok, a, b, device)
        ok += int(pred == a + b)
    return ok, n_tests, ok / max(1, n_tests)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="mini_gpt2.pt")
    ap.add_argument("--lo", type=int, default=1)
    ap.add_argument("--hi", type=int, default=9)
    ap.add_argument("--tests", type=int, default=100)
    ap.add_argument("--repl", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | CUDA: {torch.version.cuda}")

    model, tok = load_model(args.ckpt, device)

    if args.repl:
        print("Entrer des requêtes du type: 4 + 5   (Ctrl+C pour quitter)")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye."); break
            nums = re.findall(r"\d+", line)
            if len(nums) < 2:
                print("format attendu: a + b"); continue
            a, b = int(nums[0]), int(nums[1])
            pred, raw = predict_sum(model, tok, a, b, device, args.max_new_tokens)
            print(f"{a} + {b} = {pred}   (raw: {raw!r})")
    else:
        ok, total, acc = eval_random(model, tok, args.lo, args.hi, args.tests, device)
        print(f"Accuracy: {ok}/{total} = {acc:.2%} sur [{args.lo}, {args.hi}]")

if __name__ == "__main__":
    main()
