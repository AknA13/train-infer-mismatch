"""Stage 4: single-GPU GRPO with vLLM rollouts, under one mismatch treatment.

    python -m rl.grpo --arm none
    python -m rl.grpo --arm tis --mode tis
    python -m rl.grpo --arm fp16 --dtype fp16
    python -m rl.grpo --arm fp32head --fp32-head both
    python -m rl.grpo --arm bi --batch-invariant

One process, one GPU: vLLM engine (rollouts) + trainer (bf16 params, fp32
master). Every step: sample G completions per prompt, verify against gsm8k
gold, group-normalise rewards, one gradient step, push weights to the engine,
reset its prefix cache. The MismatchMonitor sees every token of every step, so
the mismatch is measured *as training proceeds*, not only at step 0.

Resumable: checkpoints every --ckpt-every steps to $TIM_DATA_ROOT/rl/<arm>/ckpt
(latest only), step log appended to steps.jsonl, and the data cursor and RNG
are restored so a preempted run continues with the same prompt order.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# must precede any vllm import
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
if "--batch-invariant" in sys.argv:
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from mismatch.common import build_prompt, load_problems, verify_answer
from mismatch.monitor import MismatchMonitor
from rl import losses as L


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="run name (directory under $TIM_DATA_ROOT/rl)")
    ap.add_argument("--model", default=C.MODEL_ID)
    ap.add_argument("--mode", default="none", choices=L.MODES)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--fp32-head", default="none", choices=["none", "trainer", "both"])
    ap.add_argument("--batch-invariant", action="store_true")
    ap.add_argument("--clip", type=float, default=C.TIS_CLIP)
    ap.add_argument("--prompts-per-step", type=int, default=32)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gmu", type=float, default=0.40)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--no-prefix-caching", action="store_true")
    ap.add_argument("--stop-after-sec", type=float, default=0, help="checkpoint and exit before the SLURM wall")
    return ap.parse_args()


def log_line(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")


def main():
    args = parse()
    run_dir = C.RL_DIR / args.arm
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=1))
    steps_path = run_dir / "steps.jsonl"
    ckpt_dir = run_dir / "ckpt"
    t_start = time.time()

    from transformers import AutoTokenizer
    from rl.engine import RolloutEngine
    from rl.policy import Policy
    if args.batch_invariant:
        from vllm.model_executor.layers.batch_invariant import enable_batch_invariant_mode
        enable_batch_invariant_mode()          # trainer-side ops too (same process)

    tok = AutoTokenizer.from_pretrained(args.model)
    train = load_problems("gsm8k", "train")
    test = load_problems("gsm8k", "test", n=args.eval_n, seed=args.seed)
    rng = random.Random(args.seed)
    order = list(range(len(train))); rng.shuffle(order)
    test_ids = [tok(build_prompt(tok, p["problem"], thinking=not args.no_thinking), add_special_tokens=False)["input_ids"] for p in test]

    torch.manual_seed(args.seed)
    print(f"[grpo] arm={args.arm} mode={args.mode} dtype={args.dtype} fp32_head={args.fp32_head} "
          f"bi={args.batch_invariant} model={args.model}", flush=True)
    policy = Policy(args.model, dtype=args.dtype, lr=args.lr, fp32_head=args.fp32_head in ("trainer", "both"))
    print(f"[grpo] trainer up: {policy.n_params/1e9:.2f}B params, {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)
    engine = RolloutEngine(args.model, dtype={"bf16": "bfloat16", "fp16": "float16"}[args.dtype], gmu=args.gmu,
                           max_model_len=args.max_model_len, seed=args.seed,
                           prefix_caching=not args.no_prefix_caching, fp32_head=args.fp32_head == "both")
    print(f"[grpo] engine up in {engine.startup_s:.0f}s; total {torch.cuda.memory_allocated()/2**30:.1f} GiB allocated", flush=True)
    monitor = MismatchMonitor(clip=args.clip)

    start_step, cursor = 0, 0
    if (ckpt_dir / "trainer.pt").exists():
        extra = policy.load(ckpt_dir)
        start_step, cursor = extra["step"], extra["cursor"]
        rng.setstate(tuple(tuple(x) if isinstance(x, list) else x for x in extra["rng"]))
        monitor.history = extra.get("monitor_history", [])
        print(f"[grpo] resumed from step {start_step}", flush=True)
        # engine must match the resumed trainer weights
        engine.sync_weights(policy.named_weights())
    elif start_step == 0:
        # step 0 sanity: engine and trainer hold identical weights already
        pass

    def evaluate(step):
        t0 = time.time()
        outs = engine.greedy(test_ids, args.max_tokens)
        acc = np.mean([o["finish"] != "length" and verify_answer(tok.decode(o["ids"], skip_special_tokens=True), p["answer"])
                       for o, p in zip(outs, test)])
        trunc = np.mean([o["finish"] == "length" for o in outs])
        rec = {"step": step, "eval_acc": float(acc), "eval_trunc": float(trunc),
               "eval_len": float(np.mean([len(o["ids"]) for o in outs])), "eval_s": time.time() - t0}
        log_line(run_dir / "eval.jsonl", rec)
        print(f"[eval] step {step} acc={acc:.3f} trunc={trunc:.2f} len={rec['eval_len']:.0f} ({rec['eval_s']:.0f}s)", flush=True)

    if start_step == 0 and not (run_dir / "eval.jsonl").exists():
        evaluate(0)

    B, G = args.prompts_per_step, args.group
    for step in range(start_step, args.steps):
        t0 = time.time()
        batch = [train[order[(cursor + i) % len(order)]] for i in range(B)]
        cursor += B
        prompt_ids = [tok(build_prompt(tok, p["problem"], thinking=not args.no_thinking), add_special_tokens=False)["input_ids"] for p in batch]
        groups = engine.generate(prompt_ids, n=G, max_tokens=args.max_tokens, temperature=args.temperature,
                                 seed=args.seed * 100003 + step)
        t_gen = time.time() - t0

        # rewards + advantages
        seqs, rewards = [], []
        for p, pids, g in zip(batch, prompt_ids, groups):
            for c in g:
                text = tok.decode(c["ids"], skip_special_tokens=True)
                ok = c["finish"] != "length" and verify_answer(text, p["answer"])
                rewards.append(1.0 if ok else 0.0)
                seqs.append({"prompt": pids, "comp": c["ids"], "lp": c["logprobs"], "trunc": c["finish"] == "length"})
        rewards_t = torch.tensor(rewards)
        adv = L.group_advantages(rewards_t, G)
        n_tok_total = sum(len(s["comp"]) for s in seqs)
        lens = [len(s["comp"]) for s in seqs]

        # trainer pass in micro-batches, sorted by length
        t1 = time.time()
        order_mb = sorted(range(len(seqs)), key=lambda i: len(seqs[i]["prompt"]) + len(seqs[i]["comp"]))
        loss_sum, w_sum, ent_sum = 0.0, 0.0, 0.0
        all_tr, all_inf, all_mask = [], [], []
        Tmax_all = max(lens)
        for s in range(0, len(order_mb), args.micro_batch):
            idx = order_mb[s : s + args.micro_batch]
            full = [seqs[i]["prompt"] + seqs[i]["comp"] for i in idx]
            Lmax = max(map(len, full))
            ids = torch.zeros((len(idx), Lmax), dtype=torch.long)
            am = torch.zeros((len(idx), Lmax), dtype=torch.long)
            for r, x in enumerate(full):
                ids[r, : len(x)] = torch.tensor(x); am[r, : len(x)] = 1
            ids, am = ids.cuda(), am.cuda()
            plen = [len(seqs[i]["prompt"]) for i in idx]; clen = [len(seqs[i]["comp"]) for i in idx]
            lp_tr, ent, mask = policy.logprobs(ids, am, plen, clen)
            Tm = lp_tr.shape[1]
            lp_inf = torch.zeros_like(lp_tr)
            for r, i in enumerate(idx):
                lp_inf[r, : clen[r]] = torch.tensor(seqs[i]["lp"], device=lp_inf.device)
            loss, st = L.grpo_loss(args.mode, lp_tr, lp_inf, adv[idx].cuda(), mask, n_tok_total, clip=args.clip)
            policy.backward(loss)
            loss_sum += float(loss.detach()); w_sum += st["w_sum"]; ent_sum += float((ent * mask).sum())
            pad = lambda t: torch.nn.functional.pad(t.detach(), (0, Tmax_all - Tm))
            all_tr.append(pad(lp_tr)); all_inf.append(pad(lp_inf)); all_mask.append(pad(mask))
        opt = policy.step()
        t_train = time.time() - t1

        # mismatch on the whole step's tokens, BEFORE the weights moved (lp_tr was computed at theta_old == theta_sampler)
        mstats = monitor.update(torch.cat(all_tr), torch.cat(all_inf), torch.cat(all_mask), step=step)

        t2 = time.time()
        sync = engine.sync_weights(policy.named_weights())
        t_sync = time.time() - t2
        sync_ok = None
        if step % args.ckpt_every == 0 or step < 2:
            fe, ft = engine.fingerprint(), policy.fingerprint(engine.FINGERPRINT_NAMES)
            sync_ok = all(abs(fe[n] - ft[n]) <= 1e-3 * max(1.0, abs(ft[n])) for n in fe)
            if not sync_ok:
                print(f"[grpo] WEIGHT SYNC MISMATCH engine={fe} trainer={ft}", flush=True)

        rec = {"step": step, "reward": float(rewards_t.mean()), "acc": float(rewards_t.mean()),
               "trunc": float(np.mean([s["trunc"] for s in seqs])), "mean_len": float(np.mean(lens)),
               "max_len": int(max(lens)), "n_tokens": n_tok_total,
               "groups_all_same": float(np.mean([len(set(rewards[i*G:(i+1)*G])) == 1 for i in range(B)])),
               "loss": loss_sum, "grad_norm": opt["grad_norm"], "skipped": opt["skipped"], "loss_scale": opt["loss_scale"],
               "w_mean": w_sum / max(1, n_tok_total), "entropy": ent_sum / max(1, n_tok_total),
               "mean_logp_infer": mstats.get("mean_logp_infer"),
               "mismatch": {k: v for k, v in mstats.items() if k not in ("hist", "by_train_prob", "step")},
               "by_train_prob": mstats.get("by_train_prob"), "hist": mstats.get("hist"),
               "t_gen": t_gen, "t_train": t_train, "t_sync": t_sync, "sync_ok": sync_ok, "n_loaded": sync["n_loaded"], "t_step": time.time() - t0,
               "gpu_mem_gib": torch.cuda.max_memory_allocated() / 2**30, "cursor": cursor}
        log_line(steps_path, rec)
        print(f"[grpo] step {step:4d} acc={rec['acc']:.3f} len={rec['mean_len']:.0f} trunc={rec['trunc']:.2f} "
              f"loss={rec['loss']:+.4f} gn={rec['grad_norm']:.3f} H={rec['entropy']:.3f} "
              f"kl3={mstats['kl_k3']:.2e} |gap|={mstats['mean_abs_gap']:.3e} band={mstats['frac_outside_band']:.4f} "
              f"ess={mstats['seq_is_ess_frac']:.2f} w={rec['w_mean']:.3f} "
              f"t={rec['t_step']:.0f}s (gen {t_gen:.0f} train {t_train:.0f} sync {t_sync:.2f}{'' if sync_ok is None else (' ok' if sync_ok else ' MISMATCH')})", flush=True)

        done = step + 1
        if done % args.eval_every == 0 or done == args.steps:
            evaluate(done)
        out_of_time = args.stop_after_sec and (time.time() - t_start) > args.stop_after_sec
        if done % args.ckpt_every == 0 or done == args.steps or out_of_time:
            t3 = time.time()
            policy.save(ckpt_dir, {"step": done, "cursor": cursor, "rng": rng.getstate(),
                                   "monitor_history": monitor.history[-500:]})
            print(f"[grpo] checkpoint at step {done} ({time.time()-t3:.0f}s)", flush=True)
        if out_of_time and done < args.steps:
            print(f"[grpo] stop-after-sec reached at step {done}; exit 75 for requeue", flush=True)
            sys.exit(75)
    (run_dir / "DONE").write_text(json.dumps({"steps": args.steps, "monitor_overhead_s": monitor.overhead_s}))
    print(f"[grpo] done: {args.steps} steps, monitor overhead {monitor.overhead_s:.1f}s", flush=True)


if __name__ == "__main__":
    main()
