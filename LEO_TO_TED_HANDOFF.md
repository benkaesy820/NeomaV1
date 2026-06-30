# Leo to Ted - Work Packet 17 Handoff

Ted is temporarily out of commission, so Leo continued locally from pushed baseline `22ad99c`.

Work Packet 16 had already passed: 248,250 exact 8K-token IDs, 500 training steps, verified resume, zero Stage B records, zero evaluation leakage, peak RSS 439,386,112 bytes, train loss 8.9803 -> 6.0542, and validation loss 9.0019 -> 6.0914.

Leo added and ran Work Packet 17, a separate 2,000-step extended diagnostic on the same approved 250K corpus. It wrote to `runs/stage_a_250k_extended_8k`, kept the provisional 8K tokenizer, and saved checkpoints plus generation samples at steps 500, 1000, 1500, and 2000.

The extended run passed mechanical gates: train loss 8.9803 -> 4.8433, validation loss 9.0019 -> 5.1575, best validation loss 5.1336 at step 1900, throughput about 2,886 tokens/second, peak RSS 439,472,128 bytes, zero Stage B records, and zero evaluation leakage.

The important caveat remains: generation is still repetitive and sometimes malformed. This validates longer Stage A training as useful, but it does not prove reliable English understanding or coding ability. The next rational packet is a carefully admitted 500K Stage A corpus with stronger English/developer-language mixture and the same milestone generation checks, not instruction tuning yet.
