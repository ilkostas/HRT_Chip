What you need to do right now, in order

Step 1: Get real numbers (do this today)
Run the search_hybrid solver on all 17 benchmarks with the official evaluator, not the stub. This is the most important thing you can do right now because everything else depends on knowing where you actually stand. Use a reasonable time budget — maybe 5 minutes per benchmark for a first pass. Record the proxy cost for every benchmark. Compare to RePlAce's numbers from the README table.
This tells you one of three things. Either you're within 15% of RePlAce and you're in great shape, just needs tuning. Or you're 15–30% behind and the search is working but needs better operators or schedule tuning. Or you're more than 30% behind and something fundamental isn't working right — maybe the net data isn't loading correctly for net-aware moves, maybe the temperature is wrong, maybe the objective isn't matching the official proxy well.

Step 2: Run surrogate alignment check
You already built the surrogate correlation gating. Use it. Take 50+ placements from a benchmark run (different stages of the SA search), score them with both your fast surrogate and the official evaluator, check the Spearman correlation. If it's below 0.8, your density/congestion grid approximation is misleading the search and you should fall back to HPWL-only until you fix it.

Step 3: Tune based on what the numbers say
This is where it gets benchmark-specific. Look at the proxy cost component breakdown per benchmark (wirelength vs density vs congestion). If you're losing mostly on wirelength, your net-aware operator needs work or more weight. If you're losing on density, your grid resolution or density penalty scaling might be off. If congestion, same idea.

Step 4: Per-benchmark parameter overrides
Some benchmarks have 246 macros, some have 537. The SA schedule and operator mix that works for small benchmarks probably isn't optimal for large ones. Try different configs on your 3 worst benchmarks and save overrides.

Step 5: Time budget optimization
You have a 1-hour limit per benchmark in the competition. Right now you're probably running much shorter for testing. Try a full 30–60 minute run on a couple of benchmarks to see how much improvement the SA finds with more time. If the cost curve flattens after 10 minutes, you know more time won't help and you should invest in better moves instead. If it's still improving at 30 minutes, longer runs are valuable.

Step 6: Final submission prep
Run your best configs 3 times each with different seeds. Check stability. Verify zero overlaps on all benchmarks. Run replay verification. Submit.

The honest timeline from here
If your numbers from Step 1 are reasonable (within 20–30% of RePlAce), you could have a competitive submission in 2–3 more intense sessions of tuning and experimentation. The code is built. What remains is measurement and optimization.
If the numbers are bad, you need to debug first — and that's unpredictable. Could be a 20-minute fix, could be a day.
