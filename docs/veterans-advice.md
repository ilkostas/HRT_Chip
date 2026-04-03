What's next, step by step

Step 1: Run it on all 17 benchmarks and record baseline numbers (30 minutes)
You need to know your current scores. Run a full sweep, save the results. This is your "before" snapshot. Every future change gets measured against this. The number you care about: your average proxy cost vs RePlAce's 1.4578 and vs greedy row placer's 2.2109. If you're below 2.0, the engine is working. If you're above 2.0, something in the SA needs debugging before you move forward.

Step 2: Add net-aware shift operator (1–2 hours)
This is the single biggest bang-for-buck improvement. Right now your shift operator moves rectangles randomly. A net-aware shift calculates "where do this rectangle's neighbors want it to be?" — the center of mass of all rectangles connected to it by nets — and moves it toward that point. This directly reduces wire length, which is the heaviest weight in the proxy cost formula.
You need the net/connectivity data from the benchmark for this. Your official evaluator adapter already loads benchmarks, so the pin-to-net mapping should be accessible. The operator itself is simple: for the chosen macro, gather all nets it belongs to, find the average center of all other macros on those nets, shift toward that center by a random fraction.

Step 3: Add pairwise swap operator (30 minutes)
Pick two macros of similar size, swap their positions. Check legality for both new positions. This helps escape local minima where two rectangles would be better off trading places but neither can get there through individual shifts.

Step 4: Run the 17-benchmark sweep again, compare (30 minutes)
You should see measurable improvement from the new operators. If net-aware shift isn't helping, either the net data isn't being loaded correctly or the move magnitude needs tuning. Diagnose before moving on.

Step 5: Tune the SA temperature schedule (1–2 hours)
This is experimentation, not coding. Try different starting temperatures, cooling rates, and acceptance thresholds on 3–4 benchmarks. Track acceptance rate over time — you want it starting around 70–80% and ending below 5%. If it starts too low, your initial temperature is too cold and you're doing hill-climbing not annealing. If it ends too high, you're not cooling fast enough and accepting too much junk.

Step 6: Add adaptive operator selection (30 minutes)
Track which operator is producing accepted improving moves. Every N moves, update the probability of picking each operator. If net-aware shift is producing 80% of improvements, pick it 80% of the time. Simple and effective.

Step 7: Add seed screening — successive halving (1 hour)
Run all seeds for 2 minutes each. Evaluate. Kill the bottom half. Give the survivors the remaining time budget split evenly. This ensures your best starting arrangements get the most search time.

Step 8: Add density and congestion to the objective (2–3 hours)
This is the hardest coding piece remaining. For density: overlay a grid on the board, count how much macro area falls in each grid cell, penalize cells that are too full. For congestion: estimate how many net bounding boxes cross each grid cell, penalize hotspots. Weight them into the objective as 1.0 × HPWL + 0.5 × density + 0.5 × congestion to match the official formula.
The tricky part: making this incremental. When you move one macro, you only need to update the grid cells it moved out of and into, not the whole grid. Getting this right is what makes the search fast enough.

Step 9: Validate surrogate vs official proxy alignment (1 hour)
Run 50+ varied placements through both your fast objective and the official evaluator. Compute Spearman correlation. If it's above 0.9, trust your surrogate for search. If it's below 0.8, your density/congestion approximation needs work — go back and fix it before relying on it.

Step 10: Full sweep, compare to RePlAce (30 minutes)
This is your checkpoint. If you're within 15–20% of RePlAce's average, you're on track. If not, focus tuning time on whichever benchmarks have the biggest gap.

Step 11: Per-benchmark tuning (2–3 hours)
Some benchmarks respond better to different operator mixes or temperature schedules. Run experiments on your 3–4 worst benchmarks. Try different seed counts, operator weights, and cooling rates. Save the best config per benchmark as an override.

Step 12: Submission hardening (1–2 hours)
Run your final configs 3 times each with different seeds to make sure results are stable. Check that no benchmark has overlaps. Run replay verification. Package your submission.

Total time estimate at your speed
Steps 1–7 (working competitive solver): ~6–8 hours of focused work
Steps 8–12 (polished competitive submission): another 6–8 hours
So roughly two intense days to have a complete, tuned solver. Then additional days of experimentation only make it better — diminishing returns but still valuable.

The one thing that determines whether you win
Everything above gets you a strong entry. The difference between a strong entry and a winning entry is how good your net-aware operators are and how well your temperature schedule is tuned per benchmark. Those two things account for most of the gap between a basic SA and a competition-winning SA. Spend your experimentation time there, not on adding more features.
