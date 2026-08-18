When Smart Routing Falls Apart

A negative result for engram-based adapter routing, and why the failure is more useful than the fix would have been.


I built a routing system that scored 100% on its benchmark. Then I tried to break it. It broke in a specific, instructive way that changes how I think about the whole architecture. Here is what happened.


The setup

The system in question is what I have been calling engram-as-address routing. The idea is simple. You have a base language model and a library of small adapters, one per absorbed passage of content. When a query arrives, you take a single vector — the mean of the input embeddings, what I call the L0 engram — and you compare it via cosine similarity against stored keys for each adapter. The closest match wins. Load that adapter, generate the answer.

A wrinkle: the keys are stored at a different layer than where queries are extracted. Queries come from L0, the input embedding layer. Keys come from L5, the model's last layer before the output projection. To bridge them, I trained a 1024-by-1024 linear projection W using a contrastive loss on 80 training prompts. Routing then becomes: project the L0 query through W, compare to the L5 keys, pick the top match.

On the benchmark, this hits 100% routing accuracy and 97% retrieval. The benchmark has 20 adapters, each absorbing a distinct passage. 60 held-out paraphrase queries.

That is the published result. This article is about what happens when you ask whether the published result generalizes.


First question: is the trained projection actually necessary?

I started by asking what the trained W is doing. If the L0-to-L5 mapping is something the base model already computes implicitly, maybe contrastive training is just surfacing pre-existing structure. If so, you might not need W at all. You might be able to swap in something simpler.

I ran the same routing test with seven alternative choices for the projection matrix.

The trained W gets 100% routing, 97% retrieval. That is the baseline.

If you use the identity matrix and route directly between L0 queries and L5 keys, you get 25% — barely above chance. The two layers live in different subspaces; raw cosine across them carries almost no signal.

If you use any of the obvious base-model-derived projections — composition of the model's output projection matrices, the key-projection matrix at the input layer, the key-projection matrix at the output layer — you get 5 to 8 percent. Indistinguishable from a random orthogonal control. So whatever maps L0 to L5 is not sitting in any easy-to-find attention or output subspace.

If you fit a least-squares linear regression from L0 to L5 using the 80 training prompts, with no contrastive supervision and no labels, you get 75% routing. So a partial bridge is in the base model — recoverable from data alone — but it is noisier than the trained W.

And then the surprise. If you skip the projection entirely and just store keys at L0, comparing L0 queries to L0 keys directly, you get 93% routing and 90% retrieval. Almost as good as the trained version. Without any projection at all.

That last result raises an architectural question. The trained W with L5 keys gets you 7 routing points over the simpler L0-keys-no-W setup. Is that 7-point lift worth training a 1024-by-1024 matrix? Or is the architectural complexity unjustified?


Second question: what does the trained projection buy you?

The 7-point routing lift looks small. So I tested whether the trained version's value lives in a different metric: specificity. A deployed routing system has to do two things — pick the right adapter when the query is in-library, and reject queries that fall outside the library. Routing accuracy measures the first. Specificity measures the second.

I built an out-of-distribution test set: 30 random WikiText passages not in the library, length-matched to library passages. For each one, I measured the maximum cosine to any library adapter under both routing methods. Then I calibrated each method's threshold to keep 95% recall on in-library queries, and counted what fraction of OOD queries spuriously exceeded the threshold.

I predicted, with high confidence, that the trained W would have clean specificity and the L0-keys baseline would have poor specificity. My reasoning: if in-library cosines under L0 routing are already near 1.0, OOD cosines on the same metric have nowhere to go but similarly high. So OOD would slip past the threshold and trigger spurious adapter loads.

I was wrong. Opposite direction.

Under the L0-keys baseline, in-library queries averaged cosine 0.93 against their correct adapter. OOD queries averaged 0.51. A separation of 0.42, with the two distributions linearly separable. False-positive rate at 95% in-library recall: zero. The L0-keys baseline naturally rejects unrelated English text.

Under the trained-W version, in-library queries averaged 0.25 and OOD averaged 0.12. A separation of only 0.12, with heavy distributional overlap. False-positive rate at 95% recall: 83%. To match the L0-keys baseline's clean rejection, the trained version would have to drop its in-library recall below 70%.

So the trained projection does not buy specificity. It actually has worse specificity than the simpler architecture. The 7-point routing improvement comes at a much larger specificity cost.

The mechanism is what makes this interesting. The contrastive loss I used during training optimized which library adapter is closest among the library, but it never saw out-of-distribution examples as negatives. So the trained W learned to rank library adapters relative to each other, but learned nothing about pushing OOD queries away. Meanwhile the L0 cosine has a natural property I had not noticed: it separates semantic paraphrase from unrelated English text without any supervision at all. Paraphrases of the same content share enough structure to cluster around 0.93. Random English text shares only generic English-frame structure and lands around 0.51.

So instead of one architecture replacing the other, you get a clean decomposition: use L0 cosine for the rejection question, use trained W for the selection question. Each method does what it is good at. Two-stage architecture.

I was about to commit this story to the paper. Then I ran one more experiment.


Third question: what about hard out-of-distribution?

The OOD passages I had been testing were random WikiText. Semantically very different from library content. That is the easy case. The deployment-realistic case is harder: queries that share library structure but reference different specific entities. If the library holds "Dr. Smith prefers 72 degrees," the hard rejection is "Dr. Jones prefers a quiet environment." Same template, same domain, different entity.

I built a hard-OOD test set by taking each library passage's template and substituting a different entity from the same category. The resulting passages are structurally indistinguishable from library entries except in the load-bearing content.

Both architectures failed.

Under the L0-keys baseline, in-library queries averaged 0.93 as before. Hard-OOD queries averaged 0.89. The separation collapsed from 0.42 down to 0.04. False-positive rate at 95% recall: 92%.

Under the trained-W version, in-library queries averaged 0.25 and hard-OOD averaged 0.15. Separation 0.10. False-positive rate at 95% recall: 92%.

Both stages of the two-stage architecture I had just designed broke at the same condition. And they broke for the same reason. The trained W is a linear function of L0; if L0 cannot distinguish the queries, no linear projection of L0 can either.

I tried to rescue this by retraining the projection with hinge-loss negatives drawn from random WikiText. The training converged — the model successfully pushed random negatives away — but the rescue did not transfer. The hard-OOD failures live in a different region of L0 space than where the random-WikiText negatives sit. Best result with rescue: 73% false-positive on hard-OOD. Still terrible.

I tried last-N-token pooling, in case the failure was mechanistic — maybe entity tokens get washed out by mean-pooling over 128 tokens but would survive in a smaller window near the end. Pooling over the last single token degenerated because question-final tokens are punctuation. Pooling over the last 5 tokens cut hard-OOD false-positive from 95% to 62%. Better than nothing, but not deployment-grade. Pooling over the last 10 tokens reverted to the mean-pool failure.

Three independent rescues failed. The conclusion: this is not a mechanistic problem you fix with better pooling or more training data. The entity identity is not recoverably encoded in L0 by any positional cut. The architecture inherits this representational limit and cannot work around it.


The mechanism

All the results fit one explanation. Mean-pooling L0 over a passage produces a vector where syntactic frame and common nouns dominate, and entity-specific tokens contribute weakly. The frame is most of the signal because the frame is most of the tokens.

This produces three regimes. Paraphrases of the same passage share frame and common nouns, giving high cosine around 0.93. Same-template-different-entity OOD shares frame and common nouns just as strongly, giving cosine around 0.89. Unrelated English text shares only generic English structure, giving cosine around 0.51.

The trained projection W, being linear in L0, can redistribute and rescale these signals but cannot extract entity information that mean-pooling has averaged out. The contrastive loss creates intra-library discrimination — pulling library directions apart — but cannot create rejection capability against any distribution it never saw, and cannot create entity-grain discrimination when its input lacks entity-grain signal.

The L0-keys baseline's clean rejection of unrelated English is a side effect of how L0 represents text, not an architectural feature. When OOD shares the frame, that side effect disappears.


What this means

The original published result remains valid under the conditions on which it was obtained: a library where each adapter has a unique entity instantiation, so that template-grain routing is operationally equivalent to entity-grain routing. Twenty different templates, twenty different entities, each one cleanly separable.

It does not generalize to libraries that grow to contain multiple entries with overlapping templates. A robot trying to remember preferences for multiple users, a medical assistant trying to remember protocols for multiple patients, an agent trying to maintain context across multiple instances of a recurring entity type — these are exactly the conditions where the architecture fails.

The architecturally clean response is not better routing. It is to separate routing from verification. Let routing operate at template grain, retrieving the adapter most likely to contain a relevant template. Then add a second stage that operates at entity grain, examining whether the loaded adapter actually contains the specific entity in the query — and either returning the answer or rejecting.

This is the next experiment. I have not built it yet. The design space is wide enough that it deserves its own treatment — possible mechanisms include classifier heads on the loaded adapter, generation-and-match against the adapter's training content, learned reject heads, attention-based entity matching. I list this as the principal next direction.


Why the negative result is informative

Three things make this finding more useful than the architectural rescue would have been.

It identifies a clean boundary. The architecture works on one side of the template-overlap line and fails on the other. That boundary is precise and testable.

It is mechanistically explained. Not an empirical observation in search of a story — a single account that predicts every result in the sequence.

It points the next experiment precisely. The fix is verification, not better routing. I know what to build next, and I know why.

If you are working on adapter routing, retrieval-augmented generation, or any architecture that uses learned embeddings as content-addressable pointers, this failure mode is worth checking for in your own system. Test against same-template-different-entity OOD. If your routing collapses, the fix probably is not in the routing layer.
