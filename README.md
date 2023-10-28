# TSP-Transformer
## Abstract
Holistic scene understanding includes semantic segmen-
tation, surface normal estimation, object boundary detec-
tion, depth estimation, etc. The key aspect of this problem
is to learn representation effectively, as each subtask builds
upon not only correlated but also distinct attributes. In-
spired by visual-prompt tuning, we propose a Task-Specific
Prompts Transformer, dubbed TSP-Transformer, for holis-
tic scene understanding. It features a vanilla transformer
in the early stage and tasks-specific prompts transformer
encoder in the lateral stage, where tasks-specific prompts
are augmented. By doing so, the transformer layer learns
the generic information from the shared parts and is en-
dowed with task-specific capacity. First, the tasks-specific
prompts serve as induced priors for each task effectively.
Moreover, the task-specific prompts can be seen as switches
to favor task-specific representation learning for different
tasks. Extensive experiments on NYUD-v2 and PASCAL-
Context show that our method achieves state-of-the-art per-
formance, validating the effectiveness of our method for
holistic scene understanding.
