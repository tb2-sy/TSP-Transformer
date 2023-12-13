# TSP-Transformer
## Abstract
Holistic scene understanding includes semantic segmentation, surface normal estimation, object boundary detection, depth estimation, etc. The key aspect of this problem
is to learn representation effectively, as each subtask builds
upon not only correlated but also distinct attributes. Inspired by visual-prompt tuning, we propose a Task-Specific
Prompts Transformer, dubbed TSP-Transformer, for holistic scene understanding. It features a vanilla transformer
in the early stage and tasks-specific prompts transformer
encoder in the lateral stage, where tasks-specific prompts
are augmented. By doing so, the transformer layer learns
the generic information from the shared parts and is endowed with task-specific capacity. First, the tasks-specific
prompts serve as induced priors for each task effectively.
Moreover, the task-specific prompts can be seen as switches
to favor task-specific representation learning for different
tasks. Extensive experiments on NYUD-v2 and PASCAL-Context show that our method achieves state-of-the-art performance, validating the effectiveness of our method for
holistic scene understanding.
### [arXiv](https://arxiv.org/pdf/2311.03427.pdf) | [Evaluation Results]()

## TODO/Future work
- [x] Upload paper and init project
- [ ] Training and Inference code
- [ ] Reproducible checkpoint

## Contact
For any questions related to our paper and implementation, please email wangshuo2022@shanghaitech.edu.cn.

<!--## Citation
If you find our code or paper helps, please consider citing:-->

<!--## Acknowledgements
The code is available under the MIT license and draws from [TensoRF](https://github.com/apchenstu/TensoRF), [DynamicNeRF](https://github.com/gaochen315/DynamicNeRF), and [BARF](https://github.com/chenhsuanlin/bundle-adjusting-NeRF), which are also licensed under the MIT license.
Licenses for these projects can be found in the `licenses/` folder.

We use [RAFT](https://github.com/princeton-vl/RAFT) and [DPT](https://github.com/isl-org/DPT) for flow and monocular depth prior.-->
