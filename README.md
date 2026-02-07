# Execution Steps

This section provides step-by-step instructions for executing the CLUE project using Edge Pruning methodology. The paper is **“CLUE: Conflict-guided Localization for LLM Unlearning Framework”** accepted by ICLR 2026. 

## Step 1: Dataset Creation
Execute the dataset creation script using Python:
```bash
python Edge-Pruning/get_wmdp.py
```

Additional datasets can be obtained from the [Edge-Pruning repository](https://github.com/princeton-nlp/Edge-Pruning). 

**Note**: If you encounter insufficient GPU memory, you can modify the `max_token` parameter to control the token length of the generated dataset. However, this may lead to unexpected deviations in the final results.

## Step 2: Circuit Generation
Execute the pruning scripts to generate Noising and Denoising circuits:
```bash
python Edge-Pruning/src/prune/fmistral_cyber.py
python Edge-Pruning/src/prune/fmistral_cyber_or.py
```

Please execute the corresponding Python files based on your specific dataset requirements.

## Step 3: Edge Refinement
Generate refined edges.json files using the visualization script:
```bash
python Edge-Pruning/src/modeling/vis_mistral.py
```

**Note**: Noising and denoising processes will produce two different JSON files.

## Step 4: Logical Circuit Generation
Extract circuits containing conjunction and disjunction operations:
```bash
python Edge-Pruning/src/solver/get_logical_edge.py
```

## Step 5: Conflict Analysis
Identify forget nodes and conflict nodes:
```bash
python Edge-Pruning/src/solver/conflict.py
```

## Step 6: Mask Generation
Generate masks for the corresponding nodes:
```bash
python Edge-Pruning/src/solver/mask_generation.py
```

## Step 7: Model Unlearning
Perform the unlearning process:
```bash
python src/exec/unlearn_model_conlict.py
```
## BIB
```
@inproceedings{
Chen2026clue,
title={{CLUE}: Conflict-guided Localization for {LLM} Unlearning Framework},
author={Chen, Hang and Zhu, Jiaying and Yang, Xinyu and Wang, Wenya},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=jtRYvazBWv}
}
```
