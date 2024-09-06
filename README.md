# Running the Experiment

This section describes the parameters available for the experiment and provides instructions on how to run it.

## Parameters

The script offers two main choices: running without an attack or with an attack. For attacks, you can choose between deletion, synonym substitution, and paraphrasing.

The script accepts the following command-line arguments:

- `--target_model`: The model to evaluate or attack 
- `--ref_model`: The reference model 
- `--output_dir`: Directory to save output files 
- `--dataset`: Name of the dataset to use
- `--sub_dataset`: Subset size of the dataset (32, 64 and 128)
- `--num_shots`: Number of shots
- `--pass_window`: Whether to exceed the context window
- `--attack_type`: Type of attack to apply. Options are:
  - `none`: No attack 
  - `del`: Deletion attack
  - `sub`: Synonym substitution attack
  - `para`: Paraphrase attack
- `--attack_strength`: Portion of words to modify for deletion and substitution attacks 

## Running the Experiment

To run the experiment, use the following command structure:

```bash
python run.py [arguments]
```

Replace `[arguments]` with the desired parameter settings. Here are some examples:

1. Run without attack:
   ```
   python run.py --target_model EleutherAI/pythia-6.9b --ref_model EleutherAI/pythia-70m --dataset wikimia --sub_dataset 32 --num_shots 7
   ```

2. Run with a deletion attack:
   ```
   python run.py --target_model EleutherAI/pythia-6.9b --ref_model EleutherAI/pythia-70m --dataset wikimia --sub_dataset 32 --attack_type del --attack_strength 0.15
   ```

3. Run with a synonym substitution attack:
   ```
   python run.py --target_model huggyllama/llama-30b --ref_model huggyllama/llama-7b --dataset wikimia --sub_dataset 64 --attack_type sub --attack_strength 0.20
   ```

4. Run with a paraphrase attack:
   ```
   python run.py --target_model EleutherAI/pythia-6.9b --ref_model EleutherAI/pythia-70m --dataset wikimia --sub_dataset 128 --attack_type para
   ```

Note that for deletion and synonym substitution attacks, you can specify the portion of words to be modified using the `--attack_strength` parameter. This parameter is not applicable for paraphrase attacks.

Here are the reference models we use for each family:

| Model Family | Reference Model |
|--------------|-----------------|
| Mamba        | state-spaces/mamba-130m-hf |
| Pythia       | EleutherAI/pythia-70m |
| GPT-NeoX     | EleutherAI/gpt-neo-125m |
| LLaMA        | huggyllama/llama-7b |


# Acknowledgement
The code is adapted from the following repositories:
- [Min-K%](https://github.com/swj0419/detect-pretrain-code)
- [ReCall](https://github.com/ruoyuxie/recall)

Datasets used in the experiment are from the following huggingface datasets:
- [wikiMIA](https://huggingface.co/datasets/swj0419/WikiMIA)
- [MIMIR](https://huggingface.co/datasets/iamgroot42/mimir)
- [WikiMIA_paraphrased_perturbed](https://huggingface.co/datasets/zjysteven/WikiMIA_paraphrased_perturbed)

# Citation
If you find this work useful, please consider citing the following paper:
```bibtex
@article{wang2024conrecall,
      title={Con-ReCall: Detecting Pre-training Data in LLMs via Contrastive Decoding}, 
      author={Cheng Wang and Yiwei Wang and Bryan Hooi and Yujun Cai and Nanyun Peng and Kai-Wei Chang},
      journal={arXiv preprint arXiv:2409.03363},
      year={2024}
}
```
