---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:999
- loss:CosineSimilarityLoss
base_model: sentence-transformers/paraphrase-MiniLM-L6-v2
widget:
- source_sentence: '"Progressive" Dems need to stop boarding the Nancy Pelosi hate
    train just to seem cool. People blame her for things she literally can''t control
    and give credit from her successes to the more popular congresspeople they already
    like.'
  sentences:
  - If you have a load of Euros and you pay a tuition fee in the UK, you will convert
    you're Euros and get more pounds for them, so yes you're debt is easier to pay
    off.
  - You know why we hate her? Because she takes money from the elite, she has corporate
    backing. Now Before this bullshit she didn't even dare to negotiate with Mitch.
    All of a sudden she supports $2K checks. Why didn't she tell him this before the
    passing. The government has you fooled.
  - It's really funny how we still mostly heat water to make a propeller move, simple
    but also outdated
- source_sentence: 'Newsmax clearly aligning themselves with the right wing. Non-American
    asking here: how do they compare to FOX?'
  sentences:
  - They are financed by billionaires who think FOX News doesn't go far enough and
    that the job of their journalists is to criticize anything left of center to give
    the simple impression to their impressionable viewers of "the right is good, the
    left is bad". Founded by Christopher Ruddy and supported by a former CIA director,
    their goal is to serve as a disinformation and conspiracy theory outlet to the
    red hats can use them as a "source".
  - By and large, the media won't report it that way, and even if they did, too many
    people wouldn't get deep enough past the headlines to understand it.
  - No you can't. That's people showing open support for their candidate and-or President,
    which they have a right to do. Or used to anyway. Believe it or not that's the
    way things worked in this country once upon a time.
- source_sentence: Maybe they should arrange a meeting in Australia, sometime next
    year.
  sentences:
  - The scientific climate community are engaging in an enormous conspiracy for some
    reason?
  - He he he ... that would probably be the smartest the UK can do!
  - I call BS on the 4 million petition before the referendum. Links or it didn't
    happen.
- source_sentence: Why would any one pay for NI? The state it's in it's more like
    one of those deals where you pay 1PS and then the seller has to commit to continue
    paying for certain things for decades to come on top of that and ensure that all
    liabilities remain with the seller. Like was the case with the Rover MBO. Not
    that that went well...
  sentences:
  - '>The state it''s in Begs the question why the UK is so keen on keeping it in
    the Union. Either it has value or it hasn''t.'
  - Thanks for pointing this out! I've pasted ONLY those states with raising curve
    (not flat, not falling). The Virus doesn't recognize political party, but political
    parties (as you pointed out) define infection and death rates. Your point accidentally
    proves why democrats are leading the anti-covid effort. Republicans sending people
    into the streets to protest for open hair saloons, nail saloons and parties, rushed
    openings, ignoring the virus or low testing - the best way for disaster.
  - 'There is nothing more comfortable for a conservative than an Uncle Tom reassuring
    the right that, "No sir, YOU''RE not racist. In fact, racism doesn''t exist in
    America." Because it confirms your childlike simple view that "America is #1 and
    the greatest at everything"'
- source_sentence: They were told to arrest her bf... they shot blindly into the house,
    hitting her 8 times. Listen to his call to 911, he had no idea who it was. All
    they had to do was identify themselves.
  sentences:
  - yea, with confederate statues i would be ok with them being taken down as long
    as people from the town voted on it. But Some of these statues are like roosevelt
    lmao. An environmentalist, and the first president to have a black man in the
    White house for dinner. God damn just vote on it XD they don't have to tear down
    the thing with a mob
  - Ironically, Hansen's objection lies in the opposite direction-his perception of
    the political impossibility of the GND's ambitious reduction schedule. He would
    think degrowth no less of a utopian project.
  - They were told to do a no knock raid. I DO AGREE that the situation was messed
    up. But the error happened on a level ABOVE the cops.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/paraphrase-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2) <!-- at revision 9a27583f9c2cc7c03a95c08c5f087318109e2613 -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'They were told to arrest her bf... they shot blindly into the house, hitting her 8 times. Listen to his call to 911, he had no idea who it was. All they had to do was identify themselves.',
    'They were told to do a no knock raid. I DO AGREE that the situation was messed up. But the error happened on a level ABOVE the cops.',
    "Ironically, Hansen's objection lies in the opposite direction-his perception of the political impossibility of the GND's ambitious reduction schedule. He would think degrowth no less of a utopian project.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 999 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 999 samples:
  |         | sentence_0                                                                         | sentence_1                                                                          | label                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                             | string                                                                              | float                                                         |
  | details | <ul><li>min: 5 tokens</li><li>mean: 48.92 tokens</li><li>max: 128 tokens</li></ul> | <ul><li>min: 13 tokens</li><li>mean: 43.07 tokens</li><li>max: 128 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.5</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                     | sentence_1                                                                                                                                                                                                                                                                                                | label            |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>You could say the same thing about red MAGA hats. Just sayin'.</code>                                                                                                                                                                                                                                    | <code>No you can't. That's people showing open support for their candidate and-or President, which they have a right to do. Or used to anyway. Believe it or not that's the way things worked in this country once upon a time.</code>                                                                    | <code>0.0</code> |
  | <code>What affects climate? In terms of what people do, it's greenhouse gas emissions. Mostly CO2 from burning fossil fuels, some methane from natural gas leakage and ruminants, and a bit more CO2 from deforestation. Refrigerant leakage is also an issue, as are a few other industrial chemicals.</code> | <code>I was just trying to gauge what a person can do at an individual level to counteract some issues that are receiving far less thought and effort. What about refrigerant leakage is bad for the environment? This is the first I'm hearing of this</code>                                            | <code>0.0</code> |
  | <code>It's really shameful how deep people have been indoctrinated with these lies about the EU being an oppressive colonist regime. It takes a lot of work to get people to be ready to throw their livelihoods in the trash.</code>                                                                          | <code>>It takes a lot of work to get people to be ready to throw their livelihoods in the trash. You've actually got it backwards. It's easy, it's one of the easiest things in the world to make someone shut off their brain and act without thinking. You need tell them they are under attack.</code> | <code>0.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Framework Versions
- Python: 3.12.6
- Sentence Transformers: 3.3.1
- Transformers: 4.46.3
- PyTorch: 2.5.1+cpu
- Accelerate: 1.1.1
- Datasets: 3.1.0
- Tokenizers: 0.20.3

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->