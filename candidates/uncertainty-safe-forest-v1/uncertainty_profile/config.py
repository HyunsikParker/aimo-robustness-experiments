"""Fixed feature ordering for the data-only numeric predictor."""

FEATURE_NAMES: tuple[str, ...] = (
    'generation_mean_logprob',
    'generation_mean_prob',
    'generation_mean_nll',
    'generation_ppl',
    'generation_min_logprob',
    'generation_max_logprob',
    'generation_std_logprob',
    'generation_min_k_logprob',
    'generation_mean_entropy',
    'generation_mean_top1_prob',
    'generation_mean_top2_margin',
    'generation_frac_selected_is_top1',
    'generation_frac_high_conf_tokens',
    'generation_frac_low_conf_tokens',
)
