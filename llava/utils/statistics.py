import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings

def compute_effect_statistics(effect_list: List[float],
                             alpha: float = 0.05,
                             test_value: float = 0.0) -> Dict:
    """
Calculate statistical summaries for effect values.

Parameters
----------
effect_list : List[float]
A list of effect values ​​(e.g., ATE or AIE values).
alpha : float
The significance level (alpha), default is 0.05.
test_value : float
The null value for the hypothesis test; default is 0.0.

Returns
-------
Dict
A dictionary containing the following statistics:
- mean: Mean
- std: Standard deviation
- sem: Standard error of the mean
- n: Sample size
- ci_lower, ci_upper: Confidence interval bounds
- t_stat: t-statistic
- p_value: p-value (from a one-sample t-test)
- cohen_d: Cohen's d effect size
- significant: Whether the result is statistically significant (p < alpha)
"""
    if len(effect_list) == 0:
        raise ValueError("The effects list cannot be empty.")

    effect_array = np.array(effect_list)
    n = len(effect_array)
    mean = np.mean(effect_array)
    std = np.std(effect_array, ddof=1)  
    sem = std / np.sqrt(n)  

    ci = stats.t.interval(1 - alpha, n - 1, loc=mean, scale=sem)
    ci_lower, ci_upper = ci

    if std == 0:
        if mean == test_value:
            t_stat, p_value = 0.0, 1.0
        else:
            t_stat, p_value = np.inf, 0.0
    else:
        t_stat, p_value = stats.ttest_1samp(effect_array, test_value)

    if std == 0:
        cohen_d = np.inf if mean != test_value else 0.0
    else:
        cohen_d = (mean - test_value) / std

    significant = p_value < alpha

    return {
        'mean': mean,
        'std': std,
        'sem': sem,
        'n': n,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        't_stat': t_stat,
        'p_value': p_value,
        'cohen_d': cohen_d,
        'significant': significant,
        'alpha': alpha
    }

def compare_effects(effect_list1: List[float],
                   effect_list2: List[float],
                   alpha: float = 0.05,
                   paired: bool = True) -> Dict:
    if len(effect_list1) == 0 or len(effect_list2) == 0:
        raise ValueError("The effects list cannot be empty.")

    arr1 = np.array(effect_list1)
    arr2 = np.array(effect_list2)

    if paired:
        # 配对t检验
        if len(arr1) != len(arr2):
            raise ValueError("A paired test requires the two sets of data to be of equal length.")

        diff = arr1 - arr2
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        n = len(diff)

        if std_diff == 0:
            if mean_diff == 0:
                t_stat, p_value = 0.0, 1.0
            else:
                t_stat, p_value = np.inf, 0.0
        else:
            t_stat, p_value = stats.ttest_rel(arr1, arr2)

        sem_diff = std_diff / np.sqrt(n)
        ci = stats.t.interval(1 - alpha, n - 1, loc=mean_diff, scale=sem_diff)
        ci_lower, ci_upper = ci

        if std_diff == 0:
            cohen_d = np.inf if mean_diff != 0 else 0.0
        else:
            cohen_d = mean_diff / std_diff
    else:
        mean_diff = np.mean(arr1) - np.mean(arr2)
        n1, n2 = len(arr1), len(arr2)

        t_stat, p_value = stats.ttest_ind(arr1, arr2, equal_var=False)

        var1 = np.var(arr1, ddof=1)
        var2 = np.var(arr2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        se_diff = np.sqrt(var1/n1 + var2/n2)
        df = se_diff**4 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))

        t_critical = stats.t.ppf(1 - alpha/2, df)
        margin = t_critical * se_diff
        ci_lower = mean_diff - margin
        ci_upper = mean_diff + margin

        if pooled_std == 0:
            cohen_d = np.inf if mean_diff != 0 else 0.0
        else:
            cohen_d = mean_diff / pooled_std

    significant = p_value < alpha

    return {
        'mean_diff': mean_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        't_stat': t_stat,
        'p_value': p_value,
        'cohen_d': cohen_d,
        'significant': significant,
        'paired': paired,
        'alpha': alpha
    }

def compute_contribution_statistics(aie_list: List[float],
                                   ate_list: List[float],
                                   alpha: float = 0.05) -> Dict:
    if len(aie_list) != len(ate_list):
        raise ValueError("The lengths of the AIE and ATE lists must be identical.")

    aie_array = np.array(aie_list)
    ate_array = np.array(ate_list)

    # 仅计算ATE ≠ 0的样本
    valid_mask = ate_array != 0
    valid_count = np.sum(valid_mask)
    total_count = len(ate_array)

    contribution_ratios = []
    if valid_count > 0:
        contribution_ratios = ((aie_array[valid_mask] / ate_array[valid_mask]) * 100).tolist()

    result = {
        'contribution_ratios': contribution_ratios,
        'mean_ratio': np.mean(contribution_ratios) if len(contribution_ratios) > 0 else 0,
        'median_ratio': np.median(contribution_ratios) if len(contribution_ratios) > 0 else 0,
        'valid_count': valid_count,
        'total_count': total_count,
        'valid_percentage': valid_count / total_count * 100 if total_count > 0 else 0
    }

    if len(contribution_ratios) > 1:
        ratio_stats = compute_effect_statistics(contribution_ratios, alpha=alpha, test_value=0)
        result['ratio_stats'] = ratio_stats

    return result

def multiple_comparison_correction(p_values: List[float],
                                  method: str = 'fdr_bh') -> Dict:
    
    if len(p_values) == 0:
        return {'original_p': [], 'corrected_p': [], 'significant': []}

    try:
        from statsmodels.stats.multitest import multipletests
        reject, corrected_p, _, _ = multipletests(p_values, alpha=0.05, method=method)

        return {
            'original_p': p_values,
            'corrected_p': corrected_p.tolist(),
            'significant': reject.tolist(),
            'method': method
        }
    except ImportError:
        warnings.warn("statsmodels is not installed; using Bonferroni correction as an alternative.")
        n = len(p_values)
        corrected_p = [min(p * n, 1.0) for p in p_values]
        reject = [p < 0.05 for p in corrected_p]

        return {
            'original_p': p_values,
            'corrected_p': corrected_p,
            'significant': reject,
            'method': 'bonferroni'
        }

def format_statistics_summary(stats_dict: Dict, precision: int = 4) -> str:
    if not stats_dict or len(stats_dict) == 0:
        return "No statistical data"

    fmt = f"{{:.{precision}f}}"
    mean_str = fmt.format(stats_dict['mean'])
    ci_str = f"[{fmt.format(stats_dict['ci_lower'])}, {fmt.format(stats_dict['ci_upper'])}]"
    p_str = fmt.format(stats_dict['p_value'])
    cohen_str = fmt.format(stats_dict['cohen_d'])
    sig_str = "Yes" if stats_dict['significant'] else "No"

    lines = [
        f"Mean: {mean_str}",
        f"95% Confidence Interval: {ci_str}",
        f"Sample Size: {stats_dict['n']}",
        f"Standard Deviation: {fmt.format(stats_dict['std'])}",
        f"Standard Error: {fmt.format(stats_dict['sem'])}",
        f"t-statistic: {fmt.format(stats_dict['t_stat'])}",
        f"p-value: {p_str}",
        f"Cohen's d: {cohen_str}",
        f"Significant (α={stats_dict['alpha']}): {sig_str}"
]

    return "\n".join(lines)