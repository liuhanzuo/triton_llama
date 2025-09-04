### Online Softmax
$$
    m_i^{new}=\max(m_i, \tilde{m}_{i,j})\in\mathbb{R}^{B_r}
$$
这里$m_i$是行的最大值,$\tilde{m}_{i,j}$是当前块的最大值.
$$
    l_i^{new} = e^{m_i-m_i^{new}}l_i + e^{\tilde{m}_{i,j}-m_i^{new}}\tilde{l}_{i,j}\in\mathbb{R}^{B_r}
$$
这里$l_i$是行的归一化因子,$\tilde{l}_{i,j}$是当前块的归一化因子.