## Recurrent and Recursive Nets
---

### LSTM : Long-Short Term Memory

![image](./photos/figure.tiff)

$$ i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i) $$

$$ \tilde{C_t} = tanh(W_c x_t + U_c h_{t-1} + b_c) $$

$$  f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f) $$

$$ C_t = i_t * \tilde{C_t} + f_t * C_{t-1} $$

$$ o_t = \sigma(W_o x_t + U_o h_{t-1} + V_o C_t + b_o) $$

$$ h_t = o_t * tanh(C_t) $$