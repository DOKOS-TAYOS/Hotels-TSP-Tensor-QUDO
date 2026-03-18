# Cost Formulations

Mathematical formulations for the Hotel TSP problem in Tensor-QUDO and QUBO form.

Reference: [Introduction to QUDO, Tensor QUDO and HOBO formulations](https://arxiv.org/abs/2508.01958) (arXiv:2508.01958).

## Conventions

- $N$: number of cities (including depot)
- $x_t$: city visited at timestep $t$, $x_t \in \{0, 1, \dots, N-2\}$
- $x_{N-1} = x_{-1} = N-1$: depot (fixed start/end)
- $E^v_{t,a,b}$: travel cost at timestep $t$ from $a$ to $b$
- $E^h_{t,a}$: hotel cost at timestep $t$ in city $a$
- $H(\cdot)$: Heaviside step function
- $P$: set of precedence pairs $(i,j)$ meaning city $i$ must precede city $j$
- $\lambda_0, \lambda_1, \lambda_2$: penalty coefficients for constraints

---

## Tensor-QUDO

### Cost in terms of $E^v$, $E^h$ and penalties

$$
\begin{align*}
C(\vec{x}) = &\sum_{t=0}^{N-3} E^v_{t+1,x_t, x_{t+1}} + E^v_{0,N-1, x_{0}} + E^v_{N-1,x_{N-2}, N-1} +\\
&+\sum_{t=0}^{N-3}E^h_{t,x_t} + E^h_{N-2,x_{N-2}} + \lambda_1 \sum_{t, t', t'>t}^{N-3, N-2} H(t'-t-0.5)\delta_{x_t,x_{t'}}+\\
&+ \lambda_2 \sum_{t, t', t'>t}^{N-3, N-2} H(t'-t-0.5) \sum_{(i,j)\in P}\delta_{x_t,j}\delta_{x_{t'}, i}, \quad x_i \in \{0,1,\dots, N-2\}
\end{align*}
$$

### Compact tensors $E_{t,a,b}$ and $E_{t,t',a,b}$

$$
\begin{align*}
E_{t,a,b} =& E^v_{t+1,a,b} + \delta_{0,t}E^v_{0,N-1, a} + \delta_{N-3,t}E^v_{N-1,b, N-1} +\\
&+ E^h_{t,a} + \delta_{N-3,t}E^h_{N-2,b}\\
&t \in \{0,1,\dots, N-3\}, \quad a,b \in \{0,1,\dots, N-2\}
\end{align*}
$$

$$
\begin{align*}
E_{t,t',a,b} = H(t'-t-0.5) \left(\lambda_1 \delta_{a,b} + \lambda_2\sum_{(i,j)\in P}\delta_{a,j}\delta_{b, i}\right)\\
t \in \{0,1,\dots, N-3\},\quad t' \in \{1,\dots, N-2\}, \quad a,b \in \{0,1,\dots, N-2\}
\end{align*}
$$

### Final cost (implemented as `Etab`, `Ettprimeab`)

$$
C(\vec{x})= \sum_{t=0}^{N-3}  E_{t,x_t, x_{t+1}} + \sum_{t, t', t'>t}^{N-3, N-2} E_{t,t', x_t, x_{t'}}, \quad x_i \in \{0,1,\dots, N-2\}
$$

---

## QUBO

### Cost in terms of binary variables $x_{t,i} \in \{0,1\}$

$$
\begin{align*}
C(\vec{x}) = &\sum_{t,i,j, i\neq j}^{N-3, N-2, N-2} E^v_{t+1,i,j} x_{t,i}x_{t+1,j}+ \sum_{i=0}^{N-2} E^v_{0,N-1, i}x_{0,i}+\\
& + \sum_{j=0}^{N-2} E^v_{N-1,j,N-1} x_{N-2,j}+\sum_{t,i}^{N-2, N-2}E^h_{t,i}x_{t,i}+\\
& + \lambda_0 \sum_{t, i, j, i\neq j}^{N-2, N-2, N-2} x_{t,i}x_{t,j}+ \lambda_1 \sum_{t, t', i, t\neq t'}^{N-2, N-2, N-2} x_{t,i}x_{t',i}+\\
& + \lambda_2 \sum_{t, t', t\lt t'}^{N-3, N-2}\sum_{(i,j)\in P} x_{t',i}x_{t,j}, \quad x_i \in \{0,1\}
\end{align*}
$$

### Standard QUBO form $x^T Q x$

$$
C(\vec{x}) = \sum_{i=0}^{N-2} Q_{ii} x_i + \sum_{i \lt j}^{N-3,N-2} 2Q_{ij} x_i x_j, \quad x_i \in \{0,1\}
$$

---

## Implementation mapping

| Formulation | Code (`instance_gen_process`) | Tensors / matrix |
|-------------|------------------------------|------------------|
| Tensor-QUDO | `ProblemTQUDO`, `generate_TQUDO_from_problem` | `Etab[t,a,b]` = $E_{t,a,b}$, `Ettprimeab[t,t',a,b]` = $E_{t,t',a,b}$ |
| QUBO        | `ProblemQUBO`, `generate_QUBO_from_problem`  | `qubo_matrix` = $Q$ |
