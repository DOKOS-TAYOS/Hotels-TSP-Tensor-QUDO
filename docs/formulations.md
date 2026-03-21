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
& + \lambda_0 \sum_{t, i, j, i\neq j}^{N-2, N-2, N-2} x_{t,i}x_{t,j} - \lambda_0 \sum_{t, i}^{N-2,N-2} x_{t,i}+ \lambda_1 \sum_{t, t', i, t\neq t'}^{N-2, N-2, N-2} x_{t,i}x_{t',i}- \lambda_1 \sum_{t, i}^{N-2,N-2} x_{t,i}+\\
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

---

## Cirq Tensor-QUDO: Native Qudit Implementation

The Cirq TQUDO backend (`src/solvers/cirq_solver/qaoa_circuit_tqudo.py`) uses
**native d-dimensional Cirq qudits** (`cirq.LineQid(i, dimension=d)`), not
qubit emulation.  This section documents the three custom gates that form the
QAOA ansatz and the key design decisions.

> The previous qubit-emulation approach is preserved in
> `qaoa_circuit_tqudo_qubit_emulation.py` for reference.

### Qudit allocation

Each logical qudit is a single `cirq.LineQid` with `dimension = d = N - 1`
(the number of available cities).  The total Hilbert space dimension is $d^n$,
not $2^{n \lceil\log_2 d\rceil}$, which **eliminates all spurious basis states**
when $d$ is not a power of two.

### Initial state — `QuditHadamardGate`

Creates the d-dimensional uniform superposition:

$$|+_d\rangle = \frac{1}{\sqrt{d}} \sum_{k=0}^{d-1} |k\rangle$$

The unitary is the $d \times d$ DFT matrix:
$F_d[j,k] = \omega^{-jk} / \sqrt{d}$, with $\omega = e^{2\pi i / d}$
(equivalently $F_d[j,k] = e^{-2\pi i\, jk/d} / \sqrt{d}$, matching the
``np.fft.fft(…, norm="ortho")`` convention used in the code).
For $d = 2$ this is the standard Hadamard $H$ (up to global phase).

### Cost layer — `QuditDiagonalCostGate`

A diagonal two-qudit gate of size $d^2 \times d^2$ that encodes the entire
cost tensor slice for a pair of qudits $(t, t')$ in a single operation:

$$U_\text{cost} = \text{diag}\!\left(e^{-i\gamma \, E_{t,x_0,x_1}}\right)_{x_0, x_1 \in \{0,\ldots,d-1\}}$$

The basis ordering follows Cirq's convention:
$|x_0\rangle \otimes |x_1\rangle \to$ index $x_0 \cdot d + x_1$.

This replaces the $d^2$ X-flip → multi-controlled-Z → X-unflip sequences per
qudit pair in the old qubit-emulation approach, reducing the abstract gate
count per layer from $O(n^2 \cdot d^2)$ to $O(n^2)$.

### Mixer layer — `QuditRingMixerGate`

Implements the ring mixer:

$$U_\text{mix}(\beta) = \exp\!\left(-i\,\beta\, M_d\right), \quad M_d = \frac{X_d + X_d^\dagger}{2}$$

where $X_d = \sum_{k=0}^{d-1} |k{+}1 \bmod d\rangle\langle k|$ is the
d-dimensional cyclic-shift operator.  The Hermitian generator
$M_d = (X_d + X_d^\dagger)/2$ is used instead of $X_d$ alone because $X_d$ is
**not Hermitian** for $d > 2$, and only Hermitian generators produce unitary
matrix exponentials.

For $d = 2$: $M_2 = X$ (Pauli-X), and $\exp(-i\beta X) = R_x(2\beta)$,
exactly matching the standard QAOA qubit mixer.

### Mixer equivalence note

The ring mixer $\exp(-i\beta\, M_d)$ is **not identical** to the old per-qubit
$R_x(2\beta)$ mixer for $d > 2$.  The qubit-emulation mixer mixed individual
bits independently, whereas the ring mixer rotates in the d-dimensional
cyclic-shift basis.  The equivalence holds exactly only for $d = 2$.  For
$d > 2$ the optimisation landscape changes; this is inherent to the
native-qudit formulation and should be evaluated experimentally.

### Measurement

Cirq returns integers $0 \ldots d{-}1$ per qudit directly — no bitstring
decoding is needed.  Sample keys use a dash-separated format (e.g. `"0-3-1"`).
