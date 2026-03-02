# Single-Site Baryons at Rest

One common method of lattice QCD is hadron-spectroscopy. To be brief, hadron spectroscopy is the method ofdetermining finite-volume stationary-state energies of QCD from matrices of temporal correlations

$$
    C_{ij}(t_F-t_0)=\langle{}0|TO_{i}(t_F)\bar{O}_{j}(t_0)|0\rangle{}
$$

Here $T$ is a time-ordering operator, $O_i$ and $\bar{O}_{j}$ are sink and source operators respectively. The entire focus of this package is construcing the proper source and sink operators.

The intention of this guide is to walk you through how single-site zero-momentum baryon operators (the simplest type of operator to construct) can be constructed, alongside references to the respective classes and methods used to do so.

## Introduction
The first step to constructing baryon operators on the lattice is to compile a list of *indpendent* baryon operators $B^{\Lambda\lambda{}F}_{j}$, which is defined as a linear superposition of gauge-invariant elemental three quark operators $\Phi$ of the form

$$
    \Phi{}^{ABC}_{\alpha{}\beta{}\gamma{}}(t)=\varepsilon{}_{abc} q^{A}_{a\alpha{}}(t)q^{B}_{b\beta{}}(t)q^{C}_{c\gamma{}}(t)
$$

$$
    \bar{\Phi{}}^{ABC}_{\alpha{}\beta{}\gamma{}}(t)=\varepsilon{}_{abc}q^{C}_{c\gamma{}}(t)q^{B}_{b\beta{}}(t)q^{A}_{a\alpha{}}(t)
$$

where $t$ is time, $\mathbf{p}$ is the momentum, $\mathbf{x}$ is the position at a given lattice site, $q_{a\alpha}$ are the quark fields (denoted by color $a$ and dirac index $\alpha$), and $\varepsilon_{ijk}$ is the Levi-Civita symbol. Since we are dealing with stationary baryons for now, our elemental operators can be simplified to $\Phi^{ABC}_{\alpha\beta\gamma}(t)$. In some cases, these elemental operators are writeen as $B^F_i$.

Constituent quark fields $q_{a\alpha}$ can be created using the constructor `QuarkField.create()`, which we pass any string to denote the flavor. Since the `QuarkField` class is just a handle, we need to explicitly define the color $a$ using the `ColorIdx` constructor and the dirac index $\alpha$ using the `DiracIdx` constructor.

Thus, a generic construction of an elemental operator might look something like:
```python
q = QuarkField.create('q')

a = ColorIdx('a')
b = ColorIdx('b')
c = ColorIdx('c')

i = DiracIdx('i')
j = DiracIdx('j')
k = DiracIdx('k')


Baryon = Eijk(a,b,c) * q[a,i] * q[b,j] * q[c,k]
```
Once we have our elemental operator defined, we need to create an independent set of these operators, this process is much more involved, and requires some manual work to determine this set of operators. 

*How to determine proper set of operators based on quark flavor/displacement*

## Projecting onto the irreps $O^{D}_{h}$
With our list of $M_B$ elemental operators, we now need to project these operators onto the different irreducible representations of $O_h$, known as the octehedral group. The octehedral group contains a total of 48 elements (known as the groups *order* amongst mathemeticians). Because baryons are fermions, we need to work with the double cover of this group $O^{D}_{h}$, which contains 96 elements.

To compute this projection, we first need to determine the representation matrices $W_{ij}$ for each transformation $R\in{}O_{h}^{D}$
$$
U_RB^F_i(t)U^\dagger{}_R=\sum^{M_B}_{j=1}B_j^F(t)W_{ij}(R)
$$
where $U_R$ is a unitary operator acting on the baryon state $B_j^F$. Because these baryons are considered fermions and these quark fields are dependent only on time, our unitary operators reduce $U_R$ to spinor transofrmations $S(R)$ by the relations:
$$
U_Rq_{a\alpha{}j}^{A}(t)U^{\dagger}_R=S^{-1}_{\alpha\beta}(R)q^A_{\alpha\beta{}j}(t)
$$
$$
U_R\bar{q}_{a\alpha{}j}^{A}(t)U^{\dagger}_R=\bar{q}^A_{\alpha\beta{}j}(t)S_{\beta\alpha}(R).
$$
Thus, under the assumption that these tranformations act on each individual quark $q$, we get the following relation
$$
\begin{align*}
U_RB^F_i(t)U^\dagger{}_R & = U_R\Phi^{ABC}_{\alpha\beta\gamma}(t)U_R^\dagger \\
&\Rightarrow\varepsilon_{abc}(U_Rq^{A}_{a\alpha}U_{R}^{\dagger})(U_Rq^{B}_{b\beta}U_R^{\dagger})(U_Rq^{B}_{c\gamma}U_R^{\dagger}) \\
&= \varepsilon_{abc}[S(R)q^{A}_{a\alpha}][S(R)q^{B}_{b\beta}][S(R)q^{C}_{c\gamma}].
\end{align*}
$$
Note that $S(R)$ is also commonly denoted as $\Gamma(R)$, and are $4\times4$ matrices with respect to each group element of $O_h^D$.

Obtaining this representation matrix can be a very lengthy process for even comparatively small sets of elemental operators. The `OperatorRepresentation` class handles the generation of these representation matrices for each group element.


