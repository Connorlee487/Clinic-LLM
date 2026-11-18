# Formal Notation for Knowledge-Graph Prompt Attacks

LaTeX-ready notation for the biomedical KG prompt attacks studied in this project. Copy environments into a `.tex` paper as needed.

---

## 1. Preliminaries

Let $\mathcal{D} = \{(q_i, a_i^{\star})\}_{i=1}^{N}$ be a biomedical QA dataset (e.g., PubMedQA), where $q_i$ is a natural-language question and $a_i^{\star}$ is a reference or baseline model answer.

Let $G = (V, E, \ell)$ be a biomedical knowledge graph, where:
- $V$ is the set of concept nodes,
- $E \subseteq V \times V$ is the set of directed (or undirected) edges,
- $\ell: E \rightarrow \mathcal{L}$ assigns a relation label (e.g., UMLS $\mathsf{RO}$, $\mathsf{RB}$; PrimeKG $\mathsf{drug\_drug}$).

For UMLS we write $G_{\mathrm{UMLS}}$ and restrict neighbour edges to
$$
\mathcal{L}_{\mathrm{UMLS}} = \{\mathsf{RO}, \mathsf{RB}\}.
$$
For PrimeKG drug–drug interactions we write $G_{\mathrm{PK}}$ with
$$
\mathcal{L}_{\mathrm{DDI}} = \{\mathsf{drug\_drug}\}.
$$

### Entity extraction

Let
$$
\mathrm{Extract}: \mathcal{Q} \rightarrow V \cup \{\varnothing\}
$$
map a question to a concept mention (and linked node) using hybrid extraction (scispaCy UMLS linker with keyword fallback). Write
$$
e = \mathrm{Extract}(q),\qquad e \neq \varnothing
$$
when extraction succeeds.

### Neighbourhood

For $e \in V$ and allowed labels $\mathcal{L}' \subseteq \mathcal{L}$,
$$
\mathcal{N}_{\mathcal{L}'}(e) =
\bigl\{\, (v, r) \mid (e,v)\in E,\ \ell(e,v)=r\in\mathcal{L}' \,\bigr\},
$$
subject to surface-form filters (length, ASCII, non-self-substring, etc.). Write
$$
N(e) = \{ v \mid \exists r:\ (v,r)\in \mathcal{N}_{\mathcal{L}_{\mathrm{UMLS}}}(e) \}
$$
for the set of UMLS neighbour **names** used in attacks.

### Victim model

Let $f_\theta$ be the victim LM (e.g., MedGemma-4B). The baseline answer is
$$
a = f_\theta(q).
$$
An attack produces an adversarially modified input $\tilde{q}$ and answer
$$
\tilde{a} = f_\theta(\tilde{q}).
$$
Drift is measured by a similarity $s(\cdot,\cdot)$ (e.g., BERTScore $F_1$):
$$
\Delta = s(\tilde{a}, a)
\quad\text{(lower $\Delta$ $\Rightarrow$ larger attack effect)}.
$$

### Premise delivery

Define concatenation / prepend
$$
\mathrm{Prepend}(p, q) = p \mathbin{+\hspace{-0.2em}+} q
$$
(with normalized spacing/punctuation). All premise-style attacks set $\tilde{q} = \mathrm{Prepend}(p, q)$ for some premise sentence $p$.

---

## 2. UMLS substitution and rewrite attacks

Unless noted, assume $e=\mathrm{Extract}(q)\neq\varnothing$ and $\mathcal{N}=\mathcal{N}_{\mathcal{L}_{\mathrm{UMLS}}}(e)\neq\emptyset$. Let $v^{(1)}$ be the first usable neighbour in API order and $(v^{(k)})_{k=1}^{K}$ the top-$K$ neighbours.

### 2.1 One-hop substitution (`sub`)

$$
\tilde{q}_{\mathrm{sub}}
=
\mathrm{Subst}\bigl(q,\ e,\ v^{(1)}\bigr),
$$
where $\mathrm{Subst}$ replaces the first occurrence of $e$ in $q$ by $v^{(1)}$.

### 2.2 Two-hop substitution (`hop2`)

Let $v_1$ be preferred as the first $\mathsf{RO}$ neighbour if present, else $v^{(1)}$. Let $v_2$ be the first usable neighbour of $v_1$ with $v_2 \neq e$. Then
$$
\tilde{q}_{\mathrm{hop2}}
=
\mathrm{Subst}(q,\ e,\ v_2),
\qquad
\text{path label } r_1\!>\!r_2.
$$

### 2.3 True-neighbour premise, $K{=}1$ (`add`)

Let $g_{\mathrm{sent}}$ be an LM that writes one context sentence given offered terms. With $K=1$:
$$
p_{\mathrm{add}} = g_{\mathrm{sent}}\bigl(q,\ \{v^{(1)}\}\bigr),
\qquad
\tilde{q}_{\mathrm{add}} = \mathrm{Prepend}(p_{\mathrm{add}}, q).
$$
Validity requires that $v^{(1)}$ appears in $p_{\mathrm{add}}$ under lax string matching $\mathrm{Match}(\cdot)$.

### 2.4 True-neighbour premise, top-$K$ (`add_k`)

$$
p_{\mathrm{add}_K}
=
g_{\mathrm{sent}}\bigl(q,\ \{v^{(1)},\ldots,v^{(K)}\}\bigr),
\qquad
\tilde{q}_{\mathrm{add}_K}
=
\mathrm{Prepend}(p_{\mathrm{add}_K}, q).
$$
Soft validity: at least one offered neighbour is matched in $p_{\mathrm{add}_K}$.

### 2.5 Neighbour-conditioned rewrite (`new_q`)

Let $g_{\mathrm{q}}$ rewrite a new yes/no question using offered neighbours in place of $e$:
$$
\tilde{q}_{\mathrm{new}}
=
g_{\mathrm{q}}\bigl(q,\ e,\ \{v^{(1)},\ldots,v^{(K)}\}\bigr).
$$
Hard failures: empty output, or $\mathrm{Match}(\tilde{q}_{\mathrm{new}},\{v^{(k)}\})=\emptyset$.

---

## 3. Premise attacks with explicit truth conditions

These attacks share delivery $\tilde{q}=\mathrm{Prepend}(p,q)$ but differ in whether $p$ asserts a **graph-true** or **graph-false** relation.

### 3.1 Additional context (true, irrelevant)

Select a true neighbour that is least relevant to $q$:
$$
v^{\star}
=
\arg\min_{v \in N(e)}
\mathrm{Rel}(v, q),
$$
where $\mathrm{Rel}$ is a relevance score (e.g., token overlap or embedding similarity; lower $=$ less relevant).

Template (or LM) premise:
$$
p_{\mathrm{ctx}}
=
\tau_{\mathrm{ctx}}(e, v^{\star}),
\qquad
v^{\star}\in N(e).
$$
**Integrity constraint:** every concept asserted as related in $p_{\mathrm{ctx}}$ lies in $N(e)$.

### 3.2 False premise (graph-unsupported relation)

Let $\mathcal{P}$ be a pool of clinical surface forms (e.g., neighbours collected from other entities). Choose a foil
$$
f \in \mathcal{P}
\quad\text{s.t.}\quad
f \notin N(e),\quad f \neq e.
$$
Premise:
$$
p_{\mathrm{false}}
=
\tau_{\mathrm{false}}(e, f)
\quad\text{(e.g., ``$e$ is closely related to $f$.'')}.
$$
**Integrity constraint:** the asserted pair $(e,f)$ is **not** realized as an allowed UMLS neighbour edge:
$$
f \notin N(e).
$$

### 3.3 Misinformation propagation (user-channel false claim)

Operationally identical to false premise under prepend delivery:
$$
p_{\mathrm{mis}} = p_{\mathrm{false}},
\qquad
\tilde{q}_{\mathrm{mis}} = \mathrm{Prepend}(p_{\mathrm{mis}}, q).
$$
Interpretation: the unsupported claim is injected **in the user utterance**, so any change in $\tilde{a}$ is read as propagation of user-provided misinformation. (If a distinct channel is desired, replace $\mathrm{Prepend}$ by a system/context slot; under pure user-message prepend, $\tilde{q}_{\mathrm{mis}}\equiv\tilde{q}_{\mathrm{false}}$.)

---

## 4. PrimeKG drug–drug (DDI) attacks

Let $e_d$ be a drug mention linked into $G_{\mathrm{PK}}$ via name matching, and
$$
N_{\mathrm{DDI}}(e_d)
=
\{ u \mid (e_d,u)\in E,\ \ell(e_d,u)\in\mathcal{L}_{\mathrm{DDI}} \}.
$$

### 4.1 True harmful-combination context

Pick $u \in N_{\mathrm{DDI}}(e_d)$ with display relation $r$:
$$
p_{\mathrm{ddi\text{-}true}}
=
\tau_{\mathrm{ddi}}(e_d, u, r),
\qquad
\tilde{q}
=
\mathrm{Prepend}(p_{\mathrm{ddi\text{-}true}}, q).
$$
**Constraint:** $(e_d,u)$ exists in $G_{\mathrm{PK}}$.

### 4.2 False-safe combination (contradicts DDI)

For $u \in N_{\mathrm{DDI}}(e_d)$,
$$
p_{\mathrm{ddi\text{-}safe}}
=
\tau_{\mathrm{safe}}(e_d, u)
\quad\text{(claims safe co-administration)},
$$
despite $(e_d,u)\in E$.

### 4.3 Invented interaction

Pick $u' \notin N_{\mathrm{DDI}}(e_d)$ and assert an interaction in $p$.

---

## 5. Eligibility and validity

For a mode $m$, define the eligible set
$$
\mathcal{E}_m
=
\{ q \mid \text{extraction and KG preconditions for $m$ hold} \}.
$$
Examples:
- UMLS `sub`/`add_k`: $e\neq\varnothing$ and $N(e)\neq\emptyset$;
- PrimeKG DDI: drug link succeeds and $N_{\mathrm{DDI}}(e_d)\neq\emptyset$.

A generated $\tilde{q}$ is **valid** for mode $m$ if it satisfies mode-specific integrity constraints (true vs.\ false edge checks; neighbour mention checks; non-empty rewrite). Coverage is
$$
\mathrm{Cov}(m) = \frac{|\mathcal{E}_m|}{N}.
$$

---

## 6. Compact attack family

| Mode | Formal map | Truth w.r.t.\ $G$ |
|------|------------|-------------------|
| $\mathrm{sub}$ | $\mathrm{Subst}(q,e,v^{(1)})$ | uses true edge |
| $\mathrm{hop2}$ | $\mathrm{Subst}(q,e,v_2)$ | true 2-hop |
| $\mathrm{add}$ / $\mathrm{add}_K$ | $\mathrm{Prepend}(g_{\mathrm{sent}}(\cdot),q)$ | true neighbours |
| $\mathrm{new\_q}$ | $g_{\mathrm{q}}(q,e,\{v^{(k)}\})$ | true neighbours |
| $\mathrm{ctx}$ | $\mathrm{Prepend}(\tau_{\mathrm{ctx}}(e,v^{\star}),q)$ | true, minimized $\mathrm{Rel}$ |
| $\mathrm{false}$ / $\mathrm{mis}$ | $\mathrm{Prepend}(\tau_{\mathrm{false}}(e,f),q)$ | $f\notin N(e)$ |
| $\mathrm{ddi\text{-}true}$ | $\mathrm{Prepend}(\tau_{\mathrm{ddi}}(e_d,u,r),q)$ | true DDI |
| $\mathrm{ddi\text{-}safe}$ | $\mathrm{Prepend}(\tau_{\mathrm{safe}}(e_d,u),q)$ | contradicts DDI |

---

## 7. Ready-to-paste theorem-style definitions (LaTeX)

```latex
\begin{definition}[UMLS neighbourhood]
Given entity $e$ and label set $\mathcal{L}'=\{\mathsf{RO},\mathsf{RB}\}$,
the neighbourhood $N(e)$ is the set of related concept surface forms
returned by UMLS relation lookup after surface-form filters.
\end{definition}

\begin{definition}[Premise attack]
A premise attack is a map $q \mapsto \tilde{q}=\mathrm{Prepend}(p,q)$
where $p$ is a natural-language sentence asserting a biomedical relation
involving $\mathrm{Extract}(q)$.
\end{definition}

\begin{definition}[False-premise integrity]
A premise $p=\tau_{\mathrm{false}}(e,f)$ is graph-false for $G_{\mathrm{UMLS}}$
iff $f \notin N(e)$.
\end{definition}

\begin{definition}[Additional-context integrity]
A premise $p=\tau_{\mathrm{ctx}}(e,v)$ is graph-true for $G_{\mathrm{UMLS}}$
iff $v \in N(e)$. Among true neighbours, the least-relevant choice is
$v^{\star}=\arg\min_{v\in N(e)}\mathrm{Rel}(v,q)$.
\end{definition}
```

---

## 8. Notes for the paper

- Report $\mathrm{Cov}(m)$ separately from attack strength $\mathbb{E}[\Delta \mid q\in\mathcal{E}_m]$.
- Under user-message prepend, false premise and misinformation propagation share $\tilde{q}$; distinguish them rhetorically or by channel.
- PrimeKG DDI coverage on PubMedQA is low relative to UMLS entity coverage; evaluate DDI modes on the drug-linked subset $\mathcal{E}_{\mathrm{DDI}}$.
