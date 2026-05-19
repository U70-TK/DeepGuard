## Why Steering Works:

## Toward a Unified View of Language Model Parameter Dynamics

## Ziwen Xu1,2, Chenyan Wu^1 , Hengyu Sun^1 , Haiwen Hong^2 *, Mengru Wang^1 , Yunzhi Yao^1 ,

## Longtao Huang^2 , Hui Xue^2 , Shumin Deng^1 , Zhixuan Chu^1 , Huajun Chen^1 , Ningyu Zhang^1 *

(^1) Zhejiang University, (^2) Alibaba Group

## Abstract

```
Methods for controlling large language mod-
els (LLMs), including local weight fine-tuning,
LoRA-based adaptation, and activation-based
interventions, are often studied in isolation, ob-
scuring their connections and making compari-
son difficult. In this work, we present a uni-
fied view that frames these interventions as
dynamic weight updates induced by a control
signal, placing them within a single concep-
tual framework. Building on this view, we pro-
pose a unified preference-utility analysis. This
analysis separates control effects into two com-
ponents: preference, defined as the tendency
toward a target concept, and utility, defined
as coherent and task-valid generation. Both
components are measured on a shared log-odds
scale using polarity-paired contrastive exam-
ples. Across methods, we observe a consis-
tent trade-off between preference and utility:
stronger control increases preference while pre-
dictably reducing utility. We further explain
this behavior through an activation manifold
perspective, in which control shifts represen-
tations along target-concept directions to en-
hance preference, while utility declines primar-
ily when interventions push representations off
the model’s valid-generation manifold. Finally,
we introduce a new steering approach SPLIT
guided by this analysis that improves prefer-
ence while better preserving utility^1.
```
## 1 Introduction

```
Large Language Models (LLMs) have demon-
strated remarkable capabilities and are increas-
ingly deployed in real-world applications (Zhao
et al., 2023). Growing demands for safety, con-
trollability, and personalization make reliable con-
trol over model behavior a central challenge. To
address this, prior work has developed diverse
*Corresponding Author.
```
(^1) https://github.com/zjunlp/EasyEdit/blob/main/
examples/SPLIT.md.
z
Vector
LoRA
Local Weight
Multiplierm
Utility
Preference
W
**∆** W
W
W
W
b **∆** b
W+
b+
WA
WB
b
b
v
m **∆** W
m **∆** b
b+m **∆** b
W+m **∆** W
b
m
m
mm
0
0
Local WeightLoRA
Vector
Local WeightLoRA
Vector
Multiplierm
Figure 1: The figure illustrates how different methods
operate on the linear layers of the model. We present
a unified view in which diverse large language model
intervention methods are casted as dynamic weight
updates. The right panel shows the changes in model
utility and preference across different control methods
under varying intervention multipliers. Further details
are provided in Section 3.
paradigms for controlling LLMs, spanning training-
time adaptations, such as local weight fine-tuning
and parameter-efficient methods like LoRA (Hu
et al., 2022b; Ding et al., 2023; Mao et al.,
2025), and inference-time interventions, includ-
ing activation-level steering via hidden-state ma-
nipulation (Rimsky et al., 2024; Han et al., 2024;
Bartoszcze et al., 2025; Shu et al., 2025).
Despite their empirical success, these ap-
proaches are often studied in isolation, under dif-
ferent assumptions, objectives, and evaluation pro-
tocols. This fragmentation hinders rigorous com-
parison and obscures shared failure modes. In this
work, as shown in Figure 1, we mathematically
observe that local weight fine-tuning, LoRA, and
activation-level steering can all be formulated as in-
stances of a common dynamic weight update frame-
work (Eq. 1). Building on this unified perspective,
we introduce a preference–utility analysis and show
that, across methods instantiated within this frame-
work, both preference and utility exhibit consistent,
predictable patterns as control strength varies.
hi+1= (W +m 1 ∆W) hi+ (b +m 2 ∆b). (1)

# arXiv:2602.02343v3 [cs.CL] 12 Apr 2026


Note that a particular challenge in controlled
text generation is the trade-off between enforcing
the target concept and preserving task validity:
as control strength increases, the target attribute
is amplified, but undesirable side effects—such
as incoherence, instruction violations, or context
drift—also become more frequent, reducing overall
task effectiveness. Moreover, because control qual-
ity is typically evaluated via realized outputs, degra-
dation in task validity can confound assessments
and obscure the intended concept signal. Guided
by this mechanistic understanding, we propose a
training objective that explicitly optimizes prefer-
ence while preserving utility, and experimentally
demonstrate that it achieves superior performance.

```
Our contributions are as follows:
```
- Unified View. We propose a unified view
    of dynamic weight updates that casts local
weight fine-tuning, parameter-efficient fine-
tuning (e.g., LoRA), and activation interven-
tions (steering) into a common intervention
form. Building on this view, we introduce a
unified preference–utility analysis and show
that, across methods instantiated within the
dynamic-update framework, both preference
and utility exhibit consistent regularities as
control strength varies.
- Preference–Utility Analysis. We introduce
    an activation manifold hypothesis and ana-
    lyze preference and utility under this assump-
    tion, suggesting that preference is jointly de-
    termined by (i) the projection onto a target-
    preference direction and (ii) activation valid-
    ity, which degrades as representations deviate
    from the manifold, while utility degradation
    is primarily driven by this off-manifold devia-
    tion and the resulting activation invalidation.
We further derive two quantitative relation-
ships between the preference log-odds andm,
and between the utility log-odds andm, and
validate them with high-R^2 fits.
- New Steering Method. Guided by this mech-
    anism, we introduce SPLIT, a training ob-
    jective that explicitly optimizes preference
while preserving utility, and demonstrate that
it achieves better overall performance.

## 2 Preliminary

```
2.1 Intermediate Representations in LLMs
During the forward propagation of intermediate
layers in LLMs, several key representations occur
at specific points in the computation, such as FFN
outputs, residual stream states, and linear projec-
tions within the attention mechanism (Query, Key,
Value, and final output). These representations can
be uniformly expressed as the output of an affine
transformation:
```
```
hi+1= Whi+ b, (2)
```
```
wherehiandhi+1denote the input and output
representations of a linear layer, andW,bare its
weights and biases.
For example, in an FFN block, the up-projection
is computed ashmid= Wuphin+ bup, followed
by a non-linear activation,hmid,act = σ(hmid),
and then the down-projection is computed as
hout= Wdownhmid,act+ bdown. Similarly, the
Q,K,V, and output projections in the attention
module follow the same affine form as in Eq. 2.
```
```
2.2 Parameter Update
We consider two parameter adaptation methods
for large language models: Low-Rank Adaptation
(LoRA) and local weight fine-tuning.
LoRA LoRA freezes the original weight ma-
trixWand introduces a trainable low-rank up-
date∆W = BA, whereA ∈ Rr×k,B ∈ Rd×r,
and the rankr ≤ min(d,k). At inference, the
adapted weights are given byW ← W + ∆W.
In its canonical form, LoRA applies only to the
weight matrix while keeping the bias termbfixed,
although extensions exist that also adapt biases.
Local Weight Fine-tuning Local weight fine-
tuning updates parameters within a restricted sub-
set of the network, leaving all other parameters
frozen. It can be applied to any layer or param-
eter type, with full-parameter training represent-
ing the special case where the subset covers the
entire model. A generic update for the weight
matrixWand bias vectorbcan be expressed as:
(W, b) ← (W + ∆W, b + ∆b). In our experi-
ments, parameter updates are applied only to the
FFN down-projection layer.
```
```
2.3 Activation Steering
Activation Steering Activation steering modifies
intermediate representations during inference by
```

```
adding a steering vector to selected activations. Its
mathematical form can be written as
```
```
hi+1= Whi+ b + mv, (3)
```
wherevis a predetermined direction andmis a
scalar coefficient controlling its magnitude. This
approach builds on the linear representation hy-
pothesis (Mikolov et al., 2013; Pennington et al.,
2014; Nanda et al., 2023; Tigges et al., 2023; Park
et al., 2024) that abstract concepts correspond ap-
proximately to linear subspaces of representation
space.
The steering vectorvcan be equivalently ex-
pressed as a bias adjustment∆b, yieldingb ←
b + m∆b. This formulation highlights activation
steering as a special case of dynamic parameter
update, closely related to methods such as LoRA
and local weight fine-tuning.
From a unified perspective, both parameter up-
dates and activation steering operate by injecting
a change vector∆hinto intermediate representa-
tions during forward propagation, differing only in
the mechanism by which∆his generated. More
related works can be found in Appendix 6.

## 3 Unified View of Dynamic Weights in

## Inference

We present a unified framework for dynamic in-
terventions during inference, as illustrated in Fig-
ure 1. Our unified view has three components: (i)
a unified dynamic weights intervention view that
expresses local weight updates, LoRA, and acti-
vation steering as dynamic weight updates, (ii) a
unified analysis view based on preference/utility
log-odds, and (iii) a unified dynamics observation
showing consistent preference–utility response pat-
terns across intervention forms.

```
3.1 Unified Dynamic Weight Formulation
```
We propose a unified framework that encompasses
both parameter update methods and activation steer-
ing methods, by viewing them as dynamic weight
updates. Under this formulation, both can be ex-
pressed through a shared affine transformation view
of intermediate representations; detailed deriva-
tions and formulations are provided in Section 2.
Formally, the dynamic modification of the
weight matrixWand bias vectorbduring infer-
ence can be written as:

```
hi+1= (W +m 1 ∆W) hi+ (b +m 2 ∆b), (4)
```
```
where∆Wand∆bare update terms, andm 1 ,m 2
are scalar scaling coefficients controlling their mag-
nitudes (Fierro and Roger, 2025). In other words,
the original parameters are updated toW′ =
W + m 1 ∆Wandb′= b + m 2 ∆bbefore com-
puting the next-layer activation.
When a model weight is modified, the effect
can be equivalently interpreted from the activation
perspective, as a change to the activation at the
corresponding position. In this view, diverse in-
tervention methods are unified as adding a change
term to the activation:
∆h = m 1 ∆W hi+ m 2 ∆b. (5)
Under this unified view, local weight fine-tuning,
LoRA, and activation steering are all specific in-
stances, differing only in which components are
updated: local weight fine-tuning modifies bothW
andb; LoRA modifiesWvia low-rank factors;
activation steering modifies onlyb. Table 1 sum-
marizes their affine forms, corresponding activation
update, and parameter sizes.
Notably, introducing explicit scaling coefficients
extends traditional formulations and enables contin-
uous control over perturbation strength, a capability
that plays a central role in our subsequent analysis.
```
```
3.2 Unified Analysis View: Preference and
Utility Log-Odds
We analyze intervention effects along two com-
plementary dimensions. Preference denotes the
model’s internal inclination toward a target concept,
independent of whether the model completion is
well-formed. For the prompt “Write a short review
for this restaurant”, generating “The food was ex-
cellent and the service was wonderful” indicates a
positive preference, while “The food was terrible
and the service was disappointing” indicates a neg-
ative preference. Utility denotes the model’s task
competence that is independent of the target con-
cept. It captures whether the model can produce a
task-valid completion that is coherent, relevant to
the prompt, and consistent with the requested for-
mat. For the same prompt, utility is high when the
output is a readable restaurant review, regardless of
polarity. Utility is low when the output is incoher-
ent such as “food food wonderful ??? service 19%
##”, off-topic such as “Here is a Python script to
scrape restaurant data...”, or instruction-violating
even if polarity-bearing words appear.
In controlled generation, performance is typi-
cally evaluated from the realized outputs. When
```

```
Form Unified Affine Formula Activation Impact (∆h) Param. Size
Local Weight (W + m ∆W) hi+ (b + m ∆b) m (∆W hi+ ∆b) din× dout+ dout
LoRA (W + m BA) hi+ b m (BA hi) din× r + r× dout
Steering Vector W hi+ (b + m ∆b) m ∆b dout
```
```
Table 1: All methods in our unified framework, expressed under the affine weight-update formulation and their
corresponding activation changes∆h.dinanddoutdenote the input and output dimensions of the layer;ris the
LoRA rank with r ≪ min(din,dout).
```
preference is increased at the expense of util-
ity, completions often become incoherent or
instruction-violating, reducing usability and ob-
scuring the intended concept signal under output-
based evaluation. Therefore, effective model con-
trol should shift preference while preserving utility.
Notation. Given a queryq, we construct a polar-
ity pair of completions: a concept-positive answer
Apand a concept-negative answerAn. We denote
their conditional probabilities asP (Ap | q)and
P (An | q), and define the corresponding cross-
entropy losses asLp ≜ − logP (Ap | q)and
Ln≜− logP (An| q). We further introduce latent
preference probabilitiesP (pp| q)andP (pn| q),
as well as a polarity-invariant task-success proba-
bility P (u| q).
Preference–Utility Factorization. Following
prior work that assumes concept directions are mu-
tually orthogonal (Bigelow et al., 2025), we like-
wise treat concept preference as independent from
task utility for a given queryq. Concretely, for a
polarity pair (Ap,An), we decompose

```
P (Ap| q) = P (u| q)P (pp| q),
P (An| q) = P (u| q)P (pn| q), (6)
```
whereP (u| q)is shared across the pair andP (pp|
q) + P (pn| q) = 1.
Preference Log-odds. The shared utility cancels
in the likelihood ratio, yielding

```
PrefOdds(q)≜ log
```
```
P (pp| q)
P (pn| q)
=Ln−Lp. (7)
```
```
Utility Log-odds. The total probability mass as-
signed to the matched pair recovers utility,P (u |
q) = P (Ap | q) + P (An | q); substituting
P (A| q) = e−Lgives
```
```
UtilOdds(q)≜ log
P (u| q)
1 − P (u| q)
```
```
= log
e−Lp+ e−Ln
1 − e−Lp− e−Ln
```
### . (8)

```
We usePrefOddsandUtilOddsthroughout to
track how interventions shift concept preference
versus task utility on a common additive scale, with
additional derivations in Appendix D.
```
```
3.3 Unified Dynamics Observation
Experimental Setup. We evaluate dynamic in-
terventions on two types of tasks: (i) a person-
ality tendency classification task (Psychopathy),
and (ii) open-ended generation using PowerSeek-
ing and the top 10 concept subsets from AxBench.
We run experiments onGemma-2-9B-ITat layer
20 andQwen-2.5-7B-Instructat layer 14, fol-
lowing Bigelow et al. (2025), and consider three
intervention types: local weight, low-rank adap-
tation LoRA, and vector. We train each type us-
ing both the SFT objective and the RePS objec-
tive. Additionally, for vector, we include a train-
free method called DiffMean (Marks and Tegmark,
2023). More details are provided in Appendix A.
```
```
Metrics. For each queryqwith matched answers
(Ap,An), we compute preference and utility log-
odds in Eqs.(7)and(8). These metrics allow us to
track how preference and utility evolve as we vary
the intervention scale m.
```
```
Unified Dynamics. Experimental results show
that, under the unified perspective framework,
different intervention forms exhibit remarkably
consistent dynamic patterns. As shown in Fig-
ure 2, localized weight updates, low-rank adap-
tation (LoRA), and vector-based interventions dis-
play highly similar overall curve shapes. Addi-
tional results are included in Appendix A.
For preference log-odds, all methods typically
follow a three-stage pattern when plotted against
the steering factorm: for small|m|, they enter
a Linear Region, where log-odds grows approxi-
mately linearly withm(Bigelow et al., 2025); this
is followed by a Transitional Region with a notice-
able change in trend, and finally a Convergence
Region where the curve flattens and stabilizes.
Utility log-odds, in contrast, generally peak near
```

Figure 2: Unified preference and utility dynamics under steering. Solid lines represent preference log-odds, and
dashed lines represent utility log-odds. The top panel shows steering with vector-form parameter modifications, and
the bottom panel shows parametric interventions including LoRA and local weight updates. Results are shown for
theGemma-2-9B-ITmodel on the AxBench dataset, evaluated over its top 10 concept subsets. The horizontal axis
corresponds to the steering factor.

m≈ 0 , and remain near their maximum within this
narrow range. As|m|increases, utility gradually
declines and eventually stabilizes.
These patterns reveal a unified steering response
of preference and utility.

## 4 Capability Dynamics: Mechanism

## Analysis and Optimization

Motivated by the unified preference–utility dy-
namics observed across intervention forms (Fig-
ure 2), this section provides a mechanistic account
and an empirical characterization. We take an
activation-manifold perspective and introduce a
simple validity-decay factor to capture the tendency
for capability to degrade as steering pushes activa-
tions away from the activation manifold, without
committing to a specific underlying geometry. On
this basis, we express preference as the combined
effect of (i) steering-induced preference projection

```
changes and (ii) validity decay, while utility is mod-
eled as being dominated by the validity decay term.
Finally, under this hypothesis we formalize how
the steering factormshapes both preference and
utility log-odds, and show via curve-fitting that the
resulting forms match the observed log-odds–m
dynamics well across settings.
```
```
4.1 Activation Manifold Hypothesis
Prior work suggests that model activations of-
ten concentrate on low-dimensional, manifold-like
sets in representation space (Bricken et al., 2023;
Wollschläger et al., 2025). Adopting this manifold
perspective, we analyze additive steering as a trans-
lation of hidden states along an approximately fixed
direction in activation space. Intuitively, small
translations may adjust model behavior in a tar-
geted way, whereas large translations may push
representations away from the high-density region
learned during training, increasing the risk of a
```

```
Figure 3: Mechanism of projection gain and validity decay. Right: An activation manifold view illustrating
Assumption 4.1. An activationPlies on or near the manifold. Steering using preference vectorvwith scaling
factorsm+andm−movesPtoP 1 andP 2 , corresponding to intersections with the manifold. Top-left: Projection
gain. Projections onto the utility axis exhibit limited variation, whereas projections along the preference direction
differ betweenP 1 andP 2 , suggesting that steering primarily influences preference-related components. Bottom-left:
Steering-induced validity decay. As assumed in Assumption 4.2, increasing steering factor increases off-manifold
deviation, leading to a monotonic decrease in validity and degraded downstream decoding.
```
```
representation–decoder mismatch and thus degrad-
ing general capability.
We formalize this view with two assumptions.
```
```
Assumption 4.1 (Training-Induced Ac-
tivation Manifold). Fix a layerl with
hidden dimension dl. There exists a
low-dimensional set (or its neighborhood)
Ml⊂ Rdlsuch that for inputsxdrawn
from a set of stably handled inputsXstable,
the corresponding activationhl(x)lies on
or nearMlwith high probability:
```
```
Pr
x∼Xstable
[d(hl(x),Ml)≤ ε]≥ 1 − δ, (9)
```
```
whered(·,Ml) denotes distance toMl,
ε > 0 is a neighborhood radius, andδ ∈
(0, 1) is a small failure probability.
```
Assumption 4.1 asserts that pre-training induces
a “typical” region of activation space where rep-
resentations concentrate for stably handled inputs.
We next introduce a generic notion of represen-
tation validity, which is high nearMland de-
creases as hidden states move away from it. This
abstraction avoids committing to a specific geome-
try forMlwhile retaining the key implication: suf-
ficiently off-manifold activations are more likely to

```
be decoded unreliably by the remaining network.
```
```
Assumption 4.2 (Steering-Induced Valid-
ity Decay). LetFl→Ldenote the remain-
der of the model from layerlto the out-
put logits. There exists a validity function
Vl: Rdl→ [0, 1]that is monotonically non-
increasing ind(h,Ml), capturing how well
Fl→Lcan stably decode an activation h.
For an additive steering intervention at layer
l,
̃hl(m) = hl+ m ∆h, (10)
```
```
with steering direction∆hand steering fac-
torm∈ R, define the average validity over
stably handled inputs:
```
```
D(m)≜ Ex∼Xstable
```
```
h
Vl
```
### 

```
̃hl(m)
```
```
i
```
. (11)

```
We assume thatD(m) ∈ [0, 1]decreases
with|m|(i.e., larger interventions induce
larger off-manifold shifts on average), and
that the resulting capability degradation is
dominated by this validity decay.
```
```
To connect Assumptions 4.1–4.2 to a concrete
functional form, we view steering as moving an
activation along a one-dimensional line in repre-
sentation space, ̃hl(m) = hl+ m ∆h. Under the
```

```
manifold hypothesis, degradation is governed pri-
marily by how far this line trajectory departs from
the typical region nearMl, so it is natural to model
D(m)as a smooth function of the (signed) dis-
tance along this line to the nearest “on-manifold”
locations. In particular, as illustrated in Fig. 3, the
steered trajectory may intersect the manifold neigh-
borhood at one or more values{mi}(e.g., one for
m > 0 and one form < 0 ). We therefore model
validity as being highest near these intersection
points and decaying as|m− mi| grows.
A convenient choice that is positive, smooth,
and exhibits heavy-tailed distance-based decay is
the rational quadratic (RQ) form, widely used in
kernel methods and Gaussian processes to model
multi-scale, polynomial-rate attenuation with dis-
tance (Rasmussen, 2004). Prior research on con-
trollability metrics has established that model steer-
ability is often asymmetric (Miehling et al., 2025),
exhibiting varying degrees of responsiveness along
different directions of the same dimension. Moti-
vated by this observation, we employ a piecewise
parameterized model to quantify degradation:
```
```
D(m) =
```
### 

### 

### 

### 

```
1 +(m−m+)
```
```
2
L+
```
```
−p+
if m≥ 0

1 +(m−m−)
```
```
2
L−
```
−p−
if m < 0
(12)
wherem±corresponds to the signed distance from
the original activation pointPto an on-manifold in-
tersection pointP±along the steering line (Fig. 3);
L±sets the characteristic scale of decay and re-
flects how fast the distance-to-manifold grows
along the steering direction (larger when the direc-
tion is locally parallel to the manifold and smaller
when it cuts across it); andp±controls the decay
rate (tail heaviness) as the trajectory moves away
from the manifold neighborhood.

```
4.2 Preference Capability: Projection Gain
With Decay
```
We study how additive steering changes a model’s
preference through intermediate activations. An
intervention at layerlupdates the hidden state as
̃h(m) = h + m ∆h.
Prior work under LRH-style assumptions often
models preference probability with a logistic form,
P (pp | h) = σ

### 

```
− (ωTph + bp)
```
### 

```
, whereωpis
the preference vector. Separately, work on activa-
tion geometry suggests that after low-dimensional
projection (e.g., PCA), opposite preference labels
are often approximately linearly separable. Un-
```
```
der the activation-manifold view, this motivates a
two-dimensional preference plane and a preference
direction whose signed coordinate reflects prefer-
ence intensity. Our contribution is to incorporate
validity attenuationD(·)(Assumption 4.2) to ac-
count for off-manifold steering.
To model this, we write the steered preference
probability as
```
```
P (pp| ̃h(m)) = σ
```
### 

### −

### 

```
ωTph+αpm
```
### 

```
Dp(m)−bp
```
### 

### ,

### (13)

```
whereαp≜ωTp∆hmeasures how much the steer-
ing direction aligns with the preference vector:α
is large when∆his aligned withωp, andαp= 0
when ∆h is orthogonal toωp.
This implies the preference log-odds
```
```
log
P (pp| ̃h(m))
1 − P (pp| ̃h(m))
```
### =

### 

```
ωTph+αpm
```
### 

```
Dp(m)+bp.
(14)
```
```
Key implication (linear regime→non-
linear collapse). From Eq.(14), them-
dependence enters asαpmDp(m). When
|m−m±|≪ L±, Eq.(12)givesDp(m)≈
1 , hence preference log-odds is approxi-
mately linear inmwith slopeα(match-
ing the near-linear regime in Bigelow et al.
(2025)). As|m− m±|grows and becomes
comparable to or larger thanL±, Eq.(12)
implies substantial decay inDp(m), so at-
tenuation dominates and the log-odds re-
sponse becomes strongly nonlinear and can
collapse off-manifold.
```
```
Fitting Form. We fit the measured preference
log-odds as a function of m with
```
```
log
```
```
P (pp| ̃h(m))
1 − P (pp| ̃h(m))
```
```
= (αpm+βp)Dp(m) + bp,
(15)
where βp=ωTph is a per-example constant (since
h is fixed for a given input), and bpis an offset.
Fit Results. Table 2 reports the fit quality of
Eq.(15), withR^2 values exceeding 0.95 across
most settings. These results validate the model’s
ability to accurately characterize the dynamics of
preference log-odds. Details are in Appendix E.
```
```
4.3 Utility Capability: Only Validity Decay
Utility Log-odds Under Manifold-Validity De-
cay. Leth ∈ Rdldenote the activation at layer
```

```
l. We quantify utility capability by the log-odds
of positive vs. negative utility outcomes (up/un).
Similar to preference, we assume utility is also as-
sociated with a directionωuin activation space.
Under steering ̃h(m) = h + m ∆h,, we model
```
```
log
P (u| ̃h(m))
1 − P (u| ̃h(m))
```
```
=ωTuhDu(m) + bu, (16)
```
whereDu(m)follows the manifold-validity decay
in Eq.(12)and decreases with|m|. Crucially, for
preference steering directions we typically have
ωTu∆h≈ 0 , so utility is affected primarily through
validity decay rather than a direct projection term.
Fitting form. Accordingly, we fit the measured
utility log-odds with a pure decay curve:

```
log
P (u| ̃h(m))
1 − P (u| ̃h(m))
```
```
= βuDu(m) + bu, (17)
```
whereβuis the baseline log-odds andbuis an offset
capturing residual bias.
Fit Results. Table 2 reports the fit quality of
Eq.(17). Uniformly highR^2 values (typically
> 0. 97 ) suggest utility variations under preference
steering are well captured by the proposed formu-
lation. Additional details are in Appendix E.

## 5 Method

```
5.1 Preference–Utility Joint Optimization
Building on the preceding mechanistic analy-
sis, we propose Steering with Preference–UtiLity
IntervenTion (SPLIT), a training objective improv-
ing preference while delaying utility degradation.
Utility Loss. To preserve utility, we train on both
the positive and negative samples for the same input
using the language-modeling cross-entropy:
```
```
Lutil= λpLp+ λnLn, (18)
```
whereLpandLnare the token-level cross-entropy
losses on positive and negative samples, respec-
tively, and λp,λncontrol their relative weight.
Preference Loss. By Eq.(7), the loss gapLn−
Lpis exactly the preference log-odds. We therefore
maximize this gap via a hinge-style margin loss:

```
Lpref= γ· σ(θ− (Ln−Lp)), (19)
```
whereσ(·)is ReLU andθis a margin threshold,
andγtrades off preference improvement against
utility preservation.

```
Form Method Preference R
```
(^2) ↑ Utility R (^2) ↑
PSY PWR AXB Avg PSY PWR AXB Avg
Gemma-2-9B-IT
Weight SFT 0.970.980.990.980.980.990.980.
RePS 0.990.990.990.990.960.990.990.
LoRA SFT 0.920.990.980.960.980.990.990.
RePS 0.830.990.990.940.990.990.990.
Vector DiffMean0.970.990.990.980.970.990.980.
SFT 0.930.970.980.960.990.990.990.
RePS 0.990.980.950.970.990.990.990.
Qwen-2.5-7B-IT
Weight SFT 0.990.990.990.990.990.990.980.
RePS 0.990.990.990.990.990.990.970.
LoRA SFT 0.970.990.990.980.990.990.990.
RePS 0.940.990.990.970.980.960.990.
Vector DiffMean0.980.950.980.970.990.990.970.
SFT 0.930.970.980.960.980.990.990.
RePS 0.970.980.930.960.990.980.990.
Table 2: Curve fitting performance. Results on Psy-
chopathy (PSY), PowerSeeking (PWR), and AxBench
(AXB). We reportR^2 (higher is better), measuring
alignment between theoretical curves and empirical data.
Color intensity indicatesR^2 values. Consistently dark
shading shows high fidelity across settings (R^2 > 0. 95 ).
Final Objective. We combine the two compo-
nents as
L =Lutil+Lpref. (20)
5.2 Experiment Results.
We evaluate the proposed preference-utility joint
optimization method under three intervention
forms: local weight update, low-rank adaptation
(LoRA), and activation vector steering. As shown
in Table 3, our approach consistently achieves
higher scores compared with baseline methods
across all three intervention types. These results
demonstrate the robustness and generality of the
proposed optimization strategy. See Appendix A
and B for more experimental details and results.

## 6 Related Work

```
Mechanism. Most activation steering methods
assume linear structure in activation space, control-
ling concepts by adding scaled direction vectors
to hidden states (Mikolov et al., 2013; Pennington
et al., 2014; Nanda et al., 2023; Tigges et al., 2023;
Park et al., 2024; Wang et al., 2024; Yao et al.,
2025; Zhang et al., 2026; Hu et al., 2026). Build-
ing on this view, Bigelow et al. (2025) show that
```

```
Model Form Method
```
```
Psychopathy PowerSeeking AxBench
Acc(%, 0–100)↑ Concept(0–4)↑ Concept(0–2)↑ Harmonic(0–2)↑
```
```
Gemma-2-9B-IT
```
```
Vanilla Vanilla 50.00 1.87 0.4750 0.
```
```
Local Weight
```
```
SFT 100.00 3.50 1.6625 1.
REPS 100.00 3.39 1.7750 1.
SPLIT (Ours) 100.00 3.59 1.8500 1.
```
```
LoRA
```
```
SFT 100.00 3.41 1.7625 1.
REPS 99.00 3.44 1.7375 1.
SPLIT (Ours) 100.00 3.56 1.7750 1.
```
```
Vector
```
```
DiffMean 53.00 2.95 1.1625 1.
SFT 97.00 3.30 1.7000 1.
REPS 98.00 3.61 1.7000 1.
SPLIT (Ours) 99.00 3.62 1.8500 1.
```
```
Qwen-2.5-7B-IT
```
```
Vanilla Vanilla 50.00 2.24 0.4500 0.
```
```
Local Weight
```
```
SFT 97.00 3.53 1.5375 1.
REPS 96.00 3.24 1.6875 1.
SPLIT (Ours) 98.00 3.66 1.7000 1.
```
```
LoRA
```
```
SFT 99.00 3.05 1.4875 1.
REPS 100.00 3.34 1.4875 1.
SPLIT (Ours) 100.00 3.59 1.7375 1.
```
```
Vector
```
```
DiffMean 55.00 3.17 0.9500 0.
SFT 97.00 3.58 1.5750 1.
REPS 88.00 3.63 1.7375 1.
SPLIT (Ours) 98.00 3.65 1.8125 1.
```
Table 3: Main task performance of steering methods evaluated on three datasets. Psychopathy is reported with
classification accuracy (Acc, %), PowerSeeking is evaluated using LLM-judge preference scores on a 0–4 scale,
and AxBench reports the concept score and the harmonic mean (HM) over concept, instruction, and fluency scores,
each on a 0–2 scale, as evaluated by an LLM judge. All methods perform inference-time interventions on hidden
representations. Best and second-bestresults are highlighted within each model and intervention form.

steering yields an approximately linear trend in pos-
terior odds, but mainly in the small-scale regime.
Recent studies further report non-monotonic or
adverse effects under stronger steering, challeng-
ing a naive global linearity assumption (Bricken
et al., 2023; Wollschläger et al., 2025). Meanwhile,
representation-manifold work provides a comple-
mentary geometric lens for understanding steering
and its limitations (Modell et al., 2025; Li and He,
2025; Xie et al., 2025).

Activation Steering. Activation steering controls
the behavior of LLMs by intervening in hidden
states during forward propagation, using steering
vectors to control single attributes as well as more
complex behavioral targets (Turner et al., 2023;
Rimsky et al., 2024; van der Weij et al., 2024; Rahn
et al., 2024; Scalena et al., 2024; Tan et al., 2024;
Bhattacharjee et al., 2024; Postmus and Abreu,
2024; Konen et al., 2024; Hazra et al., 2024; Han
et al., 2025; Jiang et al., 2025). However, recent
studies have shown that the coarse-grained nature
of activation steering can lead to degradation in
model utility (Wang et al., 2025; Wu et al., 2025a).
Cao et al. (2024); Wu et al. (2025b) introduce ex-
plicit preference learning objectives to optimize
activation steering, enabling more precise control.

```
Parameter-Efficient Fine-Tuning. Parameter-
Efficient Fine-Tuning (PEFT) methods, including
adapters and LoRA, show that effective adapta-
tion of LLMs does not require updating all param-
eters. LoRA achieves performance comparable
to full fine-tuning, indicating that adaptation re-
lies on structured low-rank weight updates rather
than full parameter changes (Hu et al., 2022a; Zi
et al., 2023; Zhang et al., 2023; Hayou et al., 2024;
Kopiczko et al., 2024; Zhang et al., 2024; Chen
et al., 2024). Local weight updates further reveal
that LLM knowledge is highly localized, as modi-
fying a small subset of parameters in specific layers
suffices to change factual associations (Geva et al.,
2021; Zaken et al., 2022; Ding et al., 2022; Chen,
2024; Yang et al., 2025).
```
## 7 Conclusion

```
We propose a unified dynamic weight update frame-
work that incorporates parameter updates, LoRA,
and activation interventions, revealing a consistent
preference–utility decay pattern in the log-odds
space. Building on this mechanistic insight, we de-
sign a joint optimization method that consistently
improves preference while mitigating utility degra-
dation across diverse intervention forms, demon-
strating versatility and robustness.
```

## Limitations

While our unified dynamic weight update frame-
work provides a coherent perspective on LLM
control and enables predictable preference–utility
trade-offs, several limitations remain. First, our
analysis assumes that model representations lie
near a well-structured activation manifold, which
may not hold for extremely large or highly di-
verse models, potentially reducing the accuracy
of our quantitative predictions. Second, our exper-
iments focus primarily on attribute-level control
(e.g., sentiment, style), leaving the applicability to
complex multi-turn reasoning or safety-critical con-
tent largely unexplored. Third, while our proposed
training objective mitigates the utility–preference
trade-off, it does not guarantee complete avoid-
ance of undesirable side effects such as subtle in-
struction violations or context drift under extreme
control strengths. Finally, our study evaluates con-
trol under pre-defined intervention multipliers, and
generalization to adaptive or dynamically varying
control signals requires further investigation.

## Ethics Statement

```
Controlled LLM generation carries inherent ethical
considerations. While our framework aims to im-
prove controllability and preserve task validity, it
could potentially be misused to manipulate user per-
ception, amplify biased viewpoints, or generate per-
suasive yet misleading content. Our experiments
are conducted on standard benchmark datasets and
do not involve sensitive personal information. We
emphasize that the proposed methods should be
deployed with human oversight, adherence to fair-
ness guidelines, and robust monitoring to prevent
harm. By explicitly modeling preference–utility
trade-offs, we aim to make LLM interventions
more interpretable and safer, but responsible us-
age depends on context-aware implementation and
alignment with societal norms.
```
## Acknowledgements

We would like to express our sincere gratitude to
the anonymous reviewers for their thoughtful and
constructive feedback. This work was supported by
the National Natural Science Foundation of China
(No. 62576307, No. NSFCU23B2055, No. NS-
FCU19B2027), the Fundamental Research Funds
for the Central Universities (226-2023-00138), the
Yongjiang Talent Introduction Programme (2021A-
156-G), and the Information Technology Center

```
and State Key Lab of CAD&CG, Zhejiang Univer-
sity. This work was supported by Alibaba Group
through Alibaba Innovative Research Program.
```
## References

```
Lukasz Bartoszcze, Sarthak Munshi, Bryan Sukidi, Jen-
nifer Yen, Zejia Yang, David Williams-King, Linh
Le, Kosi Asuzu, and Carsten Maple. 2025. Represen-
tation engineering for large-language models: Survey
and research challenges. CoRR, abs/2502.17601.
Amrita Bhattacharjee, Shaona Ghosh, Traian Rebedea,
and Christopher Parisien. 2024. Towards inference-
time category-wise safety steering for large language
models. CoRR, abs/2410.01174.
Eric J. Bigelow, Daniel Wurgaft, YingQiao Wang,
Noah D. Goodman, Tomer D. Ullman, Hidenori
Tanaka, and Ekdeep Singh Lubana. 2025. Belief
dynamics reveal the dual nature of in-context learn-
ing and activation steering. CoRR, abs/2511.00617.
Trenton Bricken, Adly Templeton, Joshua Batson,
Brian Chen, Adam Jermyn, Tom Conerly, Nick
Turner, Cem Anil, Carson Denison, Amanda Askell,
Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas
Schiefer, Tim Maxwell, Nicholas Joseph, Zac
Hatfield-Dodds, Alex Tamkin, Karina Nguyen,
Brayden McLean, Josiah E Burke, Tristan Hume,
Shan Carter, Tom Henighan, and Christopher
Olah. 2023. Towards monosemanticity: Decom-
posing language models with dictionary learning.
Transformer Circuits Thread. Https://transformer-
circuits.pub/2023/monosemantic-
features/index.html.
Yuanpu Cao, Tianrong Zhang, Bochuan Cao, Ziyi Yin,
Lu Lin, Fenglong Ma, and Jinghui Chen. 2024. Per-
sonalized steering of large language models: Versa-
tile steering vectors through bi-directional preference
optimization. In Advances in Neural Information
Processing Systems 38: Annual Conference on Neu-
ral Information Processing Systems 2024, NeurIPS
2024, Vancouver, BC, Canada, December 10 - 15,
2024.
Huajun Chen. 2024. Large knowledge model: Per-
spectives and challenges. DATA INTELLIGENCE,
6(3):587–620.
Songlin Chen, Weicheng Wang, Xiaoliang Chen, Peng
Lu, Zaiyan Yang, and Yajun Du. 2024. Llama-lora
neural prompt engineering: A deep tuning frame-
work for automatically generating chinese text logical
reasoning thinking chains. DATA INTELLIGENCE,
6(2):375–408.
Ning Ding, Yujia Qin, Guang Yang, Fuchao Wei, Zong-
han Yang, Yusheng Su, Shengding Hu, Yulin Chen,
Chi-Min Chan, Weize Chen, Jing Yi, Weilin Zhao,
Xiaozhi Wang, Zhiyuan Liu, Hai-Tao Zheng, Jianfei
Chen, Yang Liu, Jie Tang, Juanzi Li, and Maosong
Sun. 2022. Delta tuning: A comprehensive study of
```

```
parameter efficient methods for pre-trained language
models. CoRR, abs/2203.06904.
```
Ning Ding, Yujia Qin, Guang Yang, Fuchao Wei,
Zonghan Yang, Yusheng Su, Shengding Hu, Yulin
Chen, Chi-Min Chan, Weize Chen, et al. 2023.
Parameter-efficient fine-tuning of large-scale pre-
trained language models. Nature machine intelli-
gence, 5(3):220–235.

Constanza Fierro and Fabien Roger. 2025. Steering
language models with weight arithmetic. CoRR,
abs/2511.05408.

Mor Geva, Roei Schuster, Jonathan Berant, and Omer
Levy. 2021. Transformer feed-forward layers are key-
value memories. In Proceedings of the 2021 Confer-
ence on Empirical Methods in Natural Language Pro-
cessing, EMNLP 2021, Virtual Event / Punta Cana,
Dominican Republic, 7-11 November, 2021, pages
5484–5495. Association for Computational Linguis-
tics.

Chi Han, Jialiang Xu, Manling Li, Yi Fung, Chenkai
Sun, Nan Jiang, Tarek Abdelzaher, and Heng Ji. 2024.
Word embeddings are steers for language models.
In Proceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers), pages 16410–16430.

Peixuan Han, Cheng Qian, Xiusi Chen, Yuji Zhang,
Denghui Zhang, and Heng Ji. 2025. Internal activa-
tion as the polar star for steering unsafe llm behavior.
arXiv preprint arXiv:2502.01042.

Soufiane Hayou, Nikhil Ghosh, and Bin Yu. 2024.
Lora+: Efficient low rank adaptation of large models.
In Forty-first International Conference on Machine
Learning, ICML 2024, Vienna, Austria, July 21-27,
2024. OpenReview.net.

Rima Hazra, Sayan Layek, Somnath Banerjee, and Sou-
janya Poria. 2024. Safety arithmetic: A framework
for test-time safety alignment of language models by
steering parameters and activations. In Proceedings
of the 2024 Conference on Empirical Methods in Nat-
ural Language Processing, EMNLP 2024, Miami, FL,
USA, November 12-16, 2024, pages 21759–21776.
Association for Computational Linguistics.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and
Weizhu Chen. 2022a. Lora: Low-rank adaptation of
large language models. In The Tenth International
Conference on Learning Representations, ICLR 2022,
Virtual Event, April 25-29, 2022. OpenReview.net.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
Weizhu Chen, et al. 2022b. Lora: Low-rank adapta-
tion of large language models. ICLR, 1(2):3.

Yi Hu, Jiaqi Gu, Ruxin Wang, Zijun Yao, Hao Peng,
Xiaobao Wu, Jianhui Chen, Muhan Zhang, and
Liangming Pan. 2026. Towards a mechanistic un-
derstanding of large reasoning models: A survey

```
of training, inference, and failures. arXiv preprint
arXiv:2601.19928.
Houcheng Jiang, Junfeng Fang, Ningyu Zhang, Guo-
jun Ma, Mingyang Wan, Xiang Wang, Xiangnan He,
and Tat-seng Chua. 2025. Anyedit: Edit any knowl-
edge encoded in language models. arXiv preprint
arXiv:2502.05628.
Kai Konen, Sophie Jentzsch, Diaoulé Diallo, Peer
Schütt, Oliver Bensch, Roxanne El Baff, Dominik
Opitz, and Tobias Hecking. 2024. Style vectors for
steering generative large language models. In Find-
ings of the Association for Computational Linguis-
tics: EACL 2024, St. Julian’s, Malta, March 17-22,
2024 , pages 782–802. Association for Computational
Linguistics.
Dawid Jan Kopiczko, Tijmen Blankevoort, and Yuki M.
Asano. 2024. Vera: Vector-based random matrix
adaptation. In The Twelfth International Conference
on Learning Representations, ICLR 2024, Vienna,
Austria, May 7-11, 2024. OpenReview.net.
Tianhong Li and Kaiming He. 2025. Back to basics: Let
denoising generative models denoise. arXiv preprint
https://arxiv.org/pdf/2511.13720.
Yuren Mao, Yuhang Ge, Yijiang Fan, Wenyi Xu, Yu Mi,
Zhonghao Hu, and Yunjun Gao. 2025. A survey on
lora of large language models. Frontiers of Computer
Science, 19(7):197605.
Samuel Marks and Max Tegmark. 2023. The geometry
of truth: Emergent linear structure in large language
model representations of true/false datasets. CoRR,
abs/2310.06824.
Erik Miehling, Michael Desmond, Karthikeyan Nate-
san Ramamurthy, Elizabeth M. Daly, Kush R. Varsh-
ney, Eitan Farchi, Pierre Dognin, Jesus Rios, Djallel
Bouneffouf, Miao Liu, and Prasanna Sattigeri. 2025.
Evaluating the prompt steerability of large language
models. In Proceedings of the 2025 Conference of
the Nations of the Americas Chapter of the Associ-
ation for Computational Linguistics: Human Lan-
guage Technologies (Volume 1: Long Papers), pages
7874–7900, Albuquerque, New Mexico. Association
for Computational Linguistics.
Tomás Mikolov, Wen-tau Yih, and Geoffrey Zweig.
```
2013. Linguistic regularities in continuous space
word representations. In Human Language Technolo-
gies: Conference of the North American Chapter of
the Association of Computational Linguistics, Pro-
ceedings, June 9-14, 2013, Westin Peachtree Plaza
Hotel, Atlanta, Georgia, USA, pages 746–751. The
Association for Computational Linguistics.
Alexander Modell, Patrick Rubin-Delanchy, and Nick
Whiteley. 2025. The origins of representation
manifolds in large language models. CoRR,
abs/2505.18235.
Neel Nanda, Andrew Lee, and Martin Wattenberg. 2023.
Emergent linear representations in world models


```
of self-supervised sequence models. In Proceed-
ings of the 6th BlackboxNLP Workshop: Analyzing
and Interpreting Neural Networks for NLP, Black-
boxNLP@EMNLP 2023, Singapore, December 7,
2023 , pages 16–30. Association for Computational
Linguistics.
```
Kiho Park, Yo Joong Choe, and Victor Veitch. 2024.
The linear representation hypothesis and the geome-
try of large language models. In Forty-first Interna-
tional Conference on Machine Learning, ICML 2024,
Vienna, Austria, July 21-27, 2024. OpenReview.net.

Jeffrey Pennington, Richard Socher, and Christopher D.
Manning. 2014. Glove: Global vectors for word
representation. In Proceedings of the 2014 Confer-
ence on Empirical Methods in Natural Language Pro-
cessing, EMNLP 2014, October 25-29, 2014, Doha,
Qatar, A meeting of SIGDAT, a Special Interest Group
of the ACL, pages 1532–1543. ACL.

Joris Postmus and Steven Abreu. 2024. Steering
large language models using conceptors: Improv-
ing addition-based activation engineering. CoRR,
abs/2410.16314.

Nate Rahn, Pierluca D’Oro, and Marc G. Bellemare.

2024. Controlling large language model agents with
entropic activation steering. CoRR, abs/2406.00244.

Carl Edward Rasmussen. 2004. Gaussian Processes
in Machine Learning, pages 63–71. Springer Berlin
Heidelberg, Berlin, Heidelberg.

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong,
Evan Hubinger, and Alexander Matt Turner. 2024.
Steering llama 2 via contrastive activation addition.
In Proceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics (Volume 1:
Long Papers), ACL 2024, Bangkok, Thailand, August
11-16, 2024, pages 15504–15522. Association for
Computational Linguistics.

Daniel Scalena, Gabriele Sarti, and Malvina Nissim.

2024. Multi-property steering of large language mod-
els with dynamic activation composition. CoRR,
abs/2406.17563.

Dong Shu, Xuansheng Wu, Haiyan Zhao, Daking Rai,
Ziyu Yao, Ninghao Liu, and Mengnan Du. 2025. A
survey on sparse autoencoders: Interpreting the in-
ternal mechanisms of large language models. CoRR,
abs/2503.05613.

Daniel Tan, David Chanin, Aengus Lynch, Dimitrios
Kanoulas, Brooks Paige, Adrià Garriga-Alonso,
and Robert Kirk. 2024. Analyzing the general-
ization and reliability of steering vectors. CoRR,
abs/2407.12404.

Curt Tigges, Curt Tigges, Oskar Hollinsworth, Curt
Tigges, Atticus Geiger, Atticus Geiger, Oskar
Hollinsworth, Neel Nanda, Neel Nanda, Atticus
Geiger, and Neel Nanda. 2023. Linear repre-
sentations of sentiment in large language models.
[http://arxiv.org/abs/2310.15154.](http://arxiv.org/abs/2310.15154.)

```
Alexander Matt Turner, Lisa Thiergart, Gavin Leech,
David Udell, Juan J Vazquez, Ulisse Mini, and Monte
MacDiarmid. 2023. Activation addition: Steering
language models without optimization. arXiv e-
prints, pages arXiv–2308.
Teun van der Weij, Massimo Poesio, and Nandi Schoots.
```
2024. Extending activation steering to broad skills
and multiple behaviours. CoRR, abs/2403.05767.
Mengru Wang, Ziwen Xu, Shengyu Mao, Shumin Deng,
Zhaopeng Tu, Huajun Chen, and Ningyu Zhang.
2025. Beyond prompt engineering: Robust behavior
control in llms via steering target atoms. In Proceed-
ings of the 63rd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), ACL 2025, Vienna, Austria, July 27 - August 1,
2025 , pages 23381–23399. Association for Computa-
tional Linguistics.
Mengru Wang, Yunzhi Yao, Ziwen Xu, Shuofei Qiao,
Shumin Deng, Peng Wang, Xiang Chen, Jia-Chen
Gu, Yong Jiang, Pengjun Xie, et al. 2024. Knowl-
edge mechanisms in large language models: A sur-
vey and perspective. In Findings of the Association
for Computational Linguistics: EMNLP 2024, pages
7097–7135.
Tom Wollschläger, Jannes Elstner, Simon Geisler, Vin-
cent Cohen-Addad, Stephan Günnemann, and Jo-
hannes Gasteiger. 2025. The geometry of refusal
in large language models: Concept cones and repre-
sentational independence. In Forty-second Interna-
tional Conference on Machine Learning, ICML 2025,
Vancouver, BC, Canada, July 13-19, 2025. OpenRe-
view.net.
Zhengxuan Wu, Aryaman Arora, Atticus Geiger, Zheng
Wang, Jing Huang, Dan Jurafsky, Christopher D Man-
ning, and Christopher Potts. 2025a. Axbench: Steer-
ing llms? even simple baselines outperform sparse
autoencoders. arXiv preprint arXiv:2501.17148.
Zhengxuan Wu, Qinan Yu, Aryaman Arora, Christo-
pher D. Manning, and Christopher Potts. 2025b. Im-
proved representation steering for language models.
CoRR, abs/2505.20809.
Zhenda Xie, Yixuan Wei, Huanqi Cao, Chenggang
Zhao, Chengqi Deng, Jiashi Li, Damai Dai, Huazuo
Gao, Jiang Chang, Liang Zhao, Shangyan Zhou,
Zhean Xu, Zhengyan Zhang, Wangding Zeng,
Shengding Hu, Yuqing Wang, Jingyang Yuan, Lean
Wang, and Wenfeng Liang. 2025. mhc: Manifold-
constrained hyper-connections. arXiv preprint
https://arxiv.org/pdf/2512.24880.
Ziwen Xu, Shuxun Wang, Kewei Xu, Haoming
Xu, Mengru Wang, Xinle Deng, Yunzhi Yao,
Guozhou Zheng, Huajun Chen, and Ningyu Zhang.
2025. Easyedit2: An easy-to-use steering frame-
work for editing large language models. CoRR,
abs/2504.15133.
Wanli Yang, Fei Sun, Rui Tang, Hongyu Zang, Du Su,
Qi Cao, Jingang Wang, Huawei Shen, and Xueqi


```
Cheng. 2025. Fine-tuning done right in model edit-
ing. CoRR, abs/2509.22072.
```
Yunzhi Yao, Jiaxin Qin, Ningyu Zhang, Haoming Xu,
Yuqi Zhu, Zeping Yu, Mengru Wang, Yuqi Tang,
Jia-Chen Gu, Shumin Deng, et al. 2025. Rethink-
ing knowledge editing in reasoning era. Authorea
Preprints.

Elad Ben Zaken, Yoav Goldberg, and Shauli Ravfogel.

2022. Bitfit: Simple parameter-efficient fine-tuning
for transformer-based masked language-models. In
Proceedings of the 60th Annual Meeting of the As-
sociation for Computational Linguistics (Volume 2:
Short Papers), ACL 2022, Dublin, Ireland, May 22-
27, 2022, pages 1–9. Association for Computational
Linguistics.

Hengyuan Zhang, Zhihao Zhang, Mingyang Wang, Zun-
hai Su, Yiwei Wang, Qianli Wang, Shuzhou Yuan,
Ercong Nie, Xufeng Duan, Qibo Xue, et al. 2026.
Locate, steer, and improve: A practical survey of
actionable mechanistic interpretability in large lan-
guage models. arXiv preprint arXiv:2601.14004.

Qingru Zhang, Minshuo Chen, Alexander Bukharin,
Pengcheng He, Yu Cheng, Weizhu Chen, and
Tuo Zhao. 2023. Adaptive budget allocation for
parameter-efficient fine-tuning. In The Eleventh In-
ternational Conference on Learning Representations,
ICLR 2023, Kigali, Rwanda, May 1-5, 2023. Open-
Review.net.

Ruiyi Zhang, Rushi Qiang, Sai Ashish Somayajula, and
Pengtao Xie. 2024. Autolora: Automatically tuning
matrix ranks in low-rank adaptation based on meta
learning. In Proceedings of the 2024 Conference
of the North American Chapter of the Association
for Computational Linguistics: Human Language
Technologies (Volume 1: Long Papers), NAACL 2024,
Mexico City, Mexico, June 16-21, 2024, pages 5048–

5060. Association for Computational Linguistics.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang,
Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen
Zhang, Junjie Zhang, Zican Dong, et al. 2023. A
survey of large language models. arXiv preprint
arXiv:2303.18223, 1(2).

Bojia Zi, Xianbiao Qi, Lingzhi Wang, Jianan Wang,
Kam-Fai Wong, and Lei Zhang. 2023. Delta-lora:
Fine-tuning high-rank parameters with the delta of
low-rank matrices. CoRR, abs/2309.02411.

## A Experiment Details

Datasets. We evaluate our dynamic intervention
methods on three datasets: (i) Psychopathy (per-
sonality tendency classification), (ii) PowerSeek-
ing (open-ended generation), and (iii) the top-
concept subsets from AxBench (open-ended genera-
tion). To support training and evaluation under our
paired-setting and data availability constraints, we
construct task-specific train/test splits as follows.

```
For Psychopathy, we sample 500 instances for train-
ing and 100 for testing. For PowerSeeking, we sam-
ple 500 instances for training and 200 for testing.
For AxBench, since its original test set is randomly
sampled from an instruction-following corpus and
does not provide matched positive/negative answer
pairs, we re-split the original 72 instances per con-
cept for each of the top-10 concept subsets into 64
training instances and 8 test instances.
```
```
Evaluation and Metrics. For the experiments in
§3.3, following Bigelow et al. (2025), we compute
preference and utility log-odds (Eqs.(7)and(8))
for each queryqwith matched answers(Ap,An)
on both training and test sets, and vary the inter-
vention scalemto track their changes. For the
final performance evaluation in §5.2, we adopt
dataset-specific metrics. For Psychopathy, follow-
ing Bigelow et al. (2025), we report classification
accuracy (Acc). For PowerSeeking, following Cao
et al. (2024), we usegpt-4.1-minito score gener-
ations on the test set on a 0–4 scale. For AxBench,
following Wu et al. (2025a), we usegpt-4.1-mini
to evaluate concept score, instruction score, and
fluency score on the test set, each on a 0–2 scale;
we report the concept score and the harmonic
mean over the three scores.
```
```
Baselines. We evaluate multiple methods under
three intervention forms: local weight updates,
LoRA, and vector interventions. For each form, we
train interventions with either the SFT objective or
the RePS objective (Wu et al., 2025a). For vector
interventions, we additionally include a train-free
baseline DiffMean (Marks and Tegmark, 2023).
We also report Vanilla results without any steering.
```
```
Intervention Setup. We run experiments
on Gemma-2-9B-IT at layer 20 and on
Qwen-2.5-7B-Instruct at layer 14, follow-
ing Bigelow et al. (2025). We consider three
intervention forms: local weight updates, LoRA,
and vector interventions. For local weight and
LoRA, we train intervention parameters on
the MLP down-projection matrix; for vector
interventions, we apply the intervention directly
to the residual stream. For hyperparameters, we
largely follow the default settings in Wu et al.
(2025a); Xu et al. (2025). We optimize with
AdamW and a linear learning-rate scheduler. We
also perform reasonable hyperparameter tuning to
ensure stable and competitive performance.
```

Results: Unified Dynamics Observation.
Figures 2 and 4 show the unified prefer-
ence–utility dynamics of the Gemma-2-9B-IT
and Qwen-2.5-7B-IT models on the AxBench
dataset, evaluated over the top-10 concept subsets.
And figure 5 shows the unified preference-utility
dynamics on Power-seeking and Psychopathy
datasets under different models. We observe that
the utility can increase under slight perturbations
ofmin either the positive or negative direction. In
some cases, this suggests that the origin may not
lie exactly on the utility manifold, implying that
the utility is not always strictly optimal at m = 0.
Results: Performance Comparison. Table 3
compares our method with various baselines under
different intervention forms (local weight, LoRA,
and vector) on two base models. Across inter-
vention forms, our method remains competitive
with strong baselines, and often improves concept
metrics while maintaining comparable or higher
harmonic scores. The gains are most consistent
under LoRA and vector, where our approach typ-
ically strengthens concept control relative to both
SFT- and RePS-trained variants, and achieves the
best or near-best harmonic mean on AXBENCH in
multiple settings. Under full weight updates, we
observe smaller but still stable differences, with
our method remaining comparable and without an
apparent drop in utility. Overall, the results indi-
cate that the proposed optimization transfers across
different steering forms and can provide reliable,
albeit sometimes incremental, improvements.

## B Results of DPO-based Method

```
To provide a more exhaustive analysis of
preference-based objectives in the context of behav-
ior steering, we implemented a DPO-based method
following BiPO (Cao et al., 2024).
```
```
Form Psy. Pow. AXB.
Acc Concept Concept Harmonic
(%)↑ (0–4)↑ (0–2)↑ (0–2)↑
Local Weight 91.00 1.91 0.525 0.
LoRA 99.00 1.87 0.550 0.
Vector 99.00 1.93 0.575 0.
```
```
Table 4: Performance of the DPO-based method on
Gemma-2-9B-IT across different adaptation forms.
The evaluation encompasses Psychopathy (Psy.), Pow-
erSeeking (Pow.), and AxBench (AXB.) metrics.
```
```
As shown in Table 4, the DPO-based method
performs consistently with prior observations re-
```
```
Symbol Description
```
```
Unified Analysis Framework
m The steering scalar coefficient.
PrefOdds(q) Preference log-odds (Ln−Lp).
(7)
UtilOdds(q) Utility log-odds. (8)
P (u|q) Latent utility probability.
P (p±|q) Latent preference probability.
P (•|h) Equivalent to P (•|q) as the
weights remain unchanged.
Lp,Ln Cross-entropy losses correspond-
ing to Apand An.
```
```
Mechanistic Manifold Model
Ml The activation manifold of stably
handled inputs at layer l.
D(m) Average validity decay function.
m± Distance from P to P± along the
steering line in Fig 3
L± Characteristic scale of decay.
p± Asymptotic decay rate.
```
```
Joint Optimization
Lutil Utility loss component.
Lpref Preference loss component.
```
```
Table 5: Notations for Key Concepts. A summary of
the specialized symbols introduced for the unified anal-
ysis, mechanistic modeling, and optimization objective.
```
```
ported in RePS (Wu et al., 2025b). In contrast,
our method explicitly models the preference–utility
trade-off, achieving a more balanced improvement
across dimensions, indicating performance beyond
a standard DPO-style approach.
```
## C List of Mathematical Symbols

```
The Table 5 below lists the important symbols used
in this paper.
```
## D Derivations and Implementation

## Details for Log-Odds

```
This appendix derives the loss-based forms of
Eqs.(7)–(8)from the preference–utility indepen-
dence assumption, and states how we compute the
required sequence losses.
```

```
Figure 4: Unified preference and utility dynamics under steering. Solid lines represent preference log-odds, and
dashed lines represent utility log-odds. The top panel shows steering with vector-form parameter modifications, and
the bottom panel shows parametric interventions including LoRA and local weight updates. Results are shown for
theQwen-2.5-7B-ITmodel on the AxBench dataset, evaluated over its top 10 concept subsets. The horizontal axis
corresponds to the steering factor.
```
```
D.1 From preference–utility independence to
log-odds
```
```
For a queryqand a polarity pair(Ap,An), we
assume
```
```
P (Ap| q) = P (u| q)P (pp| q),
P (An| q) = P (u| q)P (pn| q), (21)
```
with P (pp| q) + P (pn| q) = 1.

```
Preference log-odds. Taking the ratio of(21)
cancels P (u| q):
```
```
P (Ap| q)
P (An| q)
```
### =

```
P (pp| q)
P (pn| q)
```
### . (22)

```
Applying log(·) gives
```
```
PrefOdds(q)≜ log
```
```
P (pp| q)
P (pn| q)
= log
```
```
P (Ap| q)
P (An| q)
```
### .

### (23)

```
Using the loss definitionL≜− logP (A| q), we
have P (A| q) = e−L, and thus
```
```
PrefOdds(q) = log
```
```
e−Lp
e−Ln
=Ln−Lp, (24)
which matches Eq. (7).
Utility probability and log-odds. Summing(21)
and using P (pp| q) + P (pn| q) = 1 yields
P (Ap| q) + P (An| q) = P (u| q)

P (pp| q) + P (pn| q)
```
### 

```
= P (u| q). (25)
```

```
(a) Powerseeking Results
```
```
(b) Psychopathy Results
```
Figure 5: Unified preference and utility dynamics under steering. Solid lines represent preference log-odds,
and dashed lines represent utility log-odds. Figure (a) shows the unified preference and utility dynamics of the
power-seeking dataset under two different models, while Figure (b) shows the results for the psychopathy dataset.
The horizontal axis corresponds to the steering factor.

Therefore,

```
UtilOdds(q)≜ log
P (u| q)
1 − P (u| q)
```
```
= log
```
```
P (Ap| q) + P (An| q)
1 − P (Ap| q)− P (An| q)
```
### .

### (26)

```
SubstitutingP (A | q) = e−L(A|q)gives the loss
form
```
```
UtilOdds(q) = log
```
```
e−Lp+ e−Ln
1 − e−Lp− e−Ln
```
### , (27)

```
which matches Eq.(8). Note that since(Ap,An)
are only two candidate continuations, we typically
```

```
have P (Ap| q) + P (An| q) < 1.
```
```
D.2 Computing sequence losses
LetA = (y 1 ,...,yT)be a completion (exclud-
ing the query/prompt tokens). We compute the se-
quence negative log-likelihood (cross-entropy loss)
under teacher forcing:
```
```
L(A| q)≜− logP (A| q)
```
### =−

### XT

```
t=
```
```
logP (yt| q,y<t). (28)
```
We then setLp≜L(Ap| q)andLn≜L(An| q)
and plug them into Eqs. (24) and (27).
Length normalization (optional). WhenApand
Anhave different lengths, we optionally use the
mean lossL ̄(A | q)≜ L(A | q)/Tin place of
L(A| q)to reduce length effects. In that case, the
corresponding quantities use e−L ̄instead of e−L.

```
D.3 Preference log-odds and Utility log-odds
Here we show how the preference and utility ca-
pability can be represented as(14)and(16). We
take preference log-odds as example. First, the
conditional probability before steering is given by:
```
```
P (pp| h) = σ
```
### 

```
−ωTphDp(0)− bp
```
### 

### ,

```
= σ(η) (29)
```
where η≜−ωTphDp(0)− bp.
When an intervention at layerlupdates the hiden
state as ̃h(m) = h + m∆h, we can get the steered
preference probablity as (13) :

```
P (pp| ̃h(m)) = σ
```
### 

```
−(ω⊤ph + αpm)Dp(m)− bp
```
### 

```
Next, we can represent the initial preference log-
odds as−η:
```
```
log
P (pp| h)
P (pn| h)
```
```
= log
P (pp| h)
1 − P (pn| h)
```
```
= log
```
```
σ(η)
1 − σ(η)
```
```
= log
```
```
1 /(1 + eη)
1 − 1 /(1 + eη)
```
```
= log
1 /(1 + eη)
eη/(1 + eη)
= loge−η
=−η
=ωTphDp(0) + bp (30)
```
```
Finally, when we steeringhby ̃h(m), we can get
preference log-odds by:
```
```
log
P (pp| ̃h(m))
P (pn|h ̃(m))
```
```
=−ηsteered
```
```
= (ω⊤ph + αpm)Dp(m) + bp
(31)
```
```
For utility capability, we have:
```
```
P (u| ̃h(m)) = σ
```
### 

```
−ωTu ̃h(m)Du(m)− bu
```
### 

### .

### (32)

```
For preference steering directions, we typically
haveωTu∆h≈ 0. So we can quantify utility capa-
bility by:
```
```
log
P (u| ̃h(m))
1 − P (u| ̃h(m))
```
```
=ω⊤uhDu(m) + bu
```
## E Fitting Experiment Details

```
E.1 Fitting Results on Test Set
To further validate our theoretical model, we per-
formed parameterized fitting on the test set using
the SLSQP algorithm, strictly enforcing continuity
between positive and negative segments at the ori-
gin. As shown in Table 6, the direct fitting yielded
high goodness-of-fit values (R^2 > 0. 95 ) for most
methods. This confirms that the steering effect
follows a deterministic trajectory predicted by our
theory rather than random perturbations, thereby
validating the proposed interaction mechanism.
```
```
E.2 Analysis of Generalization Ability
Following the validation of our theoretical mech-
anism, we conducted train-to-test transfer exper-
iments to evaluate the extent to which different
methods decouple "concepts" from specific inputs.
Theoretical curve parameters were derived solely
from training data and applied directly to the test
set for prediction (Table 7).
Robust Generalization Overall, the fitted curves
generalize well to held-out data, with vector-based
interventions achieving consistently strong R^2
across most settings. Input-dependent approaches
such as LoRA- and local-weight-based methods
also generalize well in many cases, but exhibit
larger variance across datasets and occasional fail-
ures, suggesting that input-dependent updates can
be more sensitive to the evaluation distribution.
```

```
Type Method Preference R
```
(^2) ↑ Utility R (^2) ↑
PSY PWR AXB Avg PSY PWR AXB Avg
Gemma-2-9B-IT
Weight SFT 0.96 0.96 0.99 0.97 0.98 0.93 0.99 0.
RePS 0.95 0.98 0.95 0.96 0.98 0.93 0.99 0.
LoRA SFT 0.99 0.99 0.98 0.99 0.98 0.98 0.99 0.
RePS 0.99 0.99 0.98 0.99 0.99 0.99 0.99 0.
Vector DiffMean 0.89 0.99 0.99 0.96 0.94 0.99 0.98 0.
SFT 0.90 0.97 0.97 0.95 0.98 0.99 0.99 0.
RePS 0.96 0.98 0.96 0.97 0.96 0.99 0.99 0.
Qwen-2.5-7B-IT
Weight SFT 0.99 0.82 0.99 0.93 0.99 0.99 0.95 0.
RePS 0.99 0.89 0.97 0.95 0.98 0.99 0.90 0.
LoRA SFT 0.70 0.95 0.98 0.88 0.99 0.99 0.99 0.
RePS 0.88 0.95 0.95 0.93 0.98 0.99 0.98 0.
Vector DiffMean 0.99 0.99 0.98 0.99 0.97 0.94 0.98 0.
SFT 0.99 0.99 0.97 0.98 0.97 0.95 0.99 0.
RePS 0.99 0.98 0.93 0.97 0.96 0.96 0.98 0.
Table 6: Performance comparison of curve fitting quality on test sets. We evaluate the models on three datasets:
Psychopathy (PSY), PowerSeeking (PWR), and AXBench (AXB).
Type Method Preference R
(^2) ↑ Utility R (^2) ↑
PSY PWR AXB Avg PSY PWR AXB Avg
Gemma-2-9B-IT
Weight SFT 0.96 0.85 -4.25 -0.81 0.98 0.98 0.61 0.
RePS 0.99 0.98 -1.16 0.27 0.96 0.98 0.73 0.
LoRA SFT 0.92 0.99 -0.56 0.45 0.98 0.99 0.96 0.
RePS 0.83 0.99 0.74 0.85 0.98 0.99 0.97 0.
Vector DiffMean -0.14 0.99 0.75 0.53 0.97 0.99 0.97 0.
SFT 0.90 0.91 0.74 0.85 0.98 0.99 0.99 0.
RePS 0.98 0.89 0.65 0.84 0.99 0.99 0.99 0.
Qwen-2.5-7B-IT
Weight SFT 0.99 -0.32 -12.03 -3.79 0.99 -1.33 -3.07 -1.
RePS 0.96 0.98 -3.82 -0.63 0.99 0.42 -1.15 0.
LoRA SFT 0.97 0.98 -0.40 0.52 0.99 0.99 0.95 0.
RePS 0.94 0.99 -0.13 0.60 0.98 0.96 0.96 0.
Vector DiffMean 0.86 0.94 0.80 0.87 0.37 0.99 0.97 0.
SFT 0.67 0.92 0.71 0.77 0.96 0.99 0.99 0.
RePS 0.97 0.93 0.74 0.88 0.99 0.98 0.98 0.
Table 7: Generalization ability of curve fitting. The table reports theR^2 scores where the curves are fitted on the
training set and evaluated on the test set across three datasets: Psychopathy (PSY), PowerSeeking (PWR), and
AXBench (AXB). Negative values imply that the fitted curves do not generalize well to unseen data.


