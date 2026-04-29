# The second conclusion is the attribution analysis research framework

## 1. Research objectives

,** LiNGAM **,:

1. Between the existing 12 driving factors and ground subsidence hazards, which causal directions are reasonable and which are unreasonable based on empirical knowledge.
2. Under future scenarios, whether these causal relationships appear stably, and which causal chains are highly reproducible and interpretable in different `SSP × ` combinations.

In other words, the core of this part of the work is not `RF + LiNGAM`, but:

**Prior knowledge constraints + LiNGAM repeated calculations + stable causal chain screening**

---

## 2. General idea

.

### Step one: Construct a priori causal constraints based on empirical knowledge

Artificially constrains the potential causal directions between 12 factors and disaster nodes to form a set of **prior knowledge matrix**.
The goal of this step is to exclude obviously unreasonable edges, such as:

- `LAI -> Depth_to_Bedrock`
- `UrbanFrac -> Distance_to_Fault`
- `Precip -> Distance_to_karst`
- `Disaster -> `

In other words, the role of prior knowledge is not to "replace" LiNGAM, but to provide a reasonable search space for LiNGAM to avoid obviously wrong causal directions in the physical or geological sense.

### Step 2: Repeat LiNGAM under future scenarios

, LiNGAM,:

- `ssp1-2040`
- `ssp1-2060`
- `ssp1-2080`
- `ssp1-2100`
- …
- `ssp5-2100`

Repeat `100` times for each `SSP × ` combination.
Causal edges, directions, signs, and effect sizes are recorded for each run.
:

- Appearance rate `>= 80%`
- Does not violate prior knowledge
- Implementation note.
- The direction and symbol are relatively stable

,.

---

## 3. Node system

This study plans to use `12` driving factors, plus `1` disaster nodes, for a total of `13` nodes.

### 3.1 Three major categories of factors

It is currently recommended to use the three categories that have become more mature in the project:

| | | |
|---|---|---|
| Anthropogenic activities | UF | UrbanFrac |
| Anthropogenic activities | IP | ImperviousIndex |
| Anthropogenic activities | PT | PopTotal |
| Anthropogenic activities | LAI | LAI |
| Anthropogenic activities | WTD | WTD |
| Climate change | PR | Precip |
| Climate change | TAS | Tas |
| Climate change | HUSS | Huss |
| Hydrogeology | DK | Distance_to_karst |
| Hydrogeology | DB | Depth_to_Bedrock |
| Hydrogeology | DF | Distance_to_Fault_m |
| Hydrogeology | HDS | HDS |

### 3.2 Disaster node

The disaster node suggestion is recorded as:

- Historical stage: `Disaster`
- Future stage: `Predicted disaster risk` or `Sinkhole risk`

Note:

- `Disaster`.
- ,** / **.

This means that although "ground collapse disaster" will still be discussed in the future stage, the statistical end node is actually:

**Predict disaster risks rather than actual observed disaster events in the future. **

### 3.3

The end node suggestions for the future stage are clearly recorded as:

- `Predicted sinkhole probability`

Moreover, this probability is not an arbitrarily constructed probability, but strictly follows the continuous probability caliber in the current GWR classification workflow.

The actual process in the current project is:

1. The GWR model first outputs the original **regression value/raw score**.
2. Then map this raw score to `0-1` using the robust sigmoid transformation fitted during the training phase.
3. This `0-1` continuous value is entered into LiNGAM as the final response node for the future attribution phase.

Therefore, the use of:

- Discrete risk level
- Jenks natural breakpoint classification results
- Artificial threshold two-class label

:

- **Continuous risk probability**

### 3.4 Implementation caliber of GWR probability mapping

This must be consistent with existing model training code.

Reference code location:

- [Train_National_GWR_classification.ipynb](/path/to/sinkhole-risk-china/code/3_gwr_model_train/national/GWR/Train_National_GWR_classification.ipynb)
- [gwr_sigmoid_utils.py](/path/to/sinkhole-risk-china/code/mgtwr/gwr_sigmoid_utils.py)

The current actual caliber is not an ordinary direct sigmoid, but:

- First fit `transform_metadata` on the training set raw GWR scores
- These include:
  - `center`
  - `scale`
  - `clip_z`
- Then use the same set of training set metadata to do robust sigmoid mapping for the future scenario raw GWR scores.

The corresponding function is:

- `mgtwr.gwr_sigmoid_utils.gwr_scores_to_probabilities`

Its core logic is:

1. `z = (raw_score - center) / scale`
2. `z` `[-clip_z, clip_z]`
3. sigmoid `0-1`

The most important constraints here are:

**Future scenarios cannot refit their own sigmoid parameters individually, but must reuse the transform metadata obtained during the training phase. **
The same sigmoid transform metadata must be used for all scenario-year predictions.

Otherwise, the probability scales between different scenarios will no longer be comparable, and subsequent LiNGAM end node definitions will also lose consistency.

---

## 4. Step one: Prior knowledge construction framework

## 4.1

The goal of the prior knowledge matrix is to limit LiNGAM’s search to the scope of "physical significance, geoscience significance and temporal logic significance".

This step recommends building two types of rules:

### A. Hard constraints

.

Suggestions include:

- `Disaster -> ` Forbidden
- Any dynamic factor `->` Static geological background factor Prohibited
- `-> DK / DB / DF / HDS`
- Climate Change Factor `-> DK / DB / DF / HDS` Prohibited
- Ecological state variable `LAI -> DB / DF / DK / HDS` prohibited
- Any self-loop is prohibited

The "static geological background factors" suggestions here include:

- `DK`
- `DB`
- `DF`
- `HDS`

### 4.1.1

In order to allow the prior matrix to be directly encoded, it is recommended to split the 13 nodes into 5 groups of roles:

#### A. Geological background root node

- `DK`
- `DB`
- `DF`
- `HDS`

** / exogenous nodes** .

means:

- Do not accept any incoming edges from other nodes
- In particular, it is prohibited to:
  - `UF/IP/PT/LAI/WTD -> DK/DB/DF/HDS`
  - `PR/TAS/HUSS -> DK/DB/DF/HDS`
  - `Disaster probability -> DK/DB/DF/HDS`

4 ,:

- A more conservative approach: all prohibited and treated as concurrent background conditions
- Slightly looser approach: Allowing the direction between `DK/DB/DF/HDS`, but not as the key explanation of the paper

From the perspective of interpretability, it is recommended to give priority to:

- **There is no direction edge between geological background variables**

#### B. Climate forcing root node

- `PR`
- `TAS`
- `HUSS`

exogenous nodes.

The reason is:

- These variables represent the external climate state within the same time window in the current study design
- ,

Suggested hard constraints:

- Forbidden `UF/IP/PT/LAI/WTD -> PR/TAS/HUSS`
- `DK/DB/DF/HDS -> PR/TAS/HUSS`
- Forbidden `Disaster probability -> PR/TAS/HUSS`

It is recommended to disable the direction edges between `PR/TAS/HUSS` first because:

- Implementation note.
- ""

#### C. Human activity root-intermediary node

- `PT`
- `UF`
- `IP`

It is recommended to make finer direction constraints:

- `PT -> UF` allowed
- `PT -> IP` allowed
- `UF -> IP` allowed

The opposite direction is recommended to be prohibited:

- `UF -> PT`
- `IP -> PT`
- `IP -> UF`

The reason is that a more reasonable explanation under the current scale is:

- Population pressure drives urban expansion
- Urban expansion further increases impermeability

#### D. Environment status intermediary node

- `WTD`
- `LAI`

These two nodes are recommended as the main intermediary layers.

##### Allowed incoming edge of `WTD`

- `DK/DB/DF/HDS -> WTD`
- `PR/TAS/HUSS -> WTD`
- `PT/UF/IP -> WTD`

##### `WTD`’s prohibited entry edge

- `Disaster probability -> WTD`
- `LAI -> WTD` It is recommended to ban it first

It is recommended to give priority to:

- `WTD -> LAI`
- instead of `LAI -> WTD`

Because under the current research scale, it is easier to explain as:

- Groundwater burial depth or groundwater environment restricts vegetation status

instead of:

- Vegetation reversely determines the depth of groundwater

##### Allowed incoming edge of `LAI`

- `PR/TAS/HUSS -> LAI`
- `PT/UF/IP -> LAI`
- `WTD -> LAI`
- , `DK/DB/DF/HDS -> LAI`

##### `LAI`’s prohibited entry/exit edge

- `Disaster probability -> LAI`
- Forbidden `LAI -> DK/DB/DF/HDS`
- `LAI -> PR/TAS/HUSS`
- `LAI -> PT/UF/IP`

#### E. Disaster risk end node

- `Predicted sinkhole probability`

** sink node** .

Allowed edges:

- All 12 drivers can point to this node

:

- `Predicted sinkhole probability -> `

### 4.1.2

Before actually implementing LiNGAM, it is recommended to clarify the matrix semantics into three states:

- `forbidden`
- `allowed`
- `unspecified`

:

- **forbidden**: According to geoscience/physics/time logic, it is clearly not allowed
- **allowed**: The direction is reasonable and LiNGAM is allowed to determine whether it exists
- **unspecified**: Theoretically not completely exclusive, but currently not the focus of explanation

When analyzing the paper, the main text figure only explains:

- `allowed`

### 4.1.3 A more specific set of recommended allowed edges

In order to reduce ambiguity during subsequent encoding, it is recommended to allow the following edges first:

- `PT -> UF`
- `PT -> IP`
- `UF -> IP`
- `PR -> WTD`
- `TAS -> WTD`
- `HUSS -> WTD`
- `PR -> LAI`
- `TAS -> LAI`
- `HUSS -> LAI`
- `DK -> WTD`
- `DB -> WTD`
- `DF -> WTD`
- `HDS -> WTD`
- `PT -> WTD`
- `UF -> WTD`
- `IP -> WTD`
- `PT -> LAI`
- `UF -> LAI`
- `IP -> LAI`
- `WTD -> LAI`
- `DK -> Disaster probability`
- `DB -> Disaster probability`
- `DF -> Disaster probability`
- `HDS -> Disaster probability`
- `PR -> Disaster probability`
- `TAS -> Disaster probability`
- `HUSS -> Disaster probability`
- `PT -> Disaster probability`
- `UF -> Disaster probability`
- `IP -> Disaster probability`
- `LAI -> Disaster probability`
- `WTD -> Disaster probability`

### 4.1.4 A more specific set of recommended forbidden edges

It is recommended that the prohibited edges explicitly written into the prior matrix include:

- `LAI -> DK`
- `LAI -> DB`
- `LAI -> DF`
- `LAI -> HDS`
- `WTD -> DK`
- `WTD -> DB`
- `WTD -> DF`
- `WTD -> HDS`
- `UF -> DK/DB/DF/HDS`
- `IP -> DK/DB/DF/HDS`
- `PT -> DK/DB/DF/HDS`
- `PR -> DK/DB/DF/HDS`
- `TAS -> DK/DB/DF/HDS`
- `HUSS -> DK/DB/DF/HDS`
- `Disaster probability -> `
- `LAI -> WTD`
- `UF -> PR/TAS/HUSS`
- `IP -> PR/TAS/HUSS`
- `PT -> PR/TAS/HUSS`
- `PR/TAS/HUSS -> DK/DB/DF/HDS`

### B. Soft expectations

These edges are not required to exist, but are more likely to be accepted if present.

Suggestions include:

- `PR / TAS / HUSS -> LAI`
- `PR / TAS / HUSS -> WTD`
- `UF / IP / PT -> LAI`
- `UF / IP / PT -> WTD`
- `DK / DB / DF / HDS -> WTD`
- `DK / DB / DF / HDS -> Disaster`
- `PR / TAS / HUSS -> Disaster`
- `UF / IP / PT / LAI / WTD -> Disaster`

Soft constraints are mainly used in the interpretation phase, not to mechanically tailor the results.

---

## 4.2 Recommended causal hierarchy

In order to avoid too much confusion in the future network, it is recommended to adopt hierarchical understanding:

### The first layer: background constraint layer

- `DK`
- `DB`
- `DF`
- `HDS`

Such variables mainly represent relatively stable geological/geomorphological background and should not be reversely determined by human activities or short-term climate conditions.

### Second layer: Environmental adjustment layer

- `PR`
- `TAS`
- `HUSS`
- `UF`
- `IP`
- `PT`
- `LAI`
- `WTD`

These variables allow for interactions, but it is recommended to give priority to the direction of "Climate/Human Activities -> Surface and Subsurface Environmental State".

### :

- `Disaster` or `Predicted disaster risk`

The disaster node should be used as the end node (sink) and no longer sends causal edges to upstream variables.

---

## 5. : LiNGAM

## 5.1

Run separately for each combination of future scenarios:

- `ssp1, ssp2, ssp3, ssp4, ssp5`
- `2040, 2060, 2080, 2100`

,:

- `hist-2020`

---

## 5.2

`SSP × ` , `100` .

:

1. First obtain the **continuous probability** of each sample in this scenario based on the current GWR prediction results.
2. Extract a fixed ratio sub-sample of the scenario sample.
3. The sampling method is fixed as: **sampling without replacement**.
4. The sampling ratio is fixed, for example, `80%` is used uniformly.
5. Unify and standardize 12 driving factors and end nodes.
6. Use `DirectLiNGAM` fitting with prior knowledge matrix.
7. , .

Here is a suggestion for further clarification:

- Implementation note.
- Repeat each scenario `100` times
- ,

### 5.2.1

:

- **Fixed ratio sampling without replacement**

:

1. The results between scenarios are easier to compare horizontally
2. Avoid the accumulation of repeated samples caused by bootstrap sampling with replacement
3. More suitable for subsequent statistics of "causal chain occurrence rate"

Recommended implementation:

- `N`
- Extract `floor(r × N)` samples each time
- Where `r` is a fixed ratio, such as `0.8`
- Use different `random_state` each time

If it is later found that the regional sample composition fluctuates too much, consider again:

- Perform equal-proportion sampling without replacement after internal stratification in `DIV_EN`

But the first version recommends using:

- **Overall fixed proportion sampling without replacement**

If the model output risk is used in future stages, the last node of each input should be:

- `Predicted disaster probability`
- or `Predicted sinkhole susceptibility`

instead of historical observations `Disaster`.

### 5.2.2 Specific input caliber for future risk probability

Before each LiNGAM recalculation, it is recommended that the end node of the future stage be obtained according to the following fixed process:

1. Use GWR to predict and get the raw regression score
2. sigmoid metadata
3. `Predicted sinkhole probability`
4. Use this probability as the only disaster end node in LiNGAM

It is recommended to write a clear rule:

**: GWR raw score robust sigmoid .**

---

## 5.3 Results to be recorded for each run

LiNGAM :

- Adjacency matrix `B`
- Implementation note.
- Whether each edge appears
- Direction of each edge
- Implementation note.
- The effect value of each edge

It is recommended that the output be unified into a long table format, for example:

| scenario | repeat_id | source | target | direct_effect | total_effect | sign |
|---|---:|---|---|---:|---:|---|
| ssp1_2040 | 1 | PR | WTD | 0.21 | 0.29 | + |
| ssp1_2040 | 1 | WTD | Disaster | 0.18 | 0.18 | + |

---

## 6. Stable causal chain screening rules

What will ultimately be displayed in the future is not "all the edges that come out", but a **stable and reasonable causal chain**.

.

### 6.1 Edge stability criteria

One edge `A -> B` is reserved and needs to satisfy:

- Appearance rate `>= 80%`
- Implementation note.
- Symbol consistency rate `>= 80%`
- The median total effect is not close to `0`
- Does not violate prior knowledge

### 6.2 Chain Stability Criteria

,:

- `PR -> WTD -> Disaster`
- `UF -> LAI -> Disaster`
- `DB -> WTD -> Disaster`

is reserved and needs to meet:

- Implementation note.
- The overall direction of the chain conforms to empirical logic
- There is no obvious "abnormal direction" in the chain

For example the following links should be excluded:

- `Disaster -> PR`
- `LAI -> DB -> Disaster`
- `UrbanFrac -> Distance_to_Fault -> Disaster`

### 6.3 Recommended minimum reporting standards

In the final picture and text, it is recommended to only display:

- Single side of occurrence rate `>= 80%`
- Two-hop chain with occurrence rate `>= 80%`
- , `>= 90%`

---

## 7.

## 7.1

"",:

- Three major categories of factors
- 12 variables
- Disaster node
- Explicitly prohibited directions
- The main direction allowed to exist

The purpose of this picture is to tell readers:

**LiNGAM is not searching for edges without constraints, but identifying stable causal structures with the support of empirical knowledge. **

## 7.2

It is recommended to output three types of results.

### A. Single-scenario stable causal network diagram

One picture for each `SSP × `, only showing:

- Appearance rate `>= 80%`
- Does not violate prior knowledge
- Symbol stable

.

### B. Cross-scenario stability heat map

The horizontal axis is `SSP × `, the vertical axis is the edge or chain, and the color indicates the occurrence rate.

This picture is suitable for display:

- Implementation note.
- Which chains only appear later in the high emissions scenario

### C. Only high-frequency main links are retained in the main text and images.

The main text can only retain the most stable and explanatory chain, for example:

- `Climate -> WTD -> Disaster`
- `Anthropogenic activities -> LAI / WTD -> Disaster`
- `Hydrogeology -> Disaster`

---

## 8. Recommended technical implementation path

It is recommended to split the existing code into three modules.

### Module A: Prior knowledge generation

:

- `prior_knowledge_matrix.csv`
- `prior_knowledge_edges_allowed.csv`
- `prior_knowledge_edges_forbidden.csv`

### Module B: Scenario Repetition LiNGAM

Input:

- Sample data of a certain `SSP × `
- Prior knowledge matrix

:

- `edge_records_long.csv`
- `adjacency_repeat_*.csv`
- `total_effect_repeat_*.csv`

### Module C: Summary and Drawing of Stable Chains

:

- `stable_edges_freq_ge_0.8.csv`
- `stable_chains_freq_ge_0.8.csv`
- `scenario_stable_network.svg`
- `cross_scenario_chain_heatmap.svg`

---

## 9.

In order to avoid confusion in method caliber, it is recommended that the paper clearly states:

> We adopted a two-step LiNGAM-based causal attribution framework.
> First, prior knowledge was used to constrain physically implausible directions among the 12 drivers and the sinkhole disaster node.
> Second, for each future scenario, LiNGAM was repeatedly fitted on resampled datasets, and only causal links with occurrence frequencies of at least 80% and without obvious physical inconsistencies were retained.

:

> This study adopted the "two-step LiNGAM causal attribution framework". First, the unreasonable causal directions between the 12 driving factors and the ground collapse disaster are constrained based on empirical knowledge; second, repeated sampling and LiNGAM fitting are performed for each future scenario, and only stable causal chains with an occurrence rate of no less than 80% and no obvious physical logic problems are retained.

---

## 10.

Before formal coding, it is recommended to confirm the following two points first.

### A.

It is now recommended to fix it directly and no longer as an item to be confirmed:

- Use **GWR to output the continuous risk probability after sigmoid mapping**

Not used:

- Binary label replacement value
- Jenks and other graded risk levels

And the existing `transform_metadata` in the training phase must be reused.

### B.

,:

- Using **fixed ratio sampling without replacement**

Recommended default value:

- Sampling ratio `r = 0.8`
- Repeat each scenario `100` times

---

## 11. A simplified version of this framework summarized in one sentence

**Empirical knowledge limits the acceptable causal direction. LiNGAM repeatedly identifies the causal structure under each future scenario, and finally reports only stable causal chains with an occurrence rate of no less than 80% and no obvious logical problems. **
