# The second conclusion is attribution analysis and drawing plan

## 1.

It is not recommended to follow the organizational method of "each picture tells a little about each picture" for this part of the picture. Instead, it should be organized directly into **two groups of pictures** around two points of conclusions.
Each group of pictures contains at least `2` sub-pictures, preferably `2-3` sub-pictures, to form a complete evidence chain.

:

### Sub-Conclusion 1

**Ground collapse risk is not determined by a single variable, but is driven by hydrogeological background, climate forcing and human activities, some of which have clear intermediary structures. **

:

- Implementation note.
- Among the three major categories of factors, which ones have direct effects and which ones have more indirect effects through intermediary variables?

### Sub-Conclusion 2

**,, `SSP × ` ;,.**

:

- Which edges or chains appear repeatedly in future scenarios?
- Which paths are "stable" and which are "situation sensitive"

---

## 2. Group Picture 1: Current Mechanism Attribution Picture Group

This set of figures is used to support **sub-conclusion 1**.

`3` .

### Figure 1a: Prior knowledge constraint framework diagram

#### Intention

Shows the theoretical structure between 12 driving factors and end nodes, not a data result diagram, but a method framework diagram.

#### Processing step.

- Hierarchical directed graph
- Implementation note.

Suggested layering:

1. `Hydrogeology`
   - `DK`
   - `DB`
   - `DF`
   - `HDS`
2. `Climate change`
   - `PR`
   - `TAS`
   - `HUSS`
3. `Anthropogenic activities / mediators`
   - `PT`
   - `UF`
   - `IP`
   - `WTD`
   - `LAI`
4. `Predicted sinkhole probability` or historical stage `Disaster`

#### Processing step.

- You’re not letting LiNGAM search for edges without restraint
- You first used your experience and knowledge to rule out obvious wrong directions.
- The real result is identified within the constrained space

#### The value of this picture

.
It does not prove the result itself, but it can resolve the question of "whether the direction of cause and effect is random" in advance.

---

### 1b: LiNGAM

#### Intention

Shows the total effect strength of 12 variables on the end node, colored by three major categories.

#### Processing step.

- Horizontal bar chart
- Sort by `|total effect|` from large to small
- Bar colors correspond to three categories:
  - Implementation note.
  - HUMAN ACTIVITIES
  - Implementation note.

#### Processing step.

- `|Total effect on sinkhole probability|`

#### Suggested vertical axis

- 12 variable abbreviations or full names

#### Processing step.

- The three major categories of factors have all entered the effective driver set
- ,
- You can intuitively see which variables are more "directly controlled" and which are just secondary variables

#### Connection with existing content

This picture can be used:

- [lingam_division.ipynb](/path/to/sinkhole-risk-china/code/4_attribution/national_fig_lingam/lingam_division.ipynb)

, `Disaster` ,:

- Historical stage: `Disaster`
- Future stage: `Predicted sinkhole probability`

---

### Figure 1c: Path decomposition diagram from three major categories of factors to the end node

#### Intention

This picture no longer stops at "how important a single variable is", but shows:

- Which variables directly point to risk probability
- Which variables mainly work indirectly through intermediary variables such as `WTD` / `LAI`

#### Processing step.

Priority recommendation is to choose one of the two:

- Implementation note.
- Simplified DAG path diagram

If you are pursuing the presentation of your paper, give priority to:

- ``

:

1. 12 variables
2. Implementation note.
3. `Sinkhole probability`

#### Processing step.

- The hydrogeological background is more reflected in "background constraints + direct effects"
- Climate change is transmitted more through intermediary paths such as `WTD`
- Human activities have both direct paths and indirect paths through ecological/hydrological conditions

#### Connection with existing content

:

- [LiNGAM_sankey_updated_12vars_equalheight_v2.ipynb](/path/to/sinkhole-risk-china/code/4_attribution/national_fig_lingam/LiNGAM_sankey_updated_12vars_equalheight_v2.ipynb)

:

- `Predicted sinkhole probability`

#### The necessity of this picture

Without this diagram, Figure 1b can only illustrate "who matters";
,",".

---

## 3. Picture Group 2: Stable Causal Chain Picture Group of Future Scenarios

This set of figures is used to support **sub-conclusion 2**.

`3` .

### Figure 2a: Heat map of frequency of stable causal edges

#### Intention

After repeatedly sampling and running LiNGAM repeatedly for each `SSP × ` combination, count the frequency of occurrence of each candidate edge.

#### Processing step.

- Heat map

#### Processing step.

- Row: candidate edge, e.g.
  - `PR -> WTD`
  - `WTD -> Risk`
  - `UF -> LAI`
  - `DB -> Risk`
- :,
  - `ssp1-2040`
  - `ssp1-2060`
  - ...
  - `ssp5-2100`

#### Processing step.

- The cell color indicates the frequency of occurrence of this edge in `100` repetitions
- You can add a significant threshold mark at the `80%` position

#### Processing step.

- Which edges are stable across scenarios?
- Which edges are only enhanced in high-emission or far-future phases?
- Which edges were originally thought to be important, but in fact the frequency of occurrence is not stable

#### This is the most crucial picture in this set of pictures

Because it directly quantifies the concept of "stability".
This picture is the main evidence picture for the future attribution part.

---

### 2b:

#### Intention

Filter out the edges with frequency `>= 80%` in Figure 2a and combine them into the final retained stable causal chain network.

#### Processing step.

- Simplified DAG diagram
- Implementation note.

Priority suggestions:

- ` DAG `

The reason is that the focus in the future stage is not "flow sense" but "chain structural stability".

#### Suggestions on how to draw

- Only the edges filtered by prior knowledge and appearing with frequency `>= 80%` are displayed
- Implementation note.
- The edge color indicates the plus or minus sign
- Implementation note.

#### Processing step.

- There are not many chains that are truly stable and preserved.
- These chains are organized in a clear way, for example:
  - `Climate -> WTD -> Risk`
  - `Anthropogenic -> LAI / WTD -> Risk`
  - `Hydrogeology -> Risk`

#### Relationship with Figure 2a

- 2a ""
- Figure 2b is the "paper main result network"

,.

---

### Figure 2c: Dominant path intensity changes between scenarios

#### Intention

This picture does not look at "whether there are edges", but looks at:

- Which of the three major types of paths is stronger in different scenarios?
- Whether the dominant mechanism shifts as `SSP` and time advance

#### Processing step.

Choose one:

- Implementation note.
- Stacked Bar Chart

More recommended:

- ``

#### Recommended indicators

First aggregate the edges into three types of path contributions:

- `Hydrogeology -> Risk`
- `Climate / Climate-mediated -> Risk`
- `Anthropogenic / Anthropogenic-mediated -> Risk`

Then calculate for each `SSP × `:

- Average total effect absolute value
- or average weighted strength of stable edges

#### Processing step.

- Future changes are not just "more risks", but "the driving structure is changing"
- For example:
  - In the near future, background constraints will still be the main focus
  - The climate-hydrology chain will strengthen in the far future
  - The human activity chain is stronger under some SSPs

#### The function of this picture

Without Figure 2c, the future part will easily stay at "which edges are repeated";
2c,"".

---

## 4. Thesis descriptions corresponding to the two sets of pictures

In order to avoid the disconnection between the drawing and the conclusion, it is recommended that the two-point conclusion in the paper be directly written in the following way.

### Corresponding expression of sub-conclusion 1

> Sinkhole risk is jointly shaped by hydrogeological background, climate forcing, and anthropogenic disturbance, with part of the influence transmitted through mediating environmental states such as groundwater conditions and vegetation status.

Corresponding evidence:

- Figure 1a: The a priori causal structure is reasonably constrained
- 1b:
- Figure 1c: There is a clear decomposition of intermediary paths

### 2

> Under future scenarios, only a limited subset of causal links remains stable across repeated LiNGAM runs, and the strength of dominant pathways shifts systematically with scenario intensity and time horizon.

Corresponding evidence:

- Figure 2a: Stability quantified by occurrence rate
- Figure 2b: Only high-frequency and reasonable causal chains are finally retained
- Figure 2c: Systematic shifts in dominant path strength under different scenarios

---

## 5. Recommended final picture combination

If you only plan to make two sets of pictures, each set of pictures should be no less than `2` sub-pictures, I suggest you directly make the following set.

### Option A: The most secure version

#### Group Picture 1: Current Mechanism

- `a` Prior knowledge constraint framework diagram
- `b` National scale LiNGAM total effect ranking chart
- `c` Path decomposition diagram from three major categories of factors to risk probability

#### Group Picture 2: Future Stability Mechanism
Group Figure 2 should summarize the future stability mechanism.

- `a` Stable causal edge frequency heat map
- `b` High-frequency stable causal chain DAG diagram
- `c` Line chart of strength changes of the three major categories of dominant paths

This is the most recommended version.
The logic is complete, and the responsibilities of each picture are clear and not repeated.

---

## 6. If you want to compress your workload

, `2` .

### Lightweight version picture 1

- `a` National scale LiNGAM total effect ranking chart
- `b` Three-row Sankey Path Map

### Lightweight version picture 2

- `a` Stable causal edge frequency heat map
- `b` High-frequency stable causal chain DAG diagram

,:

- Without the method constraint diagram, it is easier for reviewers to ask how the prior is determined.
- ,"",""

---

## 7. At this stage, the most recommended picture should be implemented first

If arranged in order of implementation, I recommend doing the following four pictures first:

1. 1b: LiNGAM
2. Figure 1c: Three-column Sankey Path Diagram
3. Figure 2a: Heat map of frequency of stable causal edges
4. Figure 2b: High-frequency stable causal chain DAG diagram

:

- These four pictures are closest to your existing code base
- They are enough to form the main picture of the paper in two sets of `2` sub-pictures.
- , 1a 2c,

---

## 8. One sentence conclusion

The two sets of graphs that are most suitable for you are not "one for relevance and one for importance", but:

- **Answer to group picture 1: How current risks are driven by three major categories of factors**
- **Answer to Group 2: Which causal chains are stable in the future and how they change with scenarios**

After being organized in this way, the picture and the second conclusion will directly correspond one to one, and the problem of "a lot of pictures are made, but they cannot support the core conclusion" will not occur.

---

## 9. Catalog and product organization suggestions

If you want to actually run this process later, it is recommended to organize it according to the following logic under `code/4_attribution`.

### 9.1 Suggested script directory

- `pre_attribution/`
  - Responsible for configuration, variable mapping, prior knowledge matrix, and sample preparation
- `post_attribution/`
  - Responsible for summary, stable edge filtering, path aggregation, and drawing

### 9.2 Suggested output directory

It is recommended to uniformly output to:

- `outputs/attribution/`

It is broken down into:

- `outputs/attribution/common/`
- `outputs/attribution/current/`
- `outputs/attribution/future/`
- `outputs/attribution/figures/`

Further suggested structure:

- `outputs/attribution/common/`
  - , ,
- `outputs/attribution/current/`
  - Put the current stage LiNGAM results
- `outputs/attribution/future/{ssp}/{year}/`
  - All results of repeating LiNGAM in single placement scenario
- `outputs/attribution/figures/group1_current/`
  - Post group picture 1
- `outputs/attribution/figures/group2_future/`
  - Group picture two

### 9.3

The style of this part of the picture should not be determined by each script independently, but should follow the same set of drawing specifications.

#### Output format

Unified requirements:

- Save only **group pictures**
- Do not save individual subgraphs
- The output format is unified as:
  - `SVG`

It is not recommended to export at the same time:

- `PNG`
- `PDF`

.

It is recommended that the file name of the group picture be concise and fixed, for example:

- `group1_current_attribution.svg`
- `group2_future_stable_paths.svg`

#### Processing step.

Unified requirements:

- Implementation note.
- Do not use transparent bottom
- ,

It is recommended to set it explicitly when exporting:

- `figure.facecolor = "#FFFFFF"`
- `axes.facecolor = "#FFFFFF"`

#### Font specification

Unified requirements:

- All text used:
  - `Times New Roman`
- :
  - `9 pt`

The scope of application includes:

- Axis title
- Coordinate axis scale
- Legend
- Node label
- Heat map tag
- Panel mark `a / b / c`

Unless there is serious overlap between individual graphics, do not change the font size level at will.

#### Nature

The "Nature style" here should be understood as:

- Implementation note.
- Restraint
- Implementation note.
- Implementation note.

Therefore, the unified requirements are:

- Does not use 3D effects
- Implementation note.
- Do not use decorative backgrounds
- Implementation note.
- Do not use highly saturated fluorescent colors
- Implementation note.
- Implementation note.

Suggested graphic features:

- Implementation note.
- The coordinate axis is simple
- The grid lines are very light or only necessary guide lines are retained
- ,
- The number of colors in each picture is controlled

#### Morandi color scheme

All pictures use the Morandi color system with low saturation and grayscale. The overall style should be close to the restrained color matching common in high-level journals.

It is recommended to fix a set of main colors instead of re-selecting colors for each picture.

Suggestions for grouping main colors:

- `Hydrogeology`
  - `#7E8FA3`
- `Climate`
  - `#9AAF9A`
- `Anthropogenic`
  - `#C69074`
- `Target / Risk`
  - `#8D6E63`

:

- white background
  - `#FFFFFF`
- Light gray grid
  - `#E6E2DD`
- Implementation note.
  - `#6F6A64`
- axis gray
  - `#8A857F`

,:

- Positive effect
  - `#C69074`
- Negative effect
  - `#8CA0B3`

Note:

- Positive and negative colors must also be kept low in saturation
- ,

#### Heatmap and Continuous Ribbon Specifications

It is not recommended to use rainbow colors for frequency heat maps like Figure 2a.

Suggestion:

- Implementation note.
- For example:
  - `#FFFFFF -> #D8E0E5 -> #AEBCCA -> #7E8FA3`

If you need to emphasize the threshold `0.8`, it is recommended:

- Add value to cell
- `0.8`

.

#### Line, border and legend specifications

Recommend unification:

- Main line width:
  - `0.8 - 1.2 pt`
- Reference grid lines:
  - `0.4 - 0.6 pt`
- Node border:
  - `0.8 pt`

Unified requirements for legends:

- The legend only retains necessary items
- The legend text is `9 pt`
- Implementation note.
- The legend does not emphasize the border, or only uses a very light border
- The legend should be close to the main image without large white spaces

#### SVG

Unified requirements:

- Keep the text as editable as possible
- Do not convert all text into paths
- Avoid transparency masks and complex filters

Reason:

- Implementation note.
- Illustrator / Inkscape / Affinity Designer

#### Principles of group picture output

Since you have explicitly requested that only the group images be saved, subsequent drawing scripts should directly generate the combined layout instead of exporting each sub-image first and then splicing them together.

Suggestion:

- Implementation note.
- `matplotlib axes` `svg`
- In the end, only the image assembly script executes the saving

---

## 10.

If the column names are not unified first, each subsequent script will repeat the mapping, and it will definitely be chaotic in the end.

### 10.1

It is recommended that the following abbreviations be used uniformly within the entire attribution process:

| | | |
|---|---|---|
| `DK` | `Distance_to_karst` | `Hydrogeology` |
| `DB` | `Depth_to_Bedrock` | `Hydrogeology` |
| `DF` | `Distance_to_Fault_m` | `Hydrogeology` |
| `HDS` | `HDS` | `Hydrogeology` |
| `PR` | `Precip` | `Climate` |
| `TAS` | `Tas` | `Climate` |
| `HUSS` | `Huss` | `Climate` |
| `PT` | `PopTotal` | `Anthropogenic` |
| `UF` | `UrbanFrac` | `Anthropogenic` |
| `IP` | `ImperviousIndex` | `Anthropogenic` |
| `LAI` | `LAI` | `Anthropogenic` |
| `WTD` | `WTD` | `Anthropogenic` |
| `RISK` | `Predicted sinkhole probability` | `Target` |

### 10.2

It is recommended that all intermediate tables retain at least these grouping fields:

- `node`
- `node_full_name`
- `group_lv1`
- `group_lv2`

Among them:

- `group_lv1`
  - `Hydrogeology`
  - `Climate`
  - `Anthropogenic`
  - `Target`
- `group_lv2`
  - can be more subdivided, for example:
    - `Static geology`
    - `Climate forcing`
    - `Human pressure`
    - `Environmental mediator`
    - `Target`

### 10.3 Scenario standard fields

All tables will be retained in the future:

- `scenario_id`
- `ssp`
- `year`
- `repeat_id`

Suggestion:

- `scenario_id`
  - For example `ssp1_2040`
  - For example `ssp5_2100`

### 10.4 Sample standard fields

All sample-level tables are retained uniformly:

- `sample_id`
- `Longitude`
- `Latitude`
- `ADCODE99`
- `NAME_EN_JX`

If there is no administrative district information at some stages, it is also recommended to retain at least:

- `sample_id`

### 10.5

Don’t confuse the historical stage with the future stage.

Suggested fixation:

- True label in historical stage:
  - `disaster_observed`
- GWR raw output:
  - `gwr_raw_score`
- Continuous probability after sigmoid mapping:
  - `risk_probability`

LiNGAM :

- `risk_probability`

---

## 11.

Here, the intermediate tables required for the entire process are directly determined.
The stable causal chains are then used for downstream visualization.

It is recommended to organize it into four categories: "public table, current stage table, future stage table, drawing summary table".

### 11.1 Public table 1: variable dictionary table

:

- `variable_dictionary.csv`

Function:

- Unique variable mapping table for the whole process
- ,

Suggested listing:

- `node`
- `node_full_name`
- `source_column`
- `group_lv1`
- `group_lv2`
- `is_target`
- `is_static`
- `is_dynamic`
- `plot_order`
- `node_order`
- `color_hex`

Description:

- `source_column`
  - CSV
- `plot_order`
  - Control the display order in bar charts, heat maps, and DAG charts

---

### 11.2 2:

:

- `prior_knowledge_edges.csv`

Function:

- Directly record the allowed/forbidden relationship between each pair of nodes
- Easier to manually check than a simple matrix

Suggested listing:

- `source`
- `target`
- `constraint_type`
- `constraint_value`
- `reason`
- `source_group`
- `target_group`

Recommended value:

- `constraint_type`
  - `forbidden`
  - `allowed`
  - `unknown`
- `constraint_value`
  - `-1`
  - `1`
  - `0`

Description:

- It is recommended to follow the three-state semantics you have set in the framework document:
  - `forbidden = -1`
  - `allowed = 1`
  - `unknown = 0`

---

### 11.3 Public Table 3: Priori Knowledge Matrix Table

:

- `prior_knowledge_matrix.csv`

Function:

- Feed the matrix directly to `DirectLiNGAM`

Suggested format:

- Implementation note.
- The cell is `-1 / 0 / 1`

Additional suggestions are to generate a metadata file:

- `prior_knowledge_matrix_nodes.csv`

List name:

- `node`
- `matrix_order`

Reason:

- You must know the order of rows and columns when reading the matrix later

---

### 11.4

:

- `current_samples_for_lingam.csv`

Function:

- LiNGAM

Suggested listing:

- `sample_id`
- `ADCODE99`
- `NAME_EN_JX`
- `Longitude`
- `Latitude`
- `DK`
- `DB`
- `DF`
- `HDS`
- `PR`
- `TAS`
- `HUSS`
- `PT`
- `UF`
- `IP`
- `LAI`
- `WTD`
- `disaster_observed`

Description:

- If the current stage is historical observation attribution, keep `disaster_observed`
- If the current stage also adopts the probabilistic approach, add another column:
  - `risk_probability`

---

### 11.5 LiNGAM

:

- `current_lingam_edges_long.csv`

Function:

- One of the core sources of Figure 1b and Figure 1c

Suggested listing:

- `source`
- `target`
- `source_group`
- `target_group`
- `edge_sign`
- `edge_weight`
- `edge_weight_abs`
- `is_direct_to_target`
- `is_mediator_edge`
- `is_selected_for_main_figure`

Description:

- `edge_sign`
  - `positive`
  - `negative`
- `edge_weight`
  - Original coefficient
- `edge_weight_abs`
  - Implementation note.

---

### 11.6 Current stage total effect table

:

- `current_total_effects_to_target.csv`

Function:

- 1b LiNGAM

Suggested listing:

- `node`
- `node_full_name`
- `group_lv1`
- `group_lv2`
- `total_effect`
- `total_effect_abs`
- `effect_sign`
- `rank_abs`
- `selected_topk`

Description:

- `effect_sign`
  - `positive`
  - `negative`

---

### 11.7 Current stage path summary table

:

- `current_path_summary.csv`

Function:

- Path decomposition, Sankey diagram or simplified DAG diagram input for Figure 1c

Suggested listing:

- `path_id`
- `path_str`
- `source_root`
- `mediator`
- `target`
- `path_length`
- `path_sign`
- `path_effect`
- `path_effect_abs`
- `source_group`
- `mediator_group`
- `target_group`
- `is_main_path`

Example:

- `PR->WTD->RISK`
- `UF->LAI->RISK`
- `DB->RISK`

---

### 11.8 Sample table for future phases

:

- `future_samples_for_lingam.csv`

It is recommended to store it in:

- `outputs/attribution/future/{ssp}/{year}/future_samples_for_lingam.csv`

Function:

- LiNGAM

Suggested listing:

- `sample_id`
- `ADCODE99`
- `NAME_EN_JX`
- `Longitude`
- `Latitude`
- `scenario_id`
- `ssp`
- `year`
- `DK`
- `DB`
- `DF`
- `HDS`
- `PR`
- `TAS`
- `HUSS`
- `PT`
- `UF`
- `IP`
- `LAI`
- `WTD`
- `gwr_raw_score`
- `risk_probability`

Description:

- `risk_probability` `transform_metadata`
- LiNGAM

---

### 11.9 Single-repeat LiNGAM edge record long table

:

- `edge_records_long.csv`

Function:

- Implementation note.
- All frequency statistics, heat maps, and stable chain extraction come from here.

Suggested listing:

- `scenario_id`
- `ssp`
- `year`
- `repeat_id`
- `source`
- `target`
- `source_group`
- `target_group`
- `edge_present`
- `edge_sign`
- `edge_weight`
- `edge_weight_abs`
- `target_is_risk`
- `passes_prior_check`

Description:

- `edge_present`
  - `1`
- ""
- 0,

---

### 11.10 Single repetition total effect table

:

- `total_effect_records_long.csv`

Function:

- Used for subsequent work on "how path intensity changes with scenarios"

Suggested listing:

- `scenario_id`
- `ssp`
- `year`
- `repeat_id`
- `node`
- `group_lv1`
- `group_lv2`
- `total_effect`
- `total_effect_abs`
- `effect_sign`

---

### 11.11 Edge frequency summary table

:

- `edge_frequency_summary.csv`

Function:

- Direct input of Figure 2a
- Screening basis of Figure 2b

Suggested listing:

- `scenario_id`
- `ssp`
- `year`
- `source`
- `target`
- `source_group`
- `target_group`
- `n_repeats`
- `n_present`
- `freq_present`
- `freq_positive`
- `freq_negative`
- `mean_weight`
- `mean_weight_abs`
- `median_weight`
- `sd_weight`
- `passes_freq_threshold`
- `passes_sign_stability`
- `is_stable_edge`

Suggested rules:

- `passes_freq_threshold`
  - `freq_present >= 0.8`
- `passes_sign_stability`
  - One side of the plus or minus sign accounts for an absolute majority, for example `>= 0.8`
- `is_stable_edge`
  - Satisfies the first two at the same time

---

### 11.12 Stable edge matrix wide table

:

- `edge_frequency_heatmap_matrix.csv`

Function:

- Figure 2a Direct reading of heat map

Suggested format:

- Row: `edge_label`
- Column: Each `scenario_id`

Suggested listing examples:

- `edge_label`
- `ssp1_2040`
- `ssp1_2060`
- `ssp1_2080`
- `ssp1_2100`
- ...
- `ssp5_2100`

`edge_label` It is recommended to unify as:

- `PR -> WTD`
- `WTD -> RISK`
- `UF -> LAI`

---

### 11.13

:

- `stable_path_summary.csv`

Function:

- Core source of Figure 2b and Figure 2c

Suggested listing:

- `scenario_id`
- `ssp`
- `year`
- `path_id`
- `path_str`
- `path_length`
- `source_root`
- `mediator`
- `target`
- `source_group`
- `mediator_group`
- `target_group`
- `freq_path`
- `mean_path_effect`
- `mean_path_effect_abs`
- `path_sign`
- `is_stable_path`
- `is_main_path`

Example:

- `PR->WTD->RISK`
- `UF->LAI->RISK`
- `DB->RISK`

---

### 11.14 Summary table of three major categories of contributions

:

- `group_path_contribution_summary.csv`

Function:

- 2c

Suggested listing:

- `scenario_id`
- `ssp`
- `year`
- `path_group`
- `contribution_mean`
- `contribution_mean_abs`
- `contribution_share`
- `n_stable_edges`
- `n_stable_paths`

It is recommended that `path_group` be fixed to:

- `Hydrogeology_to_RISK`
- `Climate_to_RISK`
- `Anthropogenic_to_RISK`
- `Climate_mediated_to_RISK`
- `Anthropogenic_mediated_to_RISK`

If you want to avoid being too detailed, the main text images can also be aggregated into three categories:

- `Hydrogeology`
- `Climate`
- `Anthropogenic`

---

## 12. One-to-one correspondence between the graph and the intermediate table

In order to avoid repeated groping when writing graphics scripts later, it is recommended to directly fix it to the mapping below.

### Figure 1a: Prior knowledge constraint framework diagram

:

- `variable_dictionary.csv`
- `prior_knowledge_edges.csv`

### 1b: LiNGAM

:

- `current_total_effects_to_target.csv`
- `variable_dictionary.csv`

### Figure 1c: Path decomposition diagram of the current stage

:

- `current_lingam_edges_long.csv`
- `current_path_summary.csv`
- `variable_dictionary.csv`

### Figure 2a: Heat map of frequency of stable causal edges

:

- `edge_frequency_summary.csv`
- `edge_frequency_heatmap_matrix.csv`

### Figure 2b: High-frequency stable causal chain DAG diagram

:

- `stable_path_summary.csv`
- `edge_frequency_summary.csv`
- `variable_dictionary.csv`

### 2c:

:

- `group_path_contribution_summary.csv`

---

## 13. Script splitting suggestions

Disassemble directly according to the "minimum maintainable unit" here.

### 13.1 Configuration and public modules

#### `pre_attribution/attribution_config.py`

Responsibilities:

- Implementation note.
- Definition `SSP`, year list
- Define the number of repetitions, sampling ratio, and threshold

Recommended parameters for centralized management:

- `SSPS = ["ssp1", "ssp2", "ssp3", "ssp4", "ssp5"]`
- `YEARS = ["2040", "2060", "2080", "2100"]`
- `N_REPEATS = 100`
- `SAMPLE_RATIO = 0.8`
- `FREQ_THRESHOLD = 0.8`
- `SIGN_THRESHOLD = 0.8`

#### `pre_attribution/variable_schema.py`

Responsibilities:

- Generate `variable_dictionary.csv`
- Provides node order and group mapping functions

#### `pre_attribution/prior_knowledge_builder.py`

Responsibilities:

- Generated according to a priori rules:
  - `prior_knowledge_edges.csv`
  - `prior_knowledge_matrix.csv`
  - `prior_knowledge_matrix_nodes.csv`

.

#### `post_attribution/nature_figure_style.py`

Responsibilities:

- Implementation note.
- Output Nature-style shared style interface

Suggestions include:

- `FONT_FAMILY = "Times New Roman"`
- `BASE_FONT_SIZE = 9`
- `FIGURE_FACE = "#FFFFFF"`
- `GROUP_COLORS = {...}`
- `SIGN_COLORS = {...}`
- `GRID_COLOR = "#E6E2DD"`
- `AXIS_COLOR = "#8A857F"`

Once this module is created, all subsequent drawing scripts should import it instead of repeatedly writing colors, fonts and export parameters.

---

### 13.2 Current stage attribution script

#### `pre_attribution/build_current_samples_for_lingam.py`

Responsibilities:

- Implementation note.
- Only 12 variables and target columns are retained
- Output `current_samples_for_lingam.csv`

#### `post_attribution/run_current_lingam.py`

Responsibilities:

- :
  - `current_samples_for_lingam.csv`
  - `prior_knowledge_matrix.csv`
- Fit the current stage LiNGAM once
- :
  - `current_lingam_edges_long.csv`
  - `current_total_effects_to_target.csv`

#### `post_attribution/summarize_current_paths.py`

Responsibilities:

- Extract the main path from the edge results of the current stage
- Output `current_path_summary.csv`

---

### 13.3 Preparing scripts for future phases

#### `pre_attribution/build_future_samples_for_lingam.py`

Responsibilities:

- Read the future driver table for each `SSP × `
- GWR
- Use training period metadata to do sigmoid mapping
- Generates each scenario:
  - `future_samples_for_lingam.csv`

:

- [Train_National_GWR_classification.ipynb](/path/to/sinkhole-risk-china/code/3_gwr_model_train/national/GWR/Train_National_GWR_classification.ipynb)
- [gwr_sigmoid_utils.py](/path/to/sinkhole-risk-china/code/mgtwr/gwr_sigmoid_utils.py)

---

### 13.4 Repeat LiNGAM main script in future phases

#### `post_attribution/run_future_lingam_repeats.py`

Responsibilities:

- For a single `scenario_id`:
  - Implementation note.
  - Repeat LiNGAM
  - Implementation note.

Input:

- `future_samples_for_lingam.csv`
- `prior_knowledge_matrix.csv`

:

- `edge_records_long.csv`
- `total_effect_records_long.csv`

Recommended parameters:

- `--ssp`
- `--year`
- `--n_repeats`
- `--sample_ratio`
- `--random_state`

---

### 13.5 Future phase summary script

#### `post_attribution/summarize_edge_frequency.py`

Responsibilities:

- Calculated from `edge_records_long.csv`:
  - `edge_frequency_summary.csv`
  - `edge_frequency_heatmap_matrix.csv`

#### `post_attribution/summarize_stable_paths.py`

Responsibilities:

- Extract stable paths from stable edges
- :
  - `stable_path_summary.csv`
  - `group_path_contribution_summary.csv`

---

### 13.6 Drawing script

#### `post_attribution/plot_prior_knowledge_framework.py`

correspond:

- Figure 1a

#### `post_attribution/plot_current_total_effects.py`

correspond:

- 1b

#### `post_attribution/plot_current_path_sankey.py`

correspond:

- Figure 1c

#### `post_attribution/plot_edge_frequency_heatmap.py`

correspond:

- 2a

#### `post_attribution/plot_stable_future_dag.py`

correspond:

- Figure 2b

#### `post_attribution/plot_group_path_shift.py`

correspond:

- Figure 2c

---

## 14. Recommended execution sequence

Don’t write all the graph scripts in the first place.
The correct order should be to run through the "table" first, and then make the diagram.

Suggested order:

1. `variable_schema.py`
2. `prior_knowledge_builder.py`
3. `build_current_samples_for_lingam.py`
4. `run_current_lingam.py`
5. `summarize_current_paths.py`
6. `plot_current_total_effects.py`
7. `plot_current_path_sankey.py`
8. `build_future_samples_for_lingam.py`
9. `run_future_lingam_repeats.py`
10. `summarize_edge_frequency.py`
11. `summarize_stable_paths.py`
12. `plot_edge_frequency_heatmap.py`
13. `plot_stable_future_dag.py`
14. `plot_group_path_shift.py`

The reason for this is simple:

- The current stage is run through first, and group picture 1 can be produced first.
- Repeating LiNGAM in future stages is the most time-consuming and most prone to problems
- After the equilateral frequency and stable path tables are stabilized, do the second diagram with minimal rework.

---

## 15. Minimum implementable version

, `6` .

### Processing step.

- `pre_attribution/variable_schema.py`
- `pre_attribution/prior_knowledge_builder.py`
- `post_attribution/run_current_lingam.py`
- `post_attribution/run_future_lingam_repeats.py`
- `post_attribution/summarize_edge_frequency.py`
- `post_attribution/summarize_stable_paths.py`

### The reason for doing this

Once these `6` scripts are completed, you will have:

- Data required for current total effect ranking plot
- Implementation note.
- Data required for future stable edge heat maps
- Data required for the future stable chain DAG graph

In other words, the skeletons of the two main pictures are already complete.

---

## 16.

,"",:

- **Run out the standard intermediate table first**
- ****

is the principle.

,,.
